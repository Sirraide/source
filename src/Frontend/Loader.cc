#include <srcc/Frontend/Sema.hh>

using namespace srcc;

auto Sema::LoadModuleFromArchive(
    String logical_name,
    String linkage_name,
    Location import_loc
) -> Ptr<ImportedSourceModuleDecl> {
    if (search_paths.empty()) return ICE(import_loc, "No module search path");

    // Append extension.
    std::string desc_name = std::format("{}.mod.meta", linkage_name);

    // Try to find the module in the search path.
    fs::Path mod_path, desc_path;
    for (auto& base : search_paths) {
        auto combined = fs::Path{base} / desc_name;
        if (fs::File::Exists(combined)) {
            mod_path = fs::Path{base} / std::format("{}.mod", linkage_name);
            desc_path = std::move(combined);
            break;
        }
    }

    // Couldn’t find it :(.
    if (desc_path.empty()) {
        Error(import_loc, "Could not find module '{}'", linkage_name);
        Remark("Search paths:\n  {}", utils::join(search_paths, "\n  "));
        return nullptr;
    }

    // Load the archive.
    auto file = context().try_get_file(desc_path);
    if (not file) return Error(
        import_loc,
        "Could not load module '{}' ({}): {}",
        linkage_name,
        desc_path.string(),
        file.error()
    );

    auto p = Parser::Parse(file.value());
    if (diags().has_error()) return nullptr;

    tempset importing_module = true;
    Assert(curr_scope() == global_scope());
    EnterScope _{*this};
    SmallVector<Stmt*> stmts;
    TranslateStmts(stmts, p->top_level);
    return new (*M) ImportedSourceModuleDecl(
        *curr_scope(),
        logical_name,
        linkage_name,
        M->save(mod_path.string()),
        import_loc
    );
}

void Sema::LoadModule(
    String logical_name,
    String linkage_name,
    Location import_loc,
    bool is_cxx_header
) {
    auto& link = M->linkage_imports[linkage_name];
    auto& logical = M->logical_imports[logical_name];

    // The module has already been imported.
    if (link) {
        // If we’re importing it with a different name, map that name to the same module.
        if (not logical) logical = link;

        // Conversely, complain if this name is already mapped to a different module.
        else if (logical != link) {
            Error(import_loc, "Import with name '{}' conflicts with an existing import", logical_name);
            Note(import_loc, "Import here refers to module '{}'", linkage_name);
            Note(link->location(), "But import here refers to module '{}'", link->linkage_name);
        }

        return;
    }

    // Import the module.
    ModuleDecl* m;
    if (is_cxx_header) m = ImportCXXHeader(logical_name, linkage_name, import_loc).get_or_null();
    else m = LoadModuleFromArchive(logical_name, linkage_name, import_loc).get_or_null();
    link = logical = m;
}

auto TranslationUnit::serialise() -> SmallString<0> {
    Assert(is_module, "Should never be called for programs");
    SmallString<0> out;
    out += std::format("__srcc_ser_module__ {};\n", name);
    for (auto d : exports.decls()) {
        if (auto proc = dyn_cast<ProcDecl>(d)) {
            out += std::format(
                "{};\n",
                text::RenderColours(
                    false,
                    proc->proc_type()->print(proc->name, false, proc).str()
                )
            );
            continue;
        }

        // TODO: I was going to compress the module description, but instead,
        // can we extract doc comments and paste them into this file? That way,
        // we’d basically be generating a header file, and people like to use
        // those for documentation.

        // TODO: For templates, have the parser keep track of the first and
        // last token of the function body and simply write out all the tokens
        // in between.

        d->dump_color();
        Todo("Export this decl");
    }
    return out;
}
