#include <srcc/Core/Constants.hh>
#include <srcc/Core/Utils.hh>
#include <srcc/Frontend/Sema.hh>

using namespace srcc;

auto Sema::LoadModuleFromArchive(
    String logical_name,
    String linkage_name,
    Location import_loc
) -> Ptr<ImportedSourceModuleDecl> {
    if (search_paths.empty()) return ICE(import_loc, "No module search path");

    // Append extension.
    std::string desc_name = std::format("{}.{}", linkage_name, constants::ModuleDescriptionFileExtension);

    // Try to find the module in the search path.
    fs::Path mod_path, desc_path;
    for (auto& base : search_paths) {
        auto combined = fs::Path{base} / desc_name;
        if (fs::File::Exists(combined)) {
            mod_path = fs::Path{base} / std::format("{}.{}", linkage_name, constants::ModuleFileExtension);
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
    return new (*tu) ImportedSourceModuleDecl(
        *curr_scope(),
        logical_name,
        linkage_name,
        tu->save(mod_path.string()),
        import_loc
    );
}

void Sema::LoadModule(
    String logical_name,
    ArrayRef<String> linkage_names,
    Location import_loc,
    bool is_open,
    bool is_cxx_header
) {
    if (is_open) {
        Assert(is_cxx_header);
        if (auto m = ImportCXXHeaders(logical_name, linkage_names, import_loc).get_or_null())
            tu->open_modules.push_back(m);
        return;
    }

    // Clang imports cannot be cached or redefined.
    auto& logical = tu->logical_imports[logical_name];
    if (isa_and_present<ImportedClangModuleDecl>(logical)) {
        Error(import_loc, "Cannot redefine header import '{}'", logical_name);
        Note(logical->location(), "Previous definition was here");
        return;
    }

    if (is_cxx_header) {
        logical = ImportCXXHeaders(logical_name, linkage_names, import_loc).get_or_null();
        return;
    }

    Assert(linkage_names.size() == 1, "Source module imports should consist of a single physical module");
    auto& link = tu->linkage_imports[linkage_names.front()];
    if (not link) {
        logical = link = LoadModuleFromArchive(logical_name, linkage_names.front(), import_loc).get_or_null();
        return;
    }

    // We’ve imported this before; if the logical name isn’t mapped yet, map it
    // to the same module.
    if (not logical) logical = link;

    // Conversely, complain if this name is already mapped to a different module.
    else if (logical != link) {
        Error(import_loc, "Import with name '{}' conflicts with an existing import", logical_name);
        Note(import_loc, "Import here refers to module '{}'", linkage_names.front());
        Note(link->location(), "But import here refers to module '{}'", link->linkage_name);
    }
}

auto TranslationUnit::serialise(bool use_colours) -> std::string {
    Assert(is_module, "Should never be called for programs");
    SmallUnrenderedString out;
    out += std::format("%1(__srcc_ser_module__ %4({}%);%)\n", name);

    // Sort declarations by source location for stability.
    SmallVector<Decl*> decls;
    append_range(decls, exports.decls());
    sort(decls, [](Decl* a, Decl* b) { return a->location() < b->location(); });
    for (auto d : decls) {
        if (auto proc = dyn_cast<ProcDecl>(d)) {
            out += std::format(
                "{}%1(;%)\n",
                proc->proc_type()->print(proc->name, false, proc).str()
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

    return text::RenderColours(use_colours, out.str());
}
