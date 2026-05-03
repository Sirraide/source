/* eslint-disable curly */

// TextMate grammars being JSON means that you end up with absurd amounts
// of code duplication for things like identifier patterns; this is a massive
// pain, so instead we have a JS file that generates the grammar.
let ident = '[a-zA-Z_][a-zA-Z_0-9]*';

function captures(arr: string[], start_at: number = 1): object {
    let obj: Record<string, object> = {};
    let i = start_at;
    for (let entry of arr) obj[String(i++)] = { "name": entry };
    return obj;
}

function rep<T>(arr: T[], n: number): T[] {
    let acc: T[] = [];
    for (let i = 0; i < n; i++) acc = acc.concat(arr);
    return acc;
}

let grammar = {
    "scopeName": "source.src_lang",
    "patterns": [
        {
            "begin": "// ([IRVX])[\\s\\[]",
            "end": "$",
            "beginCaptures": captures(["fchk.directive"]),
            "name": "comment.line.fchk",
            "patterns": [
                {
                    "name": "fchk.program",
                    "match": "%srcc"
                },
                {
                    "name": "fchk.variable",
                    "match": "(?<=\\s)%[a-zA-Z0-9_]+"
                },
                {
                    "match": "(-+[a-zA-Z0-9_]*)(=)([^ ]*)",
                    "captures": captures(["fchk.arg", "fchk.eq", "fchk.value"])
                },
                {
                    "match": "(-+[a-zA-Z0-9_]*)(?:\\s+([^-% ][^ ]*))?",
                    "captures": captures(["fchk.arg", "fchk.value"])
                }
            ]
        },
        {
            "begin": "//[A-Z]* ([+*p!bdeu]|re\\*|re\\+)[\\s\\[]",
            "end": "$",
            "beginCaptures": captures(["fchk.directive"]),
            "name": "comment.line.fchk"
        },
        {
            "match": "(//)\\s*(expected-no-diagnostics)$",
            "captures": captures(["srcc.verify.diag_text", "srcc.verify.directive"])
        },
        {
            "begin": "(//)\\s*(expected-(?:error|note|warning))(@([+-]?\\d+|\\*))?\\s*(\\s\\d+)?\\s*(:)\\s*(.*)",
            "end": "$",
            "beginCaptures": captures([
                "srcc.verify.diag_text",
                "srcc.verify.directive",
                "srcc.verify.operator",
                "srcc.verify.line",
                "srcc.verify.count",
                "srcc.verify.operator",
                "srcc.verify.diag_text",
            ]),
            "name": "comment.line"
        },
        {
            "begin": "//\\s*?(?=expected)",
            "end": "$",
            "name": "comment.line",
            "beginCaptures": captures(["srcc.verify.diag_text"], 0),
            "patterns": [1, 2, 3, 4, 5].map(n => ({
                "match": `(expected-(?:error|note|warning))\\s*(\\s\\d+)?\\s*(${'\\('.repeat(n)})([^\n]*?)(${'\\)'.repeat(n)})`,
                "captures": captures([
                    "srcc.verify.directive",
                    "srcc.verify.count",
                    "srcc.verify.operator",
                    "srcc.verify.diag_text",
                    "srcc.verify.operator",
                ])
            }))
        },
        {
            "begin": "//",
            "end": "$",
            "name": "comment.line"
        },
        {
            "name": "string.quoted.single",
            "match": "'[^']*'"
        },
        {
            "name": "string.quoted.double",
            "begin": "\"",
            "end": "\"",
            "patterns": [
                {
                    "name": "constant.character.escape",
                    "match": "\\\\."
                }
            ]
        },
        {
            "match": `(import)\\s+((?:<[^>]+>(,?)\\s*)+)\\s+(as)\\s+(${ident}|\\*)`,
            "captures": captures([
                "keyword.other",
                "string.other",
                "keyword.other",
                "keyword.other",
                "entity.name.module",
            ])
        },
        {
            "match": `(import)\\s+(${ident})(?:\\s+(as)\\s+(${ident}|\\*))?`,
            "captures": captures([
                "keyword.other",
                "entity.name.module",
                "keyword.other",
                "entity.name.module",
            ])
        },
        {
            "match": "\\b(pragma)\\b\\s+\\b(include)\\b",
            "captures": captures(["keyword.other", "entity.name.pragma"])
        },
        {
            "match": `\\b(program|module|__srcc_ser_module__)\\b\\s+\\b(${ident})\\b`,
            "captures": captures(["keyword.other", "entity.name.module"])
        },
        {
            "name": "keyword.other",
            "match": "#(inject|assert|if|else|elif|for|match)\\b"
        },
        {
            "name": "keyword.other",
            "match": "\\b(extern|native|nomangle|varargs|inout|out|copy|alias|and|as!|as|asm|assert|break|continue|delete|defer|do|dynamic|elif|else|enum|eval|export|f32|f64|fallthrough|false|for|goto|if|in|init|inline|is|loop|match|nil|not|or|return|static|struct|then|this|This|true|try|typeof|unreachable|val|var|variant|where|while|with|xor)\\b"
        },
        {
            "match": `\\b(proc)\\s+(?:(${ident})\\s*(::)\\s*)*(${ident})?`,
            "captures": captures([
                "keyword.other",
                "entity.name.type",
                "keyword.other",
                "entity.name.function",
            ])
        },
        {
            "name": "keyword.other",
            "match": "\\bproc\\b"
        },
        {
            "match": `\\b(struct)\\b\\s+\\b(${ident})\\b`,
            "captures": captures(["keyword.other","entity.name.type"])
        },
        {
            "name": "stmt.macro",
            "begin": `(#${ident}|\\$)\\s*(\\()`,
            "end": "\\)",
            "beginCaptures": captures(["entity.name.macro", "entity.name.macro"]),
            "endCaptures": captures(["entity.name.macro"], 0),
            "patterns": [{ "include": "$self" }]
        },
        ...["{}", "()"].map(chars => ({
            "name": "stmt.macro",
            "begin": `(quote)\\s*(\\${chars[0]})`,
            "end": `\\${chars[1]}`,
            "beginCaptures": captures(["entity.name.macro", "entity.name.macro"]),
            "endCaptures": captures(["entity.name.macro"], 0),
            "patterns": [{ "include": "$self" }]
        })),
        ...["()", "{}"].map(chars => ({
            "begin": `\\${chars[0]}`,
            "end": `\\${chars[1]}`,
            "beginCaptures": captures(["keyword.other"], 0),
            "endCaptures": captures(["keyword.other"], 0),
            "patterns": [{ "include": "$self" }]
        })),
        {
            "name": "entity.name.type",
            "match": "\\$[a-zA-Z0-9_]+"
        },
        {
            "name": "entity.name.type",
            "match": "\\b(i\\d+|int|type|tree|range|void|noreturn|bool|__srcc_ffi_char|__srcc_ffi_char16|__srcc_ffi_char32|__srcc_ffi_int|__srcc_ffi_long|__srcc_ffi_longdouble|__srcc_ffi_longlong|__srcc_ffi_short|__srcc_ffi_size_t|__srcc_ffi_wchar)\\b"
        },
        // Property access + call.
        //
        // Text mate doesn't support repeating non-capturing groups properly, so instead,
        // we just generate 6 copies for up to 6 levels of nested field access.
        {
            "begin": `(\\.)\\s*(${ident})\\s*(\\()`,
            "end": "\\)",
            "beginCaptures": captures([
                "keyword.other",
                "entity.name.function",
                "keyword.other",
            ]),
            "endCaptures": captures(["keyword.other"], 0),
            "patterns": [{ "include": "$self" }]
        },
        {
            "match": `(\\.)\\s*(${ident})`,
            "captures": captures(["keyword.other", "variable.other.property"]),
        },
        // Call without property access.
        {
            "name": "stmt.call",
            "begin": `\\b(${ident})\\s*(\\()`,
            "end": "\\)",
            "beginCaptures": captures(["entity.name.function", "keyword.other"]),
            "endCaptures": captures(["keyword.other"], 0),
            "patterns": [{ "include": "$self" }]
        },
        {
            "match": `\\b(${ident})\\b(::)`,
            "captures": captures(["entity.name.module", "keyword.other"])
        },
        {
            "name": "keyword.other",
            "match": "[;,\\.=:|?!\\\\/*\\-+&\\^(){}\\[\\]<>%~#\\$]"
        },
        {
            "name": "constant.numeric",
            "match": "\\b(0[xXbBoO][a-fA-F0-9]+|\\d+)\\b"
        },
        {
            "name": "keyword.other",
            "match": "__srcc[a-zA-Z_0-9]+__"
        },
        {
            "name": "keyword.other",
            "match": "\\b_\\b"
        }
    ]
};

console.log(JSON.stringify(grammar, null, 4));
//console.log(JSON.stringify(grammar));