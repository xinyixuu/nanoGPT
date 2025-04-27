# code_highlighter.py
"""
Map each character of a Python source file one-to-one to a single character
representing various facets of the code (token types, nesting, names, etc.).
Literal '\n' characters are kept as '\n'.  Other unmapped chars become
either a space or a WHITESPACE category per mode.

Supported modes:
  general         — broad token types (NAME, NUMBER, OP, etc.)
  exact           — fine-grained token types (EQEQUAL vs EQUAL, etc.)
  keywords        — unique char for each Python keyword (KW_IF, KW_FOR, etc.)
  nesting         — indentation-level per line (LEVEL_0, LEVEL_1, etc.)
  param_nesting   — parentheses-depth across all '('…')' (` ` outside)
  argnum          — argument-index in defs & calls (` ` outside)
  name_kind       — distinguish BUILTIN, FUNC, CLASS, IMPORT, VAR names
  literals        — split INT, FLOAT, HEX, OCT, BIN, plus string flavors
  semantic        — high-level buckets: CONTROL / DECL / LITERAL / NAME / OP / PUNCT / WS / COMMENT
  comments        — binary map: COMMENT vs CODE
  scope           — lexical scope: GLOBAL / FUNCTION / CLASS
  dot_nesting     — attribute-chain depth (obj.field -> depth=1; obj.f.g-> g depth=2)
"""

import argparse
import ast
import builtins
import io
import keyword
import re
import sys
import tokenize
from collections import OrderedDict


def parse_args():
    p = argparse.ArgumentParser(
        description="1:1 char mapping of Python source to various analyses."
    )
    p.add_argument("source", help="Path to the Python .py file.")
    p.add_argument(
        "--mode",
        choices=[
            "general",
            "exact",
            "keywords",
            "nesting",
            "param_nesting",
            "argnum",
            "dot_nesting",
            "name_kind",
            "literals",
            "semantic",
            "comments",
            "scope",
        ],
        default="general",
        help="Which mapping to use.",
    )
    return p.parse_args()


def build_mapping(keys):
    charset = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789")
    if len(keys) > len(charset):
        raise RuntimeError("Too many categories to map!")
    return OrderedDict((k, charset[i]) for i, k in enumerate(keys))


# ─── GENERAL / EXACT / KEYWORDS ────────────────────────────────────────────────
def get_token_category(tok, mode):
    if mode == "keywords":
        if tok.type == tokenize.NAME and tok.string in keyword.kwlist:
            return f"KW_{tok.string.upper()}"
    if mode == "general":
        if tok.type == tokenize.NAME and tok.string in keyword.kwlist:
            return "KEYWORD"
        return tokenize.tok_name[tok.type]
    # exact
    name = tokenize.tok_name.get(tok.exact_type)
    if name and name != "ERRORTOKEN":
        return name
    if tok.type == tokenize.NAME and tok.string in keyword.kwlist:
        return "KEYWORD"
    return tokenize.tok_name[tok.type]


# ─── LITERALS ─────────────────────────────────────────────────────────────────
def get_literal_category(tok):
    s = tok.string
    if tok.type == tokenize.NUMBER:
        l = s.lower()
        if l.startswith("0x"):
            return "HEX"
        if l.startswith("0o"):
            return "OCT"
        if l.startswith("0b"):
            return "BIN"
        if "." in s or "e" in s or "E" in s:
            return "FLOAT"
        return "INT"
    if tok.type == tokenize.STRING:
        m = re.match(r"(?i)(?P<p>[rubfRUBF]*)(?P<q>'''|\"\"\"|'|\")", s)
        pre = m.group("p") or ""
        q = m.group("q")
        qt = "TRIPLE" if len(q) == 3 else ("SINGLE" if q == "'" else "DOUBLE")
        pl = pre.lower()
        if "f" in pl:
            pt = "FSTRING"
        elif "r" in pl:
            pt = "RAW"
        elif "b" in pl:
            pt = "BYTES"
        else:
            pt = "NORMAL"
        if pt == "NORMAL":
            return f"{qt}_STR"
        return f"{pt}_{qt}_STR"
    return None


# ─── SEMANTIC ─────────────────────────────────────────────────────────────────
_control = {
    "if",
    "else",
    "elif",
    "for",
    "while",
    "try",
    "except",
    "finally",
    "with",
    "return",
    "yield",
    "assert",
    "break",
    "continue",
    "raise",
    "lambda",
}
_decl = {"def", "class", "import", "from", "as", "global", "nonlocal"}
_punct = set(",;:.()[]{}")


def get_semantic_category(tok):
    if tok.type == tokenize.COMMENT:
        return "COMMENT"
    if tok.type in (tokenize.NEWLINE, tokenize.NL, tokenize.INDENT, tokenize.DEDENT):
        return "WS"
    if tok.type == tokenize.NAME and tok.string in keyword.kwlist:
        if tok.string in _decl:
            return "DECL"
        if tok.string in _control:
            return "CONTROL"
        return "KEYWORD"
    if tok.type in (tokenize.NUMBER, tokenize.STRING):
        return "LITERAL"
    if tok.type == tokenize.NAME:
        return "NAME"
    if tok.type == tokenize.OP:
        return "PUNCT" if tok.string in _punct else "OP"
    return None


# ─── COMMENTS ─────────────────────────────────────────────────────────────────
def get_comments_category(tok):
    return "COMMENT" if tok.type == tokenize.COMMENT else "CODE"


# ─── NAME_KIND ────────────────────────────────────────────────────────────────
def build_name_kind_info(source):
    tree = ast.parse(source)
    fnames = {
        n.name
        for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    cnames = {n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}
    imps = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                imps.add(a.asname or a.name.split(".")[0])
        elif isinstance(n, ast.ImportFrom):
            for a in n.names:
                imps.add(a.asname or a.name)
    built = set(dir(builtins))
    return fnames, cnames, imps, built


def get_name_kind_category(tok, info):
    fn, cn, im, built = info
    if tok.type != tokenize.NAME or tok.string in keyword.kwlist:
        return None
    s = tok.string
    if s in fn:
        return "FUNC"
    if s in cn:
        return "CLASS"
    if s in im:
        return "IMPORT"
    if s in built:
        return "BUILTIN"
    return "VAR"


# ─── SCOPE ────────────────────────────────────────────────────────────────────
def build_scope_ranges(source):
    tree = ast.parse(source)
    funcs, classes = [], []
    for n in ast.walk(tree):
        if hasattr(n, "lineno") and hasattr(n, "end_lineno"):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                funcs.append((n.lineno, n.col_offset, n.end_lineno, n.end_col_offset))
            if isinstance(n, ast.ClassDef):
                classes.append((n.lineno, n.col_offset, n.end_lineno, n.end_col_offset))
    return funcs, classes


def find_scope(line, col, funcs, classes):
    for sr, sc, er, ec in classes:
        if (line > sr or (line == sr and col >= sc)) and (
            line < er or (line == er and col < ec)
        ):
            return "CLASS"
    for sr, sc, er, ec in funcs:
        if (line > sr or (line == sr and col >= sc)) and (
            line < er or (line == er and col < ec)
        ):
            return "FUNCTION"
    return "GLOBAL"


# ─── NESTING (INDENT) ─────────────────────────────────────────────────────────
def mode_nesting(lines):
    src = "".join(lines)
    toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
    indent_per_line, indent, ln = {}, 0, 1
    for tok in toks:
        r = tok.start[0]
        while ln < r:
            indent_per_line[ln] = indent
            ln += 1
        if tok.type == tokenize.INDENT:
            indent += 1
        elif tok.type == tokenize.DEDENT:
            indent -= 1
        indent_per_line[r] = indent
    total = len(lines)
    while ln <= total:
        indent_per_line[ln] = indent
        ln += 1
    levels = sorted(set(indent_per_line.values()))
    keys = [f"LEVEL_{lvl}" for lvl in levels]
    mapping = build_mapping(keys)

    out = []
    for i, line in enumerate(lines, start=1):
        key = f"LEVEL_{indent_per_line[i]}"
        chm = mapping[key]
        for c in line:
            out.append("\n" if c == "\n" else chm)
    return "".join(out), mapping


# ─── PARAM_PARENS NESTING ─────────────────────────────────────────────────────
def mode_param_nesting(source):
    depth, maxd = 0, 0
    for c in source:
        if c == "(":
            depth += 1
            maxd = max(maxd, depth)
        elif c == ")":
            depth = max(depth - 1, 0)
    keys = [f"LEVEL_{i}" for i in range(1, maxd + 1)]
    mapping = build_mapping(keys)
    out = []
    depth = 0
    for c in source:
        if c == "\n":
            out.append("\n")
        elif c == "(":
            depth += 1
            out.append(mapping[f"LEVEL_{depth}"])
        elif c == ")":
            out.append(mapping[f"LEVEL_{depth}"])
            depth = max(depth - 1, 0)
        else:
            out.append(" " if depth == 0 else mapping[f"LEVEL_{depth}"])
    return "".join(out), mapping


# ─── ARGUMENT INDEX ───────────────────────────────────────────────────────────
def mode_argnum(source):
    lines = source.splitlines(keepends=True)
    grid = [[None] * len(l) for l in lines]
    toks = list(tokenize.generate_tokens(io.StringIO(source).readline))

    contexts = []
    for i, tok in enumerate(toks):
        if tok.type == tokenize.NAME and tok.string in ("def", "class"):
            j = i + 1
            while j < len(toks) and not (
                toks[j].type == tokenize.OP and toks[j].string == "("
            ):
                j += 1
            if j < len(toks):
                contexts.append(j)
    for i, tok in enumerate(toks):
        if (
            tok.type == tokenize.OP
            and tok.string == "("
            and toks[i - 1].type == tokenize.NAME
        ):
            contexts.append(i)

    seen_idx = set()
    for start in contexts:
        if start in seen_idx:
            continue
        seen_idx.add(start)
        depth, arg = 0, 1
        for tok in toks[start:]:
            if tok.type == tokenize.OP and tok.string == "(":
                depth += 1
                continue
            if tok.type == tokenize.OP and tok.string == ")":
                depth -= 1
                if depth == 0:
                    break
                continue
            if tok.type == tokenize.OP and tok.string == "," and depth == 1:
                arg += 1
                continue
            if depth == 1 and tok.string not in ("(", ")", ","):
                (sr, sc), (er, ec) = tok.start, tok.end
                for r in range(sr - 1, er):
                    sc0 = sc if r == sr - 1 else 0
                    ec0 = ec if r == er - 1 else len(lines[r])
                    for c in range(sc0, ec0):
                        grid[r][c] = arg

    args = sorted(
        {
            grid[r][c]
            for r in range(len(grid))
            for c in range(len(grid[r]))
            if grid[r][c]
        }
    )
    keys = [f"ARG_{i}" for i in args]
    mapping = build_mapping(keys)
    out = []
    for r, line in enumerate(lines):
        for c, ch in enumerate(line):
            if ch == "\n":
                out.append("\n")
            else:
                v = grid[r][c]
                out.append(" " if not v else mapping[f"ARG_{v}"])
    return "".join(out), mapping


# ─── DOT_NESTING (ATTRIBUTE CHAIN DEPTH) ───────────────────────────────────────
def mode_dot_nesting(source, lines):
    """
    Track attribute access chains like obj.field1.field2:
    - Mark the base object (name before first dot) at depth=1
    - Each subsequent dot increases depth by 1 and marks the following name
    All other characters map to space, newlines are kept as-is.
    """
    toks = list(tokenize.generate_tokens(io.StringIO(source).readline))
    # grid holds category keys (e.g. "DOT", "DEPTH_1") or None
    grid = [[None] * len(line) for line in lines]

    # Scan tokens for chains
    i = 0
    n = len(toks)
    while i < n - 1:
        tok = toks[i]
        next_tok = toks[i+1]
        # detect start of chain: NAME followed by '.'
        if tok.type == tokenize.NAME and next_tok.string == '.':
            depth = 1
            # mark the base object
            sr, sc = tok.start
            er, ec = tok.end
            for r in range(sr-1, er):
                start_c = sc if r == sr-1 else 0
                end_c   = ec if r == er-1 else len(lines[r])
                for c in range(start_c, end_c):
                    grid[r][c] = f"DEPTH_{depth}"
            # now consume the chain: dot and name pairs
            j = i + 1
            while j < n - 1 and toks[j].string == '.' and toks[j+1].type == tokenize.NAME:
                # increment depth for this dot
                depth += 1
                # mark the dot itself as its own category
                dsr, dsc = toks[j].start
                der, dec = toks[j].end
                for r in range(dsr-1, der):
                    start_c = dsc if r == dsr-1 else 0
                    end_c   = dec if r == der-1 else len(lines[r])
                    for c in range(start_c, end_c):
                        grid[r][c] = "DOT"
                # mark the following name
                nsr, nsc = toks[j+1].start
                ner, nec = toks[j+1].end
                for r in range(nsr-1, ner):
                    start_c = nsc if r == nsr-1 else 0
                    end_c   = nec if r == ner-1 else len(lines[r])
                    for c in range(start_c, end_c):
                        grid[r][c] = f"DEPTH_{depth}"
                j += 2
            i = j
        else:
            i += 1

    # Determine which depths actually occurred
    # collect all depth levels and include DOT category
    found = {cell for row in grid for cell in row if cell}
    depths = sorted(int(x.split("_")[1]) for x in found if x.startswith("DEPTH_"))
    keys = ["DOT"] + [f"DEPTH_{d}" for d in depths]
    mapping = build_mapping(keys)

    # Build output, preserving newlines
    out = []
    for ridx, line in enumerate(lines):
        for cidx, ch in enumerate(line):
            if ch == '\n':
                out.append('\n')
            else:
                cat = grid[ridx][cidx]
                if not cat:
                    out.append(" ")
                else:
                    out.append(mapping[cat])
    return ''.join(out), mapping


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    source = open(args.source, encoding="utf-8").read()
    lines = source.splitlines(keepends=True)
    mode = args.mode

    if mode == "nesting":
        out, mapping = mode_nesting(lines)
    elif mode == "param_nesting":
        out, mapping = mode_param_nesting(source)
    elif mode == "argnum":
        out, mapping = mode_argnum(source)
    elif mode == "dot_nesting":
        out, mapping = mode_dot_nesting(source, lines)
    else:
        toks = list(tokenize.generate_tokens(io.StringIO(source).readline))
        # choose category function
        if mode in ("general", "exact", "keywords"):
            chooser = lambda t: get_token_category(t, mode)
        elif mode == "literals":
            chooser = get_literal_category
        elif mode == "semantic":
            chooser = get_semantic_category
        elif mode == "comments":
            chooser = get_comments_category
        elif mode == "scope":
            funcs, classes = build_scope_ranges(source)
            chooser = lambda t: find_scope(t.start[0], t.start[1], funcs, classes)
        elif mode == "name_kind":
            info = build_name_kind_info(source)
            chooser = lambda t: get_name_kind_category(t, info)
        else:
            raise RuntimeError("Unknown mode")

        skip = {tokenize.NL, tokenize.NEWLINE}
        seen = ["WHITESPACE"]
        for t in toks:
            cat = chooser(t)
            if cat and t.type not in skip and cat not in seen:
                seen.append(cat)
        mapping = build_mapping(seen)

        # stamp grid
        grid = [["WHITESPACE"] * len(l) for l in lines]
        for t in toks:
            cat = chooser(t)
            if not cat or t.type in skip:
                continue
            (sr, sc), (er, ec) = t.start, t.end
            for r in range(sr - 1, er):
                sc0 = sc if r == sr - 1 else 0
                ec0 = ec if r == er - 1 else len(lines[r])
                for c in range(sc0, ec0):
                    grid[r][c] = cat

        out = []
        for r, line in enumerate(lines):
            for c, ch in enumerate(line):
                if ch == "\n":
                    out.append("\n")
                else:
                    out.append(mapping[grid[r][c]])
        out = "".join(out)

    # write & verify
    out_path = args.source + ".mapped"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(out)

    L1, L2 = len(source), len(out)
    if L1 == L2:
        print(f"✅ Wrote {out_path} ({L1} chars).")
    else:
        print(f"⚠️ Length mismatch: {L1} vs {L2}!")

    print("\n# Legend: char → category")
    for k, v in mapping.items():
        print(f"{v} = {k}")


if __name__ == "__main__":
    main()

