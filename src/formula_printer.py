"""
Converts the serialized AST dicts (as written by build_corpus.py's
ast_to_dict) back into readable TPTP-syntax text - the inverse of
tptp_parser.py. Used so retrieval results show an actual formula
instead of a nested JSON blob.

Deliberately does not try to reproduce the exact original source text
byte-for-byte (e.g. original variable order in `!= `A - it normalizes
Not(Atom(=,...)) back into `!=` for readability, and always shows a
minimal-but-correct parenthesization) - the goal is readability and
re-parseability, not a lossless print.
"""

from __future__ import annotations


def format_term(node: dict) -> str:
    if node["_type"] == "Var":
        return node["name"]
    # Term (predicate/function/constant application)
    if not node["args"]:
        return node["name"]
    args = ", ".join(format_term(a) for a in node["args"])
    return f"{node['name']}({args})"


def _is_equality(node: dict) -> bool:
    return node["_type"] == "Atom" and node["term"]["name"] == "=" and len(node["term"]["args"]) == 2


def format_formula(node: dict, _parent_op: str = None) -> str:
    """Top-level formatter - no surrounding parens added to the result itself.
    _parent_op is internal: lets a BinOp know its immediate parent's operator
    so a chain of the SAME operator (e.g. a | b | c | d, very common in wide
    CNF clauses) can be printed flat instead of needlessly re-parenthesizing
    every level - which also avoids blowing the recursion limit on re-parse
    for very wide clauses."""
    t = node["_type"]

    if t == "Atom":
        if _is_equality(node):
            a, b = node["term"]["args"]
            return f"{format_term(a)} = {format_term(b)}"
        return format_term(node["term"])

    if t == "Not":
        inner = node["formula"]
        if _is_equality(inner):
            a, b = inner["term"]["args"]
            return f"{format_term(a)} != {format_term(b)}"
        return f"~{_format_sub(inner)}"

    if t == "BinOp":
        op = node["op"]
        left = _format_sub(node["left"], parent_op=op)
        right = _format_sub(node["right"], parent_op=op)
        return f"{left} {op} {right}"

    if t == "Quant":
        vars_str = ", ".join(node["vars"])
        body = _format_sub(node["formula"])
        return f"{node['kind']} [{vars_str}] : {body}"

    raise ValueError(f"Unknown AST node type: {t}")


def _format_sub(node: dict, parent_op: str = None) -> str:
    """Formats a child formula, adding parens only when actually needed:
    - BinOp/Quant normally need grouping when nested under anything
    - EXCEPT a BinOp whose operator matches its immediate parent's operator
      (a flat chain of the same associative connective) - safe to leave
      flat since our own parser reconstructs the identical left-assoc
      structure on re-parse either way."""
    if node["_type"] == "BinOp" and parent_op is not None and node["op"] == parent_op:
        return format_formula(node, _parent_op=parent_op)
    s = format_formula(node)
    if node["_type"] in ("BinOp", "Quant"):
        return f"({s})"
    return s


if __name__ == "__main__":
    # A few hand-built examples spanning FOF and CNF-style structure
    examples = [
        {"_type": "BinOp", "op": "|",
         "left": {"_type": "Not", "formula": {"_type": "Atom", "term":
             {"_type": "Term", "name": "p", "args": [{"_type": "Term", "name": "a", "args": []}]}}},
         "right": {"_type": "Atom", "term":
             {"_type": "Term", "name": "q", "args": [{"_type": "Var", "name": "X"}]}}},

        {"_type": "Quant", "kind": "!", "vars": ["X"],
         "formula": {"_type": "BinOp", "op": "=>",
             "left": {"_type": "Atom", "term": {"_type": "Term", "name": "p", "args": [{"_type": "Var", "name": "X"}]}},
             "right": {"_type": "Quant", "kind": "?", "vars": ["Y"],
                 "formula": {"_type": "Atom", "term": {"_type": "Term", "name": "q", "args":
                     [{"_type": "Var", "name": "X"}, {"_type": "Var", "name": "Y"}]}}}}},

        {"_type": "Not", "formula": {"_type": "Atom", "term": {"_type": "Term", "name": "=", "args":
            [{"_type": "Term", "name": "a", "args": []}, {"_type": "Term", "name": "b", "args": []}]}}},
    ]
    for ex in examples:
        print(format_formula(ex))