"""
Converts a clause AST (the same dict format used throughout this project)
into a small graph: one node per literal, one node per DISTINCT variable,
one node per DISTINCT constant (within this clause only - constants are
never linked across different clauses/problems, see project notes on why).

Node features are deliberately symbol-name-invariant: they encode
structural properties only (negated? equality? arity? node type?) -
never which specific predicate/constant/variable name is involved.

Edges connect a literal to each of its argument positions (a variable or
constant it references), carrying the argument's position as an edge
feature so that e.g. p(X, Y) and p(Y, X) remain distinguishable.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple

# Node type indices (used for the one-hot slice of the feature vector)
LITERAL, VARIABLE, CONSTANT = 0, 1, 2
N_TYPES = 3

# Per-node feature layout: [type_onehot(3), negated(1), is_equality(1), arity_norm(1)]
NODE_FEATURE_DIM = N_TYPES + 3


@dataclass
class ClauseGraph:
    node_features: List[List[float]] = field(default_factory=list)  # (n_nodes, NODE_FEATURE_DIM)
    edges: List[Tuple[int, int, float]] = field(default_factory=list)  # (node_i, node_j, position_norm)


def _literal_feature(negated: bool, is_equality: bool, arity: int) -> List[float]:
    onehot = [0.0, 0.0, 0.0]
    onehot[LITERAL] = 1.0
    return onehot + [float(negated), float(is_equality), min(arity, 10) / 10.0]


def _var_or_const_feature(is_var: bool) -> List[float]:
    onehot = [0.0, 0.0, 0.0]
    onehot[VARIABLE if is_var else CONSTANT] = 1.0
    return onehot + [0.0, 0.0, 0.0]  # negated/equality/arity don't apply


def _collect_literals(node: dict, negated: bool, out: list):
    """Flatten a clause's top-level disjunction (a chain of BinOp '|') into
    a flat list of (literal_term_dict, negated) pairs. Also handles a
    single-literal clause (no BinOp at all)."""
    t = node["_type"]
    if t == "BinOp" and node["op"] == "|":
        _collect_literals(node["left"], negated, out)
        _collect_literals(node["right"], negated, out)
        return
    if t == "Not":
        _collect_literals(node["formula"], not negated, out)
        return
    if t == "Atom":
        out.append((node["term"], negated))
        return
    # Quant nodes shouldn't appear in clausal data, but if they do, just
    # descend through them (they're all effectively universal in EPR).
    if t == "Quant":
        _collect_literals(node["formula"], negated, out)
        return
    raise ValueError(f"Unexpected node type in clause: {t}")


def build_clause_graph(ast: dict) -> ClauseGraph:
    literals = []  # (term_dict, negated)
    _collect_literals(ast, False, literals)

    graph = ClauseGraph()
    var_node_id = {}   # variable name (LOCAL to this clause) -> node index
    const_node_id = {}  # constant name (LOCAL to this clause) -> node index

    def get_var_node(name: str) -> int:
        if name not in var_node_id:
            var_node_id[name] = len(graph.node_features)
            graph.node_features.append(_var_or_const_feature(is_var=True))
        return var_node_id[name]

    def get_const_node(name: str) -> int:
        if name not in const_node_id:
            const_node_id[name] = len(graph.node_features)
            graph.node_features.append(_var_or_const_feature(is_var=False))
        return const_node_id[name]

    for term, negated in literals:
        is_equality = term["name"] == "=" and len(term["args"]) == 2
        arity = len(term["args"])
        lit_id = len(graph.node_features)
        graph.node_features.append(_literal_feature(negated, is_equality, arity))

        for pos, arg in enumerate(term["args"]):
            pos_norm = pos / max(arity - 1, 1)
            if arg["_type"] == "Var":
                arg_id = get_var_node(arg["name"])
            else:
                # a Term with args==[] (0-arity constant) - EPR guarantees
                # no positive-arity function symbols reach here
                arg_id = get_const_node(arg["name"])
            graph.edges.append((lit_id, arg_id, pos_norm))

    return graph


if __name__ == "__main__":
    # ~p(X, Y) | q(X)  -  the running example from our design discussion
    sample_ast = {
        "_type": "BinOp", "op": "|",
        "left": {"_type": "Not", "formula": {"_type": "Atom", "term":
            {"_type": "Term", "name": "p", "args": [
                {"_type": "Var", "name": "X"}, {"_type": "Var", "name": "Y"}]}}},
        "right": {"_type": "Atom", "term":
            {"_type": "Term", "name": "q", "args": [{"_type": "Var", "name": "X"}]}},
    }
    g = build_clause_graph(sample_ast)
    print(f"Nodes ({len(g.node_features)}):")
    for i, f in enumerate(g.node_features):
        print(f"  {i}: {f}")
    print(f"Edges: {g.edges}")
