"""
A minimal message-passing GNN, implemented directly in numpy (no
autograd/training here - see project notes on why an UNTRAINED,
fixed-random-weight encoder is still a meaningful first thing to try:
a few rounds of message passing already fingerprint graph structure,
even with random weights, related to the Weisfeiler-Lehman test).

Per round:
  1. every edge sends a "message" = ReLU(W_msg @ [neighbor_vec, edge_feat])
  2. every node aggregates (sums) messages from all its edges
  3. every node updates: new_vec = ReLU(W_self @ own_vec + W_agg @ aggregated + bias)

Edges are treated as undirected - a message is sent in both directions
along every (literal, argument) edge.

After R rounds, the clause embedding is the MEAN of all node vectors
(a simple, standard "readout"/pooling step).
"""

from __future__ import annotations
import numpy as np
from clause_graph import ClauseGraph, NODE_FEATURE_DIM

HIDDEN_DIM = 32
N_ROUNDS = 3
EDGE_FEATURE_DIM = 1  # just the normalized argument position


class RandomGNNEncoder:
    def __init__(self, hidden_dim: int = HIDDEN_DIM, n_rounds: int = N_ROUNDS, seed: int = 0):
        self.hidden_dim = hidden_dim
        self.n_rounds = n_rounds
        rng = np.random.default_rng(seed)

        def glorot(shape):
            limit = np.sqrt(6.0 / (shape[0] + shape[1]))
            return rng.uniform(-limit, limit, size=shape).astype(np.float32)

        # project raw node features into hidden_dim once, up front
        self.W_in = glorot((NODE_FEATURE_DIM, hidden_dim))

        # one shared set of weights, reused every round (standard GNN choice -
        # keeps parameter count fixed regardless of how many rounds we run)
        self.W_msg = glorot((hidden_dim + EDGE_FEATURE_DIM, hidden_dim))
        self.W_self = glorot((hidden_dim, hidden_dim))
        self.W_agg = glorot((hidden_dim, hidden_dim))
        self.b = np.zeros(hidden_dim, dtype=np.float32)

    def encode(self, graph: ClauseGraph) -> np.ndarray:
        n = len(graph.node_features)
        if n == 0:
            return np.zeros(self.hidden_dim, dtype=np.float32)

        X = np.array(graph.node_features, dtype=np.float32)  # (n, NODE_FEATURE_DIM)
        H = _relu(X @ self.W_in)  # (n, hidden_dim)

        # undirected adjacency as an explicit edge list, both directions
        edge_src, edge_dst, edge_feat = [], [], []
        for i, j, pos in graph.edges:
            edge_src += [i, j]
            edge_dst += [j, i]
            edge_feat += [pos, pos]
        edge_src = np.array(edge_src, dtype=np.int64)
        edge_dst = np.array(edge_dst, dtype=np.int64)
        edge_feat = np.array(edge_feat, dtype=np.float32).reshape(-1, 1)

        for _ in range(self.n_rounds):
            if len(edge_src) == 0:
                # no edges at all (shouldn't happen for a real clause, but
                # guard against a degenerate single-node graph)
                break
            neighbor_vecs = H[edge_src]                       # (n_edges, hidden_dim)
            msg_input = np.concatenate([neighbor_vecs, edge_feat], axis=1)
            messages = _relu(msg_input @ self.W_msg)          # (n_edges, hidden_dim)

            agg = np.zeros_like(H)
            counts = np.zeros(n, dtype=np.float32)
            np.add.at(agg, edge_dst, messages)
            np.add.at(counts, edge_dst, 1.0)
            counts[counts == 0] = 1.0  # isolated node guard
            agg = agg / counts[:, None]  # MEAN, not sum - degree-invariant

            H = _relu(H @ self.W_self + agg @ self.W_agg + self.b)

        pooled = H.mean(axis=0)
        norm = np.linalg.norm(pooled)
        return pooled / norm if norm > 0 else pooled


def _relu(x):
    return np.maximum(x, 0.0)


if __name__ == "__main__":
    from clause_graph import build_clause_graph

    encoder = RandomGNNEncoder()

    ast_a = {  # ~p(X, Y) | q(X)
        "_type": "BinOp", "op": "|",
        "left": {"_type": "Not", "formula": {"_type": "Atom", "term":
            {"_type": "Term", "name": "p", "args": [
                {"_type": "Var", "name": "X"}, {"_type": "Var", "name": "Y"}]}}},
        "right": {"_type": "Atom", "term":
            {"_type": "Term", "name": "q", "args": [{"_type": "Var", "name": "X"}]}},
    }
    ast_b = {  # ~same_day(A, B) | is_weekday(A)  - same STRUCTURE, different names
        "_type": "BinOp", "op": "|",
        "left": {"_type": "Not", "formula": {"_type": "Atom", "term":
            {"_type": "Term", "name": "same_day", "args": [
                {"_type": "Var", "name": "A"}, {"_type": "Var", "name": "B"}]}}},
        "right": {"_type": "Atom", "term":
            {"_type": "Term", "name": "is_weekday", "args": [{"_type": "Var", "name": "A"}]}},
    }
    ast_c = {  # ~p(X, Y) | q(Z)  - similar but NOT sharing a variable
        "_type": "BinOp", "op": "|",
        "left": {"_type": "Not", "formula": {"_type": "Atom", "term":
            {"_type": "Term", "name": "p", "args": [
                {"_type": "Var", "name": "X"}, {"_type": "Var", "name": "Y"}]}}},
        "right": {"_type": "Atom", "term":
            {"_type": "Term", "name": "q", "args": [{"_type": "Var", "name": "Z"}]}},
    }

    emb_a = encoder.encode(build_clause_graph(ast_a))
    emb_b = encoder.encode(build_clause_graph(ast_b))
    emb_c = encoder.encode(build_clause_graph(ast_c))

    def cos(u, v):
        return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9))

    print("cos(a, b) [same structure, different names] =", cos(emb_a, emb_b))
    print("cos(a, c) [similar but different variable sharing] =", cos(emb_a, emb_c))
