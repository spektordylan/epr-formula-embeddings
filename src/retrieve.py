"""
Nearest-neighbor retrieval over the clause corpus using the structural
feature baseline. Two ways to query:
  1. By clause_id already in the corpus (retrieve_by_id)
  2. By a hand-typed TPTP-style formula string, not necessarily in the
     corpus (retrieve_by_text) - parsed and vectorized fresh using the
     SAME saved vocab/normalization params fit on the corpus, so the
     resulting vector is directly comparable to the stored ones.
"""

import json
import numpy as np

from tptp_parser import parse_fof_file
from build_corpus import ast_to_dict
from clause_features import extract_clause_features, vectorize
from formula_printer import format_formula

DATA_DIR = "data/processed"


def load_index(vectors_file="clause_vectors.npy"):
    vectors = np.load(f"{DATA_DIR}/{vectors_file}")
    metadata = json.load(open(f"{DATA_DIR}/clause_metadata.json"))
    vocab_data = json.load(open(f"{DATA_DIR}/feature_vocab.json"))
    corpus = json.load(open(f"{DATA_DIR}/corpus.json"))

    # rebuild a clause_id -> ast lookup so we can pretty-print any result
    ast_by_id = {}
    for problem in corpus:
        for formula in problem["formulas"]:
            cid = f"{problem['problem_id']}::{formula['name']}"
            ast_by_id[cid] = formula["ast"]

    vocab = [tuple(sym) for sym in vocab_data["vocab"]]
    numeric_norm = {k: tuple(v) for k, v in vocab_data["numeric_norm"].items()}

    # pre-normalize the whole matrix once so every query is a single matmul
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized_matrix = vectors / norms

    return {
        "vectors": vectors,
        "normalized_matrix": normalized_matrix,
        "metadata": metadata,
        "vocab": vocab,
        "numeric_norm": numeric_norm,
        "ast_by_id": ast_by_id,
    }


def _topk_from_vector(index, query_vec, k, exclude_idx=None):
    q_norm = np.linalg.norm(query_vec) or 1.0
    q = query_vec / q_norm
    sims = index["normalized_matrix"] @ q
    if exclude_idx is not None:
        sims = sims.copy()
        sims[exclude_idx] = -np.inf
    top_idx = np.argpartition(-sims, k)[:k]
    top_idx = top_idx[np.argsort(-sims[top_idx])]
    return top_idx, sims[top_idx]


def retrieve_by_id(index, clause_id, k=5):
    ids = [m["clause_id"] for m in index["metadata"]]
    try:
        row = ids.index(clause_id)
    except ValueError:
        raise KeyError(f"clause_id not found: {clause_id}")
    query_vec = index["vectors"][row]
    top_idx, sims = _topk_from_vector(index, query_vec, k, exclude_idx=row)
    return _format_results(index, top_idx, sims)


def retrieve_by_text(index, formula_text, role="axiom", k=5):
    """formula_text should be a bare formula, e.g. '~p(a) | q(X)' - it's
    wrapped as a throwaway fof(...) unit and parsed the same way the
    corpus itself was built, so features are extracted identically."""
    wrapped = f"fof(query, {role}, {formula_text})."
    units = parse_fof_file(wrapped)
    ast = ast_to_dict(units[0].formula)
    stats, symbols = extract_clause_features(ast)
    query_vec = np.array(
        vectorize(stats, symbols, index["vocab"], index["numeric_norm"]),
        dtype=np.float32,
    )
    top_idx, sims = _topk_from_vector(index, query_vec, k, exclude_idx=None)
    return _format_results(index, top_idx, sims)


def _format_results(index, top_idx, sims):
    results = []
    for i, sim in zip(top_idx, sims):
        meta = index["metadata"][i]
        ast = index["ast_by_id"][meta["clause_id"]]
        results.append({
            "clause_id": meta["clause_id"],
            "domain": meta["domain"],
            "status": meta["status"],
            "role": meta["role"],
            "similarity": float(sim),
            "formula": format_formula(ast),
        })
    return results


def print_results(results):
    for r in results:
        print(f"  [{r['similarity']:.3f}] {r['formula']}")
        print(f"          ({r['clause_id']}  domain={r['domain']}  status={r['status']}  role={r['role']})")


if __name__ == "__main__":
    index = load_index()

    # Demo 1: query by an existing clause_id
    example_id = index["metadata"][0]["clause_id"]
    print(f"Nearest neighbors of {example_id}:")
    print(f"  query: {format_formula(index['ast_by_id'][example_id])}\n")
    print_results(retrieve_by_id(index, example_id, k=5))

    # Demo 2: query by hand-typed formula, not necessarily in the corpus
    print("\nNearest neighbors of hand-typed query '~p(a) | q(X)':")
    print_results(retrieve_by_text(index, "~p(a) | q(X)", k=5))
