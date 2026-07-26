"""
Structural feature extraction for individual clauses/formulas.

Two kinds of signal, per clause:
  1. Numeric structural stats (literal count, variable reuse, depth, ...)
  2. A "bag of symbols" - a multiset of (predicate/constant name, arity)
     pairs that occur in the clause, vectorized against a fixed
     corpus-wide vocabulary of the most frequent symbols.

These combine into one fixed-length feature vector per clause, usable
directly with cosine similarity / nearest-neighbor search as a baseline,
before any learned embedding.
"""

from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Walking the serialized AST (dicts tagged with "_type", as written by
# build_corpus.py's ast_to_dict)
# ---------------------------------------------------------------------------

def _walk_term(node: dict, symbols: Counter, var_counts: Counter, depth: int) -> int:
    """Returns max depth beneath this term node. Updates symbols/var_counts
    in place as it walks."""
    if node["_type"] == "Var":
        var_counts[node["name"]] += 1
        return depth
    # Term: a predicate/function/constant application
    arity = len(node["args"])
    symbols[(node["name"], arity)] += 1
    if arity == 0:
        return depth
    return max(_walk_term(a, symbols, var_counts, depth + 1) for a in node["args"])


def _walk_formula(node: dict, symbols: Counter, var_counts: Counter,
                   depth: int, stats: dict) -> int:
    t = node["_type"]
    if t == "Atom":
        if node["term"]["name"] == "=":
            stats["has_equality"] = True
        return _walk_term(node["term"], symbols, var_counts, depth)
    if t == "Not":
        stats["n_negations"] += 1
        return _walk_formula(node["formula"], symbols, var_counts, depth + 1, stats)
    if t == "BinOp":
        if node["op"] == "|":
            stats["n_literals"] += 1
        dl = _walk_formula(node["left"], symbols, var_counts, depth + 1, stats)
        dr = _walk_formula(node["right"], symbols, var_counts, depth + 1, stats)
        return max(dl, dr)
    if t == "Quant":
        return _walk_formula(node["formula"], symbols, var_counts, depth + 1, stats)
    raise ValueError(f"Unknown AST node type: {t}")


def extract_clause_features(ast: dict) -> Tuple[dict, Counter]:
    """Returns (numeric_stats, symbol_counter) for one clause/formula AST."""
    stats = {
        "n_literals": 1,       # this clause itself counts as 1 literal
                               # unless it's a disjunction (BinOp '|' bumps it)
        "n_negations": 0,
        "has_equality": False,
    }
    symbols: Counter = Counter()
    var_counts: Counter = Counter()
    max_depth = _walk_formula(ast, symbols, var_counts, 0, stats)

    stats["max_depth"] = max_depth
    stats["n_distinct_vars"] = len(var_counts)
    stats["n_var_occurrences"] = sum(var_counts.values())
    stats["has_equality"] = float(stats["has_equality"])
    return stats, symbols


# ---------------------------------------------------------------------------
# Corpus-wide vocabulary + vectorization
# ---------------------------------------------------------------------------

NUMERIC_FIELDS = [
    "n_literals", "n_negations", "max_depth", "n_distinct_vars", "n_var_occurrences",
]
# has_equality is handled separately - it's a near-constant binary flag,
# not something to log-transform or z-score.


def build_symbol_vocab(all_symbol_counters: List[Counter], top_k: int = 200) -> List[Tuple[str, int]]:
    """Pick the top_k most frequent (name, arity) symbols across the whole
    corpus. Anything outside this vocab gets bucketed into an 'OTHER' slot
    at vectorization time."""
    total = Counter()
    for c in all_symbol_counters:
        total.update(c)
    return [sym for sym, _ in total.most_common(top_k)]


def fit_numeric_normalizer(all_stats: List[dict]) -> Dict[str, Tuple[float, float]]:
    """log1p each numeric field across the corpus, then compute mean/std of
    the logged values. Returns {field: (mean, std)} to apply at vectorize time.
    """
    import math
    params = {}
    for field in NUMERIC_FIELDS:
        logged = [math.log1p(s[field]) for s in all_stats]
        mean = sum(logged) / len(logged)
        var = sum((x - mean) ** 2 for x in logged) / len(logged)
        std = math.sqrt(var) or 1.0  # guard against a degenerate zero-variance field
        params[field] = (mean, std)
    return params


def vectorize(stats: dict, symbols: Counter, vocab: List[Tuple[str, int]],
              numeric_norm: Dict[str, Tuple[float, float]],
              block_weight: float = 1.0) -> List[float]:
    """block_weight scales the numeric block relative to the symbol block
    after both are independently L2-normalized (both start at norm 1, so
    block_weight directly controls their relative influence)."""
    import math

    # --- numeric block: log1p, then z-score using corpus-fit params ---
    numeric = []
    for field in NUMERIC_FIELDS:
        mean, std = numeric_norm[field]
        numeric.append((math.log1p(stats[field]) - mean) / std)
    numeric.append(stats["has_equality"])  # raw 0/1, untouched
    numeric = _l2_normalize(numeric)
    numeric = [x * block_weight for x in numeric]

    # --- symbol block: log1p counts, then L2-normalize ---
    vocab_index = {sym: i for i, sym in enumerate(vocab)}
    symbol_vec = [0.0] * (len(vocab) + 1)  # +1 for OTHER bucket
    for sym, count in symbols.items():
        idx = vocab_index.get(sym, len(vocab))
        symbol_vec[idx] += math.log1p(count)
    symbol_vec = _l2_normalize(symbol_vec)

    return numeric + symbol_vec


def _l2_normalize(vec: List[float]) -> List[float]:
    import math
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


if __name__ == "__main__":
    # Smoke test against the sample clause from the parser's own demo.
    sample_ast = {
        "_type": "BinOp", "op": "|",
        "left": {"_type": "Not", "formula": {"_type": "Atom", "term":
            {"_type": "Term", "name": "p", "args": [{"_type": "Term", "name": "a", "args": []}]}}},
        "right": {"_type": "Atom", "term":
            {"_type": "Term", "name": "q", "args": [{"_type": "Var", "name": "X"}]}},
    }
    stats, symbols = extract_clause_features(sample_ast)
    print("stats:", stats)
    print("symbols:", symbols)

    vocab = build_symbol_vocab([symbols])
    norm = fit_numeric_normalizer([stats])
    vec = vectorize(stats, symbols, vocab, norm)
    print("vector length:", len(vec))