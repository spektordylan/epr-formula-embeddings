import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = next(
    (candidate for candidate in [SCRIPT_DIR, *SCRIPT_DIR.parents] if (candidate / ".git").exists() or (candidate / "data").exists()),
    SCRIPT_DIR,
)
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from clause_features import (
    NUMERIC_FIELDS,
    build_symbol_vocab,
    extract_clause_features,
    fit_numeric_normalizer,
    vectorize,
)

corpus_path = REPO_ROOT / "data" / "processed" / "corpus.json"
if not corpus_path.exists():
    corpus_path = REPO_ROOT / "data" / "corpus.json"

OUT_DIR = REPO_ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

with corpus_path.open(encoding="utf-8") as f:
    corpus = json.load(f)

# Flatten to one record per clause, carrying parent problem metadata for
# later evaluation (domain/status act as weak labels, not training targets).
# IMPORTANT: this list's order defines row order in the saved vectors array -
# clause_metadata.json and clause_vectors.npy must always be read together,
# same index = same clause.
clause_records = []
for problem in corpus:
    for formula in problem["formulas"]:
        clause_records.append({
            "clause_id": f"{problem['problem_id']}::{formula['name']}",
            "problem_id": problem["problem_id"],
            "domain": problem["domain"],
            "status": problem["status"],
            "role": formula["role"],
            "ast": formula["ast"],
        })

print(f"Total clauses: {len(clause_records)}")

# Extract stats + symbol counters for every clause
all_stats = []
all_symbols = []
for rec in clause_records:
    stats, symbols = extract_clause_features(rec["ast"])
    all_stats.append(stats)
    all_symbols.append(symbols)

vocab = build_symbol_vocab(all_symbols, top_k=200)
numeric_norm = fit_numeric_normalizer(all_stats)
print(f"Vocab size (top-200 symbols + OTHER): {len(vocab) + 1}")

vectors = [vectorize(s, sym, vocab, numeric_norm) for s, sym in zip(all_stats, all_symbols)]
vectors = np.array(vectors, dtype=np.float32)
print(f"Feature matrix shape: {vectors.shape}")

# 1. The feature vectors themselves
np.save(OUT_DIR / "clause_vectors.npy", vectors)

# 2. Metadata table, one entry per row of clause_vectors.npy, same order.
#    (ast dropped here - it's already in corpus.json if ever needed again;
#    this file is just for identifying/filtering/labeling rows.)
metadata = [
    {k: v for k, v in rec.items() if k != "ast"}
    for rec in clause_records
]
with (OUT_DIR / "clause_metadata.json").open("w", encoding="utf-8") as f:
    json.dump(metadata, f)

# 3. Vocab + normalization params, needed to vectorize any NEW clause
#    (e.g. a hand-typed query formula) consistently with this corpus.
vocab_out = {
    "vocab": [list(sym) for sym in vocab],  # [(name, arity), ...] -> [[name, arity], ...]
    "numeric_fields": NUMERIC_FIELDS,
    "numeric_norm": {k: list(v) for k, v in numeric_norm.items()},
}
with (OUT_DIR / "feature_vocab.json").open("w", encoding="utf-8") as f:
    json.dump(vocab_out, f)

print(f"\nSaved to {OUT_DIR}/:")
print("  clause_vectors.npy   -", vectors.shape, vectors.dtype)
print("  clause_metadata.json -", len(metadata), "rows")
print("  feature_vocab.json   - vocab + normalization params")