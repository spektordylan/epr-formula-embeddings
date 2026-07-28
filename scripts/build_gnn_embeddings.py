import json
import time
import numpy as np

from clause_graph import build_clause_graph
from gnn_encoder import RandomGNNEncoder

DATA_DIR = "data/processed"

corpus = json.load(open(f"{DATA_DIR}/corpus.json"))
metadata = json.load(open(f"{DATA_DIR}/clause_metadata.json"))

# metadata is already in the exact clause order used everywhere else in
# this project - reuse it so the row order matches clause_vectors.npy,
# meaning retrieve.py works unmodified against either file.
ast_by_id = {}
for problem in corpus:
    for formula in problem["formulas"]:
        cid = f"{problem['problem_id']}::{formula['name']}"
        ast_by_id[cid] = formula["ast"]

encoder = RandomGNNEncoder()

t0 = time.time()
vectors = np.zeros((len(metadata), encoder.hidden_dim), dtype=np.float32)
for i, m in enumerate(metadata):
    graph = build_clause_graph(ast_by_id[m["clause_id"]])
    vectors[i] = encoder.encode(graph)
    if i % 20000 == 0:
        print(f"  {i}/{len(metadata)}  ({time.time()-t0:.1f}s elapsed)")

np.save(f"{DATA_DIR}/gnn_vectors.npy", vectors)
print(f"\nSaved {DATA_DIR}/gnn_vectors.npy  shape={vectors.shape}  "
      f"({time.time()-t0:.1f}s total)")
