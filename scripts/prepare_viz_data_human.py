import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from formula_printer import format_formula

DATA_DIR = REPO_ROOT / "data" / "processed"

# Domains confirmed (by inspecting actual formulas) to use meaningful,
# human-chosen symbol names rather than auto-generated/bit-blasted/abstract
# placeholders. Excluded: Software Verification, Hardware Verification,
# Planning (all Skolem/state-bit style names), Syntactic + Syntactic
# (Translated) (deliberately abstract p/q placeholders), Miscellaneous
# (also abstract p/q in the sample checked).
HUMAN_DOMAINS = {
    "Geometry", "Group Theory", "Group Theory (Quasigroups)",
    "Knowledge Representation", "Lattice Theory", "Analysis",
    "Management (Organisation Theory)", "Natural Language Processing",
    "Puzzles", "Puzzle", "Set Theory",
}

vectors = np.load(DATA_DIR / "clause_vectors.npy")
with (DATA_DIR / "clause_metadata.json").open(encoding="utf-8") as f:
    metadata = json.load(f)
with (DATA_DIR / "corpus.json").open(encoding="utf-8") as f:
    corpus = json.load(f)

ast_by_id = {}
for problem in corpus:
    for formula in problem["formulas"]:
        cid = f"{problem['problem_id']}::{formula['name']}"
        ast_by_id[cid] = formula["ast"]

sample_idx = [i for i, m in enumerate(metadata) if m["domain"] in HUMAN_DOMAINS]
print(f"Filtered to human-named domains: {len(sample_idx)} clauses "
      f"(from {len(metadata)} total)")
print(dict(Counter(metadata[i]["domain"] for i in sample_idx)))

# Fit PCA on just this filtered subset - we specifically want the
# variance structure WITHIN these domains, not diluted by the huge
# synthetic domains we're excluding.
sub_vectors = vectors[sample_idx]
pca = PCA(n_components=2, random_state=0)
coords = pca.fit_transform(sub_vectors)
print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
coords = coords / (np.abs(coords).max() or 1.0)

norms = np.linalg.norm(sub_vectors, axis=1, keepdims=True)
norms[norms == 0] = 1.0
normalized = sub_vectors / norms
sim_matrix = normalized @ normalized.T

K = 6
neighbor_lists = []
for row in range(len(sample_idx)):
    sims = sim_matrix[row].copy()
    sims[row] = -np.inf
    top = np.argpartition(-sims, K)[:K]
    top = top[np.argsort(-sims[top])]
    neighbor_lists.append([(int(t), float(sims[t])) for t in top])

nodes = []
for local_i, global_i in enumerate(sample_idx):
    m = metadata[global_i]
    ast = ast_by_id[m["clause_id"]]
    nodes.append({
        "id": local_i,
        "x": float(coords[local_i, 0]),
        "y": float(coords[local_i, 1]),
        "domain": m["domain"],
        "status": m["status"],
        "problem_id": m["problem_id"],
        "role": m["role"],
        "clause_id": m["clause_id"],
        "formula": format_formula(ast),
        "neighbors": [{"id": nid, "sim": round(sim, 3)} for nid, sim in neighbor_lists[local_i]],
    })

out = {"nodes": nodes, "domains": sorted(HUMAN_DOMAINS & set(metadata[i]["domain"] for i in sample_idx))}

out_path = DATA_DIR / "viz_data_human.json"
with out_path.open("w", encoding="utf-8") as f:
    json.dump(out, f)

print(f"Saved {out_path} ({len(nodes)} nodes)")