import random
import sys
import numpy as np
from collections import Counter
from retrieve import load_index, _topk_from_vector

vectors_file = sys.argv[1] if len(sys.argv) > 1 else "clause_vectors.npy"
print(f"Evaluating: {vectors_file}\n")

random.seed(0)
index = load_index(vectors_file)
metadata = index["metadata"]
n = len(metadata)
k = 10
n_queries = 500

domain_counts = Counter(m["domain"] for m in metadata)
baseline_same_domain_prob = sum((c / n) ** 2 for c in domain_counts.values())

sample_rows = random.sample(range(n), n_queries)
same_domain_hits = 0
same_problem_hits = 0
total_neighbors = 0

for row in sample_rows:
    query_vec = index["vectors"][row]
    top_idx, _ = _topk_from_vector(index, query_vec, k, exclude_idx=row)
    q_domain = metadata[row]["domain"]
    q_problem = metadata[row]["problem_id"]
    for i in top_idx:
        total_neighbors += 1
        if metadata[i]["domain"] == q_domain:
            same_domain_hits += 1
        if metadata[i]["problem_id"] == q_problem:
            same_problem_hits += 1

print(f"Queries: {n_queries}, k={k}, total neighbor slots: {total_neighbors}")
print(f"Same-domain rate ........ {same_domain_hits/total_neighbors:.3f}")
print(f"  (chance baseline, given domain sizes: {baseline_same_domain_prob:.3f})")
print(f"Same-problem rate ....... {same_problem_hits/total_neighbors:.3f}")
print(f"  (chance baseline, ~1/{n} problems: {1/len(set(m['problem_id'] for m in metadata)):.4f})")
