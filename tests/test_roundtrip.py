import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT / "src", REPO_ROOT / "scripts"):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from tptp_parser import parse_fof_file
from build_corpus import ast_to_dict
from formula_printer import format_formula

corpus_path = REPO_ROOT / "data" / "processed" / "corpus.json"
with corpus_path.open(encoding="utf-8") as f:
    corpus = json.load(f)

all_asts = [f["ast"] for p in corpus for f in p["formulas"]]
random.seed(0)
sample = random.sample(all_asts, min(2000, len(all_asts)))

n_ok = 0
n_mismatch = 0
mismatch_examples = []

for ast in sample:
    text = format_formula(ast)
    wrapped = f"fof(rt_test, axiom, {text})."
    try:
        units = parse_fof_file(wrapped)
        reparsed = ast_to_dict(units[0].formula)
    except Exception as e:
        n_mismatch += 1
        if len(mismatch_examples) < 5:
            mismatch_examples.append((text, f"reparse error: {e}"))
        continue

    if reparsed == ast:
        n_ok += 1
    else:
        n_mismatch += 1
        if len(mismatch_examples) < 5:
            mismatch_examples.append((text, "AST mismatch after round-trip"))

print(f"Round-trip tested: {len(sample)}")
print(f"Match            : {n_ok}")
print(f"Mismatch         : {n_mismatch}")

if mismatch_examples:
    print("\nExamples of mismatches:")
    for text, reason in mismatch_examples:
        print(f"  {reason}: {text}")