import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = next(
    (candidate for candidate in [SCRIPT_DIR, *SCRIPT_DIR.parents] if (candidate / ".git").exists() or (candidate / "data").exists()),
    SCRIPT_DIR,
)
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from tptp_parser import ParseError, is_epr_eligible, parse_fof_file

SPC_RE = re.compile(r"%\s*SPC\s*:\s*(\S+)")
STATUS_RE = re.compile(r"%\s*Status\s*:\s*(\S+)")
DOMAIN_RE = re.compile(r"%\s*Domain\s*:\s*(.+)")

SIZE_CUTOFF_BYTES = 500_000

RAW_DIR = REPO_ROOT / "data" / "raw" / "epr_subset"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

all_paths = sorted(
    [p for p in RAW_DIR.rglob("*.p") if p.is_file() and p.stat().st_size <= SIZE_CUTOFF_BYTES]
)
n_skipped_large = len(list(RAW_DIR.rglob("*.p"))) - len(all_paths)
print(f"Total files: {len(list(RAW_DIR.rglob('*.p')))} (skipping {n_skipped_large} over {SIZE_CUTOFF_BYTES/1000:.0f}KB)")
print(f"Files to process: {len(all_paths)}")

n_parsed = 0
n_parse_fail = 0
n_epr_confirmed = 0
n_epr_rejected = 0
fail_examples = []
status_counts = Counter()
domain_counts = Counter()

for path in all_paths:
    text = path.read_text(encoding="latin-1")

    m = STATUS_RE.search(text)
    if m:
        status_counts[m.group(1)] += 1
    m = DOMAIN_RE.search(text)
    if m:
        domain_counts[m.group(1).strip()] += 1

    try:
        units = parse_fof_file(text)
        n_parsed += 1
    except (ParseError, SyntaxError) as e:
        n_parse_fail += 1
        if len(fail_examples) < 5:
            fail_examples.append((str(path.relative_to(REPO_ROOT)), str(e)))
        continue

    ok, _ = is_epr_eligible(units)
    if ok:
        n_epr_confirmed += 1
    else:
        n_epr_rejected += 1

summary = {
    "processed_files": len(all_paths),
    "parsed_successfully": n_parsed,
    "parse_failures": n_parse_fail,
    "epr_confirmed": n_epr_confirmed,
    "epr_rejected": n_epr_rejected,
    "status_counts": dict(status_counts),
    "domain_counts": dict(domain_counts),
    "fail_examples": fail_examples,
}
summary_path = PROCESSED_DIR / "batch_parse_summary.json"
with summary_path.open("w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print(f"\nParsed successfully : {n_parsed} / {len(all_paths)}")
print(f"Failed to parse     : {n_parse_fail}")
print(f"EPR confirmed       : {n_epr_confirmed}")
print(f"EPR rejected (!)    : {n_epr_rejected}")

print("\nStatus label distribution (from header):")
for k, v in status_counts.most_common():
    print(f"  {k:15s} {v}")

print("\nTop problem domains:")
for k, v in domain_counts.most_common(10):
    print(f"  {k:20s} {v}")

if fail_examples:
    print("\nSample parse failures:")
    for path, err in fail_examples:
        print(f"  {path}: {err}")

print(f"\nSaved summary to {summary_path.relative_to(REPO_ROOT)}")