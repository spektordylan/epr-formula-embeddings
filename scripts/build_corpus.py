import json
import os
import re
import sys
from dataclasses import fields, is_dataclass
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

SIZE_CUTOFF_BYTES = 500_000

STATUS_RE = re.compile(r"%\s*Status\s*:\s*(\S+)")
DOMAIN_RE = re.compile(r"%\s*Domain\s*:\s*(.+)")
SPC_RE = re.compile(r"%\s*SPC\s*:\s*(\S+)")

RAW_DIR = REPO_ROOT / "data" / "raw" / "epr_subset"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def ast_to_dict(node):
    """Serialize a parser AST node to a JSON-friendly dict, tagging the
    node type explicitly so it can be reconstructed later."""
    if is_dataclass(node):
        d = {"_type": type(node).__name__}
        for f in fields(node):
            val = getattr(node, f.name)
            if isinstance(val, list):
                d[f.name] = [ast_to_dict(v) for v in val]
            elif is_dataclass(val):
                d[f.name] = ast_to_dict(val)
            else:
                d[f.name] = val
        return d
    return node


def main():
    all_paths = sorted(
        [p for p in RAW_DIR.rglob("*.p") if p.is_file() and p.stat().st_size <= SIZE_CUTOFF_BYTES]
    )

    corpus = []
    for path in all_paths:
        text = path.read_text(encoding="latin-1")
        try:
            units = parse_fof_file(text)
        except (ParseError, SyntaxError):
            continue
        ok, _ = is_epr_eligible(units)
        if not ok:
            continue

        status_m = STATUS_RE.search(text)
        domain_m = DOMAIN_RE.search(text)
        spc_m = SPC_RE.search(text)

        corpus.append({
            "problem_id": path.stem,
            "status": status_m.group(1) if status_m else None,
            "domain": domain_m.group(1).strip() if domain_m else None,
            "spc": spc_m.group(1) if spc_m else None,
            "formulas": [
                {"name": u.name, "role": u.role, "ast": ast_to_dict(u.formula)}
                for u in units
            ],
        })

    out_path = PROCESSED_DIR / "corpus.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(corpus, f, indent=2)

    n_clauses = sum(len(p["formulas"]) for p in corpus)
    print(f"Problems in corpus : {len(corpus)}")
    print(f"Total formulas/clauses across all problems: {n_clauses}")
    print(f"Saved to {out_path.relative_to(REPO_ROOT)} ({out_path.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()