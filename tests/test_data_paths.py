from pathlib import Path
import json
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_build_corpus_writes_processed_output():
    out_path = REPO_ROOT / "data" / "processed" / "corpus.json"
    if out_path.exists():
        out_path.unlink()

    subprocess.run(
        [sys.executable, str(REPO_ROOT / "src" / "build_corpus.py")],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert out_path.exists(), f"Expected corpus output at {out_path}"
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert isinstance(payload, list)
    assert payload, "Expected at least one parsed EPR problem"


def test_batch_parse_writes_processed_summary():
    summary_path = REPO_ROOT / "data" / "processed" / "batch_parse_summary.json"
    if summary_path.exists():
        summary_path.unlink()

    subprocess.run(
        [sys.executable, str(REPO_ROOT / "src" / "batch_parse.py")],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert summary_path.exists(), f"Expected batch summary output at {summary_path}"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["processed_files"] >= 1
