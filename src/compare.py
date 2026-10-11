"""Compare every run under outputs/ (reads test_metrics.json): prints a Markdown table
and writes a CSV. Smoke-test runs are skipped because their scores are not real results.

Usage:
    python -m src.compare [--outputs outputs] [--csv results/comparison.csv]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

COLUMNS = ["accuracy", "macro_f1", "weighted_f1", "macro_precision", "macro_recall"]


def collect(outputs: str | Path) -> list[dict]:
    rows = []
    for metrics_file in sorted(Path(outputs).glob("*/test_metrics.json")):
        run_dir = metrics_file.parent
        cfg_file = run_dir / "run_config.json"
        if cfg_file.exists() and json.loads(cfg_file.read_text(encoding="utf-8")).get("smoke_test"):
            print(f"Skipping {run_dir.name}: smoke test, not a real result", file=sys.stderr)
            continue
        metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
        rows.append({"model": run_dir.name, **{c: metrics.get(c) for c in COLUMNS}})
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--outputs", default="outputs")
    p.add_argument("--csv", default="results/comparison.csv", help="Where to write the CSV table ('' to skip)")
    args = p.parse_args()

    rows = collect(args.outputs)
    if not rows:
        raise SystemExit(f"No test_metrics.json found under {args.outputs}/*/")

    print("| Model | Accuracy | Macro-F1 | Weighted-F1 |")
    print("|---|---|---|---|")
    for r in rows:
        print(f"| {r['model']} | {r['accuracy']:.4f} | {r['macro_f1']:.4f} | {r['weighted_f1']:.4f} |")

    if args.csv:
        out = Path(args.csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["model", *COLUMNS])
            writer.writeheader()
            for r in rows:
                writer.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
        print(f"\nWrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
