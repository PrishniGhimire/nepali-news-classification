"""Print a Markdown table comparing every run under outputs/ (reads test_metrics.json).

Usage:
    python -m src.compare [--outputs outputs]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--outputs", default="outputs")
    args = p.parse_args()

    rows = []
    for metrics_file in sorted(Path(args.outputs).glob("*/test_metrics.json")):
        m = json.loads(metrics_file.read_text(encoding="utf-8"))
        rows.append((metrics_file.parent.name, m))
    if not rows:
        raise SystemExit(f"No test_metrics.json found under {args.outputs}/*/")

    print("| Model | Accuracy | Macro-F1 | Weighted-F1 |")
    print("|---|---|---|---|")
    for name, m in rows:
        print(f"| {name} | {m['accuracy']:.4f} | {m['macro_f1']:.4f} | {m['weighted_f1']:.4f} |")


if __name__ == "__main__":
    main()
