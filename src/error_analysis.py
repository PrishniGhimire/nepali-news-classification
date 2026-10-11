"""Error analysis from a test_predictions.csv written by the training scripts.

Lists the most common confusions with real misclassified articles, accuracy by article
length, and (optionally) how two models' mistakes compare.

Usage:
    python -m src.error_analysis --predictions outputs/xlm-roberta-base/test_predictions.csv
    python -m src.error_analysis --predictions outputs/xlm-roberta-base/test_predictions.csv \
        --other outputs/baseline/test_predictions.csv --out results/error_analysis.md
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import binomtest


def length_table(df: pd.DataFrame) -> pd.DataFrame:
    """Accuracy for short / medium / long articles (terciles of character length)."""
    lengths = df["text"].str.len()
    bucket = pd.qcut(lengths.rank(method="first"), 3, labels=["short", "medium", "long"])
    rows = []
    for name in ["short", "medium", "long"]:
        part = df[bucket == name]
        rows.append({
            "length": name,
            "articles": len(part),
            "chars (min-max)": f"{int(part['text'].str.len().min())}-{int(part['text'].str.len().max())}",
            "accuracy": round(float(part["correct"].mean()), 3),
        })
    return pd.DataFrame(rows)


def top_confusions(df: pd.DataFrame, k: int) -> pd.DataFrame:
    wrong = df[~df["correct"]]
    counts = wrong.groupby(["label", "pred"]).size().sort_values(ascending=False).head(k)
    return counts.reset_index(name="count")


def snippet(text: str, n: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[:n].rstrip() + " ..."


def compare_models(df: pd.DataFrame, other: pd.DataFrame) -> dict:
    if not (df["text"].reset_index(drop=True) == other["text"].reset_index(drop=True)).all():
        raise ValueError("The two prediction files are not on the same test articles in the same order")
    a, b = df["correct"].reset_index(drop=True), other["correct"].reset_index(drop=True)
    only_main, only_other = int((a & ~b).sum()), int((~a & b).sum())
    p_value = binomtest(only_main, only_main + only_other, 0.5).pvalue if (only_main + only_other) else 1.0
    same_wrong = int(((~a) & (~b) & (df["pred"].reset_index(drop=True) == other["pred"].reset_index(drop=True))).sum())
    return {
        "both_correct": int((a & b).sum()),
        "both_wrong": int((~a & ~b).sum()),
        "only_main_correct": only_main,
        "only_other_correct": only_other,
        "both_wrong_same_label": same_wrong,
        "mcnemar_p": round(float(p_value), 3),
    }


def build_report(
    df: pd.DataFrame,
    title: str,
    top_pairs: int = 6,
    examples: int = 3,
    snippet_chars: int = 160,
    seed: int = 42,
    other: pd.DataFrame | None = None,
    other_title: str = "other",
) -> str:
    n = len(df)
    n_wrong = int((~df["correct"]).sum())
    lines = [f"# Error analysis: {title}", "",
             f"{n} test articles, {n_wrong} misclassified (accuracy {1 - n_wrong / n:.4f}).", ""]

    lines += ["## Accuracy by article length", "", "Articles are split into three equal-sized groups by character length.", "",
              length_table(df).to_markdown(index=False), ""]

    pairs = top_confusions(df, top_pairs)
    lines += ["## Most common confusions", "", pairs.rename(columns={"label": "true label", "pred": "predicted"})
              .to_markdown(index=False), ""]

    lines += ["## Examples of misclassified articles", "",
              f"Random examples (seed {seed}); text is cut to {snippet_chars} characters.", ""]
    wrong = df[~df["correct"]]
    for _, row in pairs.iterrows():
        pool = wrong[(wrong["label"] == row["label"]) & (wrong["pred"] == row["pred"])]
        lines.append(f"### {row['label']} predicted as {row['pred']} ({row['count']} articles)")
        lines.append("")
        for text in pool.sample(min(examples, len(pool)), random_state=seed)["text"]:
            lines.append(f"- {snippet(text, snippet_chars)}")
        lines.append("")

    if other is not None:
        cmp = compare_models(df, other)
        lines += [f"## Comparison with {other_title}", "",
                  f"- Correct in both: {cmp['both_correct']}",
                  f"- Wrong in both: {cmp['both_wrong']} (same wrong label in {cmp['both_wrong_same_label']})",
                  f"- Only {title} correct: {cmp['only_main_correct']}",
                  f"- Only {other_title} correct: {cmp['only_other_correct']}",
                  f"- Paired exact test (McNemar) p = {cmp['mcnemar_p']}", ""]
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--predictions", required=True)
    p.add_argument("--other", help="Predictions of a second model on the same test set")
    p.add_argument("--title", help="Name of the main model (default: parent folder name)")
    p.add_argument("--other_title", help="Name of the second model (default: parent folder name)")
    p.add_argument("--top_pairs", type=int, default=6)
    p.add_argument("--examples", type=int, default=3)
    p.add_argument("--snippet_chars", type=int, default=160)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="results/error_analysis.md")
    args = p.parse_args()

    df = pd.read_csv(args.predictions)
    other = pd.read_csv(args.other) if args.other else None
    title = args.title or Path(args.predictions).resolve().parent.name
    other_title = args.other_title or (Path(args.other).resolve().parent.name if args.other else "other")
    report = build_report(df, title, args.top_pairs, args.examples, args.snippet_chars, args.seed, other, other_title)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
