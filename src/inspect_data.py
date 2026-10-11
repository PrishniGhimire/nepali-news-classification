"""Inspect a news CSV before training: missing values, duplicates, category counts, text lengths.

Usage:
    python -m src.inspect_data --data data/news.csv
    python -m src.inspect_data --data data/news.csv --tokenizer xlm-roberta-base --max_length 256
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .text_utils import clean_text


def _length_stats(values) -> dict:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return {}
    return {
        "min": int(arr.min()),
        "median": float(np.median(arr)),
        "mean": round(float(arr.mean()), 1),
        "p90": float(np.percentile(arr, 90)),
        "p99": float(np.percentile(arr, 99)),
        "max": int(arr.max()),
    }


def summarize(
    df: pd.DataFrame,
    text_col: str = "text",
    label_col: str = "label",
    tokenizer_name: str | None = None,
    max_length: int = 256,
) -> dict:
    """Return a JSON-serialisable summary of a raw dataframe (before any cleaning is applied to it)."""
    out: dict = {"rows": int(len(df)), "columns": list(map(str, df.columns))}
    out["missing_text"] = int(df[text_col].isna().sum())
    out["missing_label"] = int(df[label_col].isna().sum())

    d = df[[text_col, label_col]].dropna()
    texts = d[text_col].map(clean_text)
    labels = d[label_col].astype(str).str.strip()
    out["empty_after_cleaning"] = int((texts.str.len() == 0).sum())

    frame = pd.DataFrame({"text": texts, "label": labels})
    frame = frame[frame["text"].str.len() > 0]
    out["exact_duplicate_rows_after_cleaning"] = int(frame.duplicated(subset="text").sum())
    per_text = frame.groupby("text")["label"].nunique()
    out["texts_with_conflicting_labels"] = int((per_text > 1).sum())

    counts = frame["label"].value_counts()
    out["n_classes"] = int(len(counts))
    out["class_counts"] = {str(k): int(v) for k, v in counts.sort_index().items()}
    out["imbalance_ratio_max_over_min"] = round(float(counts.max() / counts.min()), 2) if len(counts) else None

    out["length_characters"] = _length_stats(frame["text"].str.len())
    out["length_words"] = _length_stats(frame["text"].str.split().map(len))

    if tokenizer_name:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(tokenizer_name)
        lengths = [len(ids) for ids in tok(list(frame["text"]), truncation=False)["input_ids"]]
        out["length_tokens"] = _length_stats(lengths)
        out["tokenizer"] = tokenizer_name
        out[f"share_truncated_at_{max_length}_tokens"] = round(float(np.mean(np.array(lengths) > max_length)), 4)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True)
    p.add_argument("--text_col", default="text")
    p.add_argument("--label_col", default="label")
    p.add_argument("--tokenizer", default=None, help="Optional HF tokenizer id to measure real token lengths")
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--out", default="results/data_summary.json")
    args = p.parse_args()

    summary = summarize(pd.read_csv(args.data), args.text_col, args.label_col, args.tokenizer, args.max_length)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
