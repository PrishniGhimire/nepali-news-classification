"""Dataset loading, cleaning and deterministic stratified splitting."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .text_utils import clean_text

log = logging.getLogger(__name__)


@dataclass
class Splits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    labels: list[str]

    @property
    def label2id(self) -> dict[str, int]:
        return {label: i for i, label in enumerate(self.labels)}

    @property
    def id2label(self) -> dict[int, str]:
        return dict(enumerate(self.labels))


def load_dataset(
    path: str | Path,
    text_col: str = "text",
    label_col: str = "label",
    min_chars: int = 10,
    min_per_class: int = 10,
) -> pd.DataFrame:
    """Load a CSV with one article per row and return a cleaned DataFrame
    with columns ``text`` and ``label``."""
    df = pd.read_csv(path)
    missing = {text_col, label_col} - set(df.columns)
    if missing:
        raise ValueError(
            f"Column(s) {sorted(missing)} not found in {path}. Available: {list(df.columns)}. "
            "Use --text_col / --label_col to point at the right ones."
        )

    df = df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label"}).dropna()
    df["text"] = df["text"].map(clean_text)
    df["label"] = df["label"].astype(str).str.strip()
    df = df[df["text"].str.len() >= min_chars]

    # Identical text under different labels is ambiguous: drop it entirely.
    n_labels = df.groupby("text")["label"].nunique()
    conflicting = n_labels[n_labels > 1].index
    if len(conflicting):
        log.warning("Dropping %d texts that appear with conflicting labels", len(conflicting))
        df = df[~df["text"].isin(conflicting)]

    # Exact duplicates would leak between train and test.
    before = len(df)
    df = df.drop_duplicates(subset="text")
    if before - len(df):
        log.info("Dropped %d duplicate texts", before - len(df))

    counts = df["label"].value_counts()
    rare = sorted(counts[counts < min_per_class].index)
    if rare:
        log.warning("Dropping classes with fewer than %d examples: %s", min_per_class, rare)
        df = df[~df["label"].isin(rare)]

    if df["label"].nunique() < 2:
        raise ValueError("Need at least two classes after cleaning; check your data and column names.")

    log.info("Loaded %d articles across %d classes", len(df), df["label"].nunique())
    return df.reset_index(drop=True)


def make_splits(
    df: pd.DataFrame,
    val_size: float = 0.1,
    test_size: float = 0.1,
    seed: int = 42,
) -> Splits:
    """Stratified train/val/test split. The same seed always gives the same
    split, so the baseline and the transformer are compared on identical data."""
    labels = sorted(df["label"].unique())
    label2id = {label: i for i, label in enumerate(labels)}
    df = df.assign(label_id=df["label"].map(label2id))

    train_val, test = train_test_split(
        df, test_size=test_size, stratify=df["label_id"], random_state=seed
    )
    train, val = train_test_split(
        train_val,
        test_size=val_size / (1.0 - test_size),
        stratify=train_val["label_id"],
        random_state=seed,
    )
    return Splits(
        train=train.reset_index(drop=True),
        val=val.reset_index(drop=True),
        test=test.reset_index(drop=True),
        labels=labels,
    )


def save_label_map(labels: list[str], out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "labels.json").write_text(
        json.dumps({"labels": labels}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
