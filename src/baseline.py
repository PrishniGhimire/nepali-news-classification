"""TF-IDF (word + character n-grams) + logistic regression baseline.

Usage:
    python -m src.baseline --data data/sample_news.csv --output_dir outputs/baseline
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import FeatureUnion, Pipeline

from .data import check_no_overlap, load_dataset, make_splits, save_label_map, save_split_definition
from .metrics import save_predictions, save_report
from .text_utils import tokenize

log = logging.getLogger("baseline")


def build_pipeline(C: float = 10.0, min_df: int = 2) -> Pipeline:
    features = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    tokenizer=tokenize,
                    token_pattern=None,
                    lowercase=False,
                    ngram_range=(1, 2),
                    min_df=min_df,
                    sublinear_tf=True,
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(2, 5),
                    min_df=min_df,
                    sublinear_tf=True,
                    max_features=300_000,
                ),
            ),
        ]
    )
    clf = LogisticRegression(C=C, max_iter=2000, class_weight="balanced")
    return Pipeline([("features", features), ("clf", clf)])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="CSV with a text column and a label column")
    p.add_argument("--text_col", default="text")
    p.add_argument("--label_col", default="label")
    p.add_argument("--output_dir", default="outputs/baseline")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val_size", type=float, default=0.1)
    p.add_argument("--test_size", type=float, default=0.1)
    p.add_argument("--min_df", type=int, default=2)
    p.add_argument("--c_grid", type=float, nargs="+", default=[1.0, 3.0, 10.0, 30.0],
                   help="Values of C to try; the best macro-F1 on the validation set wins")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    df = load_dataset(args.data, args.text_col, args.label_col)
    splits = make_splits(df, args.val_size, args.test_size, args.seed)
    check_no_overlap(splits)
    log.info("train=%d val=%d test=%d", len(splits.train), len(splits.val), len(splits.test))

    best = None
    for C in args.c_grid:
        pipe = build_pipeline(C, args.min_df).fit(splits.train["text"], splits.train["label_id"])
        val_f1 = f1_score(splits.val["label_id"], pipe.predict(splits.val["text"]), average="macro")
        log.info("C=%-6g val macro-F1=%.4f", C, val_f1)
        if best is None or val_f1 > best[0]:
            best = (val_f1, C, pipe)

    val_f1, best_c, pipe = best
    log.info("Best C=%g (val macro-F1=%.4f)", best_c, val_f1)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pred = pipe.predict(splits.test["text"])
    metrics = save_report(splits.test["label_id"], pred, splits.labels, out, prefix="test")
    save_predictions(splits.test, pred, splits.labels, out / "test_predictions.csv")
    save_label_map(splits.labels, out)
    save_split_definition(splits, out / "splits.csv")
    joblib.dump({"pipeline": pipe, "labels": splits.labels}, out / "model.joblib")
    (out / "run_config.json").write_text(
        json.dumps(
            {**vars(args), "best_C": best_c, "val_macro_f1": val_f1,
             "n_train": len(splits.train), "n_val": len(splits.val), "n_test": len(splits.test)},
            indent=2,
        ),
        encoding="utf-8",
    )
    log.info("TEST  accuracy=%.4f  macro-F1=%.4f", metrics["accuracy"], metrics["macro_f1"])
    log.info("Artifacts written to %s", out)


if __name__ == "__main__":
    main()
