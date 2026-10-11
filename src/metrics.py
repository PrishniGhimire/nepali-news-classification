"""Metrics, classification report, confusion matrix and prediction dumps."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def compute_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def _plot_confusion_matrix(cm: np.ndarray, labels: list[str], path: Path) -> None:
    # Matplotlib's default font has no Devanagari glyphs; fall back to class
    # indices for non-ASCII labels (the mapping is in labels.json).
    ticks = [label if label.isascii() else str(i) for i, label in enumerate(labels)]
    size = max(5.0, 0.8 * len(labels) + 2)
    fig, ax = plt.subplots(figsize=(size, size))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(ticks, rotation=45, ha="right")
    ax.set_yticklabels(ticks)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    threshold = cm.max() / 2 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, int(cm[i, j]), ha="center", va="center",
                color="white" if cm[i, j] > threshold else "black",
            )
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_report(y_true, y_pred, labels: list[str], out_dir: str | Path, prefix: str = "test") -> dict:
    """Write ``{prefix}_metrics.json``, ``{prefix}_classification_report.txt``
    and ``{prefix}_confusion_matrix.png``; return the metrics dict."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ids = list(range(len(labels)))

    metrics = compute_metrics(y_true, y_pred)
    (out / f"{prefix}_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    report = classification_report(
        y_true, y_pred, labels=ids, target_names=labels, digits=4, zero_division=0
    )
    (out / f"{prefix}_classification_report.txt").write_text(report, encoding="utf-8")

    cm = confusion_matrix(y_true, y_pred, labels=ids)
    _plot_confusion_matrix(cm, labels, out / f"{prefix}_confusion_matrix.png")
    return metrics


def save_predictions(df: pd.DataFrame, pred_ids, labels: list[str], path: str | Path) -> None:
    """Dump test-set predictions (text, true label, predicted label) for error analysis."""
    out = df[["text", "label"]].copy()
    out["pred"] = [labels[i] for i in pred_ids]
    out["correct"] = out["label"] == out["pred"]
    out.to_csv(path, index=False, encoding="utf-8")
