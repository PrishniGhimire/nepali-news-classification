import json

import pandas as pd
import pytest

from src.compare import collect
from src.error_analysis import build_report, compare_models, top_confusions
from src.inspect_data import summarize


def _preds(correct_flags, preds=None):
    n = len(correct_flags)
    labels = ["a" if i % 2 == 0 else "b" for i in range(n)]
    pred = [l if ok else ("b" if l == "a" else "a") for l, ok in zip(labels, correct_flags)]
    return pd.DataFrame({"text": [f"article {i} " * 5 for i in range(n)], "label": labels,
                         "pred": preds or pred, "correct": correct_flags})


def test_top_confusions_counts_wrong_pairs():
    df = _preds([True, False, False, True, False, False])
    pairs = top_confusions(df, 3)
    assert pairs["count"].sum() == 4
    assert set(pairs.columns) == {"label", "pred", "count"}


def test_report_contains_examples_and_length_table():
    df = _preds([True, False] * 9)
    report = build_report(df, "model-x", top_pairs=2, examples=2)
    assert "# Error analysis: model-x" in report
    assert "Accuracy by article length" in report
    assert "predicted as" in report
    assert "article" in report


def test_model_comparison_counts_and_misaligned_inputs():
    main = _preds([True, True, False, False, True, False])
    other = _preds([True, False, True, False, True, True])
    cmp = compare_models(main, other)
    assert cmp["both_correct"] == 2 and cmp["both_wrong"] == 1
    assert cmp["only_main_correct"] == 1 and cmp["only_other_correct"] == 2
    with pytest.raises(ValueError, match="same test articles"):
        compare_models(main, other.assign(text=other["text"].str.upper()))


def test_inspect_data_summary(frame):
    raw = pd.concat([frame, frame.head(3), pd.DataFrame({"text": [None], "label": ["sports"]})], ignore_index=True)
    s = summarize(raw)
    assert s["rows"] == len(frame) + 4
    assert s["missing_text"] == 1
    assert s["exact_duplicate_rows_after_cleaning"] == 3
    assert s["n_classes"] == 3
    assert s["length_characters"]["median"] > 0


def test_compare_skips_smoke_runs(tmp_path):
    for name, smoke in (("real", False), ("quick-smoke", True)):
        d = tmp_path / name
        d.mkdir()
        (d / "test_metrics.json").write_text(json.dumps({"accuracy": 0.5, "macro_f1": 0.4, "weighted_f1": 0.4}))
        (d / "run_config.json").write_text(json.dumps({"smoke_test": smoke}))
    assert [r["model"] for r in collect(tmp_path)] == ["real"]
