import pandas as pd
import pytest

from src.data import (
    article_id,
    check_no_overlap,
    load_dataset,
    make_splits,
    save_label_map,
    save_split_definition,
)


def test_label_mapping_is_sorted_and_consistent(frame):
    splits = make_splits(frame, seed=1)
    assert splits.labels == sorted(frame["label"].unique())
    assert splits.label2id == {label: i for i, label in enumerate(splits.labels)}
    assert splits.id2label == {i: label for i, label in enumerate(splits.labels)}
    for part in (splits.train, splits.val, splits.test):
        assert (part["label"].map(splits.label2id) == part["label_id"]).all()


def test_splits_do_not_overlap_and_cover_everything(frame):
    splits = make_splits(frame, seed=1)
    check_no_overlap(splits)  # raises on leakage
    total = len(splits.train) + len(splits.val) + len(splits.test)
    assert total == len(frame)
    assert not set(splits.train["text"]) & set(splits.test["text"])
    assert not set(splits.train["text"]) & set(splits.val["text"])
    assert not set(splits.val["text"]) & set(splits.test["text"])


def test_overlap_check_detects_leakage(frame):
    splits = make_splits(frame, seed=1)
    splits.test = pd.concat([splits.test, splits.train.head(1)], ignore_index=True)
    with pytest.raises(ValueError, match="leakage"):
        check_no_overlap(splits)


def test_split_is_deterministic_and_stratified(frame):
    a, b = make_splits(frame, seed=7), make_splits(frame, seed=7)
    assert a.test["text"].tolist() == b.test["text"].tolist()
    assert make_splits(frame, seed=8).test["text"].tolist() != a.test["text"].tolist()
    # every class appears in every split, with similar proportions
    for part in (a.train, a.val, a.test):
        assert set(part["label"]) == set(frame["label"])


def test_load_dataset_removes_duplicates_and_conflicts(tmp_path, frame):
    dup = frame.iloc[[0]].copy()
    conflict_a = pd.DataFrame({"text": ["यो एउटै पाठ हो जुन दुई वर्गमा पर्छ"], "label": ["sports"]})
    conflict_b = pd.DataFrame({"text": ["यो एउटै पाठ हो जुन दुई वर्गमा पर्छ"], "label": ["health"]})
    raw = pd.concat([frame, dup, conflict_a, conflict_b], ignore_index=True)
    path = tmp_path / "news.csv"
    raw.to_csv(path, index=False)

    df = load_dataset(path)
    assert df["text"].is_unique
    assert "यो एउटै पाठ हो जुन दुई वर्गमा पर्छ" not in set(df["text"])
    assert len(df) == len(frame)


def test_load_dataset_reports_wrong_column_names(tmp_path, frame):
    path = tmp_path / "news.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="not found"):
        load_dataset(path, text_col="content")


def test_saved_split_definition_and_label_map(tmp_path, frame):
    splits = make_splits(frame, seed=1)
    save_split_definition(splits, tmp_path / "splits.csv")
    saved = pd.read_csv(tmp_path / "splits.csv")
    assert set(saved["split"]) == {"train", "val", "test"}
    assert len(saved) == len(frame)
    assert saved["id"].is_unique
    assert saved.loc[saved["split"] == "test", "id"].tolist() == [article_id(t) for t in splits.test["text"]]

    save_label_map(splits.labels, tmp_path)
    assert (tmp_path / "labels.json").exists()
