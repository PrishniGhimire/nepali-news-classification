import joblib
import numpy as np
import pytest

from src.baseline import build_pipeline
from src.data import make_splits
from src.predict import load_predictor
from src.text_utils import clean_text


@pytest.fixture
def saved_baseline(tmp_path, frame):
    """Train a tiny baseline on synthetic data and save it in the same format as src.baseline."""
    splits = make_splits(frame, seed=0)
    pipe = build_pipeline(C=10.0, min_df=1).fit(splits.train["text"], splits.train["label_id"])
    path = tmp_path / "model.joblib"
    joblib.dump({"pipeline": pipe, "labels": splits.labels}, path)
    return path, splits


def test_inference_returns_probabilities_in_label_order(saved_baseline):
    path, splits = saved_baseline
    predict_proba, labels = load_predictor(path)
    assert labels == splits.labels

    texts = [clean_text(t) for t in splits.test["text"].head(5)]
    probs = predict_proba(texts)
    assert probs.shape == (5, len(labels))
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert ((probs >= 0) & (probs <= 1)).all()


def test_predicted_label_is_one_of_the_saved_labels_and_mostly_right(saved_baseline):
    path, splits = saved_baseline
    predict_proba, labels = load_predictor(path)
    probs = predict_proba([clean_text(t) for t in splits.test["text"]])
    predicted = [labels[i] for i in probs.argmax(axis=1)]
    assert set(predicted) <= set(labels)
    accuracy = np.mean([p == t for p, t in zip(predicted, splits.test["label"])])
    assert accuracy > 0.8  # synthetic classes have distinct vocabularies


def test_missing_model_gives_helpful_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="Train one first"):
        load_predictor(tmp_path / "does-not-exist.joblib")
    with pytest.raises(FileNotFoundError, match="No trained model found"):
        load_predictor(tmp_path)  # an empty directory is not a saved model
