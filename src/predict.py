"""Classify new Nepali text with a trained baseline or transformer.

This classifies the TOPIC of a text. It does not translate, summarise or fact-check.

Usage:
    python -m src.predict --model outputs/baseline/model.joblib --text "नेपाली क्रिकेट टोलीले जित्यो"
    python -m src.predict --model outputs/xlm-roberta-base/best --file headlines.txt --top_k 3
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .text_utils import clean_text

MISSING_MODEL_HELP = (
    "No trained model found at '{path}'.\n"
    "Train one first, for example:\n"
    "  python -m src.baseline --data data/news.csv --output_dir outputs/baseline\n"
    "  python -m src.train_transformer --config configs/xlm-roberta-base.json\n"
    "Model weights are not stored in the Git repository (see the README, 'Predict')."
)


def load_predictor(model_path: str | Path):
    """Return ``(predict_proba_fn, labels)`` for a .joblib baseline or a saved HF model directory.

    ``predict_proba_fn`` maps a list of cleaned texts to an array of shape (n_texts, n_labels);
    column ``i`` corresponds to ``labels[i]``.
    """
    path = Path(model_path)
    if path.is_file() and path.suffix == ".joblib":
        import joblib

        bundle = joblib.load(path)
        pipeline, labels = bundle["pipeline"], bundle["labels"]
        return (lambda texts: pipeline.predict_proba(texts)), labels

    if path.is_dir() and (path / "config.json").exists():
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForSequenceClassification.from_pretrained(path).to(device).eval()
        labels = [model.config.id2label[i] for i in range(model.config.num_labels)]
        max_length = min(256, tokenizer.model_max_length)

        @torch.no_grad()
        def predict_proba(texts, batch_size: int = 32):
            probs = []
            for i in range(0, len(texts), batch_size):
                enc = tokenizer(texts[i:i + batch_size], truncation=True, max_length=max_length,
                                padding=True, return_tensors="pt").to(device)
                probs.append(torch.softmax(model(**enc).logits, dim=-1).cpu().numpy())
            return np.concatenate(probs)

        return predict_proba, labels

    raise FileNotFoundError(MISSING_MODEL_HELP.format(path=model_path))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="outputs/baseline/model.joblib or outputs/<name>/best")
    p.add_argument("--text", nargs="+", help="One or more texts to classify")
    p.add_argument("--file", help="Text file with one article/headline per line")
    p.add_argument("--top_k", type=int, default=3)
    args = p.parse_args()

    texts = list(args.text or [])
    if args.file:
        texts += [line for line in Path(args.file).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not texts:
        p.error("provide --text and/or --file")

    try:
        predict_proba, labels = load_predictor(args.model)
    except FileNotFoundError as err:
        raise SystemExit(str(err))
    probs = predict_proba([clean_text(t) for t in texts])
    for text, row in zip(texts, probs):
        top = np.argsort(row)[::-1][: args.top_k]
        print(f"\n{text[:100]}")
        for i in top:
            print(f"  {labels[i]:<20} {row[i]:.3f}")


if __name__ == "__main__":
    main()
