"""Fine-tune a multilingual transformer for Nepali news topic classification.

Plain PyTorch training loop on top of Hugging Face Transformers: linear warmup/decay
schedule, mixed precision on CUDA, gradient accumulation and clipping, optional class
weights, model selection on validation macro-F1 with early stopping.

Usage:
    python -m src.train_transformer --data data/news.csv --model_name xlm-roberta-base
"""
from __future__ import annotations

import argparse
import contextlib
import json
import logging
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)

from .data import load_dataset, make_splits, save_label_map
from .metrics import compute_metrics, save_predictions, save_report

log = logging.getLogger("train")


class NewsDataset(Dataset):
    """Pre-tokenised (unpadded) examples; padding is done per batch by the collator."""

    def __init__(self, texts, label_ids, tokenizer, max_length: int):
        enc = tokenizer(list(texts), truncation=True, max_length=max_length)
        self.input_ids = enc["input_ids"]
        self.attention_mask = enc["attention_mask"]
        self.labels = [int(x) for x in label_ids]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int) -> dict:
        return {
            "input_ids": self.input_ids[i],
            "attention_mask": self.attention_mask[i],
            "labels": self.labels[i],
        }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="CSV with a text column and a label column")
    p.add_argument("--text_col", default="text")
    p.add_argument("--label_col", default="label")
    p.add_argument("--model_name", default="xlm-roberta-base",
                   help="HF model id or local directory (e.g. xlm-roberta-base, bert-base-multilingual-cased)")
    p.add_argument("--output_dir", default=None, help="Default: outputs/<model_name>")
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--eval_batch_size", type=int, default=None, help="Default: 2 x batch_size")
    p.add_argument("--grad_accum", type=int, default=1, help="Gradient accumulation steps")
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=2, help="Stop after this many epochs without val macro-F1 gain")
    p.add_argument("--class_weights", action="store_true", help="Weight the loss by inverse class frequency")
    p.add_argument("--no_amp", action="store_true", help="Disable mixed precision on CUDA")
    p.add_argument("--max_train_samples", type=int, default=None, help="Subsample the training set (quick debugging)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val_size", type=float, default=0.1)
    p.add_argument("--test_size", type=float, default=0.1)
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_amp_context(use_amp: bool):
    if use_amp:
        return lambda: torch.autocast("cuda", dtype=torch.float16)
    return contextlib.nullcontext


@torch.no_grad()
def run_eval(model, loader, device, amp_ctx):
    model.eval()
    golds, preds = [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        golds.extend(batch.pop("labels").tolist())
        with amp_ctx():
            logits = model(**batch).logits
        preds.extend(logits.argmax(dim=-1).tolist())
    return np.array(golds), np.array(preds)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    set_seed(args.seed)

    out = Path(args.output_dir or f"outputs/{args.model_name.replace('/', '_')}")
    best_dir = out / "best"
    out.mkdir(parents=True, exist_ok=True)

    device = pick_device()
    use_amp = device.type == "cuda" and not args.no_amp
    amp_ctx = make_amp_context(use_amp)
    log.info("device=%s amp=%s", device, use_amp)

    # ---- data ---------------------------------------------------------------
    df = load_dataset(args.data, args.text_col, args.label_col)
    splits = make_splits(df, args.val_size, args.test_size, args.seed)
    train_df = splits.train
    if args.max_train_samples and args.max_train_samples < len(train_df):
        train_df = train_df.sample(args.max_train_samples, random_state=args.seed).reset_index(drop=True)
    n_labels = len(splits.labels)
    log.info("train=%d val=%d test=%d classes=%d", len(train_df), len(splits.val), len(splits.test), n_labels)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    max_length = min(args.max_length, tokenizer.model_max_length)
    collator = DataCollatorWithPadding(tokenizer)
    eval_bs = args.eval_batch_size or args.batch_size * 2

    def make_ds(frame):
        return NewsDataset(frame["text"], frame["label_id"], tokenizer, max_length)

    train_loader = DataLoader(make_ds(train_df), batch_size=args.batch_size, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(make_ds(splits.val), batch_size=eval_bs, collate_fn=collator)
    test_loader = DataLoader(make_ds(splits.test), batch_size=eval_bs, collate_fn=collator)

    # ---- model / optimiser --------------------------------------------------
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=n_labels,
        id2label=splits.id2label,
        label2id=splits.label2id,
    ).to(device)

    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
    named = list(model.named_parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": [p for n, p in named if not any(nd in n for nd in no_decay)],
             "weight_decay": args.weight_decay},
            {"params": [p for n, p in named if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0},
        ],
        lr=args.lr,
    )
    updates_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    total_updates = updates_per_epoch * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_updates * args.warmup_ratio), total_updates
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    weight = None
    if args.class_weights:
        counts = np.bincount(train_df["label_id"], minlength=n_labels).clip(min=1)
        weight = torch.tensor(counts.sum() / (n_labels * counts), dtype=torch.float, device=device)
        log.info("class weights: %s", np.round(weight.cpu().numpy(), 3).tolist())
    loss_fn = torch.nn.CrossEntropyLoss(weight=weight)

    # ---- training loop ------------------------------------------------------
    history, best_f1, bad_epochs = [], -1.0, 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running = 0.0
        for step, batch in enumerate(train_loader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")
            with amp_ctx():
                logits = model(**batch).logits
            loss = loss_fn(logits.float(), labels) / args.grad_accum
            scaler.scale(loss).backward()
            running += loss.item() * args.grad_accum

            if step % args.grad_accum == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        y_val, p_val = run_eval(model, val_loader, device, amp_ctx)
        val = compute_metrics(y_val, p_val)
        row = {"epoch": epoch, "train_loss": running / len(train_loader),
               **{f"val_{k}": v for k, v in val.items()}, "seconds": round(time.time() - t0, 1)}
        history.append(row)
        log.info("epoch %d | train loss %.4f | val acc %.4f | val macro-F1 %.4f | %.0fs",
                 epoch, row["train_loss"], val["accuracy"], val["macro_f1"], row["seconds"])

        if val["macro_f1"] > best_f1:
            best_f1, bad_epochs = val["macro_f1"], 0
            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)
            log.info("  new best model saved to %s", best_dir)
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                log.info("Early stopping: no improvement for %d epochs", bad_epochs)
                break

    # ---- final evaluation on the held-out test set ---------------------------
    model = AutoModelForSequenceClassification.from_pretrained(best_dir).to(device)
    y_test, p_test = run_eval(model, test_loader, device, amp_ctx)
    metrics = save_report(y_test, p_test, splits.labels, out, prefix="test")
    save_predictions(splits.test, p_test, splits.labels, out / "test_predictions.csv")
    save_label_map(splits.labels, out)
    (out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (out / "run_config.json").write_text(
        json.dumps({**vars(args), "device": str(device), "best_val_macro_f1": best_f1,
                    "n_train": len(train_df), "n_val": len(splits.val), "n_test": len(splits.test)},
                   indent=2),
        encoding="utf-8",
    )
    log.info("TEST  accuracy=%.4f  macro-F1=%.4f", metrics["accuracy"], metrics["macro_f1"])
    log.info("Artifacts written to %s", out)


if __name__ == "__main__":
    main()
