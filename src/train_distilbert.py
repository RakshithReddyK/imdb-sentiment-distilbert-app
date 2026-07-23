"""Fine-tune DistilBERT for binary sentiment classification on IMDB.

This is the production/reproducible counterpart to
`notebooks/02_imdb_distilbert_finetuning.ipynb`.

IMPORTANT (honesty note): by default this script subsamples the train/test
splits (8000/4000 examples) for fast experimentation on a laptop CPU/GPU,
mirroring what the original notebook did. The README's DistilBERT numbers
were produced with these defaults, NOT on the full 25k/25k split, while the
baseline is trained on the full split. That is a confound: DistilBERT is
compared against the baseline on less data. To train on the full dataset
(directly comparable to the baseline), pass:

    python -m src.train_distilbert --train-samples -1 --test-samples -1

Example (fast experimentation, matches the notebook / README numbers):

    python -m src.train_distilbert --train-samples 8000 --test-samples 4000 --epochs 2
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.data import load_imdb
from src.evaluate import print_report, save_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model-name", type=str, default="distilbert-base-uncased")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument(
        "--train-samples",
        type=int,
        default=8000,
        help="Subsample size for the train split. Pass -1 for the full 25k split.",
    )
    parser.add_argument(
        "--test-samples",
        type=int,
        default=4000,
        help="Subsample size for the test split. Pass -1 for the full 25k split.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="distilbert-imdb-model")
    return parser.parse_args()


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    args = parse_args()
    train_samples = None if args.train_samples == -1 else args.train_samples
    test_samples = None if args.test_samples == -1 else args.test_samples

    if train_samples is not None or test_samples is not None:
        print(
            "WARNING: training on a subsample "
            f"(train={train_samples or 'full'}, test={test_samples or 'full'}). "
            "This is NOT directly comparable to a baseline trained on the full "
            "dataset -- see the README Limitations section."
        )

    print("Loading IMDB dataset...")
    train, test = load_imdb(
        train_sample_size=train_samples, test_sample_size=test_samples, seed=args.seed
    )
    print(f"Train samples: {len(train.texts)} | Test samples: {len(test.texts)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def tokenize(texts):
        return tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        )

    train_enc = tokenize(train.texts)
    test_enc = tokenize(test.texts)

    train_ds = torch.utils.data.TensorDataset(
        train_enc["input_ids"],
        train_enc["attention_mask"],
        torch.tensor(train.labels),
    )
    test_ds = torch.utils.data.TensorDataset(
        test_enc["input_ids"], test_enc["attention_mask"], torch.tensor(test.labels)
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    device = get_device()
    print(f"Using device: {device}")

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=2
    )
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    start = time.time()
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for input_ids, attention_mask, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
                labels=labels.to(device),
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{args.epochs} - training loss: {avg_loss:.4f}")
    train_seconds = time.time() - start

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for input_ids, attention_mask, labels in test_loader:
            outputs = model(
                input_ids=input_ids.to(device), attention_mask=attention_mask.to(device)
            )
            preds = torch.argmax(outputs.logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    metrics = print_report(np.array(all_labels), np.array(all_preds))
    metrics["train_samples"] = len(train.texts)
    metrics["test_samples"] = len(test.texts)
    metrics["train_seconds"] = train_seconds
    metrics["epochs"] = args.epochs
    metrics["subsampled"] = train_samples is not None or test_samples is not None

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    save_metrics(metrics, output_dir / "metrics.json")
    print(f"Saved model, tokenizer, and metrics to {output_dir}")


if __name__ == "__main__":
    main()
