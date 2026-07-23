"""Shared evaluation helpers used by the training and comparison scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Union

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
)


def compute_metrics(y_true: Iterable[int], y_pred: Iterable[int]) -> dict:
    """Compute accuracy/precision/recall/F1 (binary, positive label = 1)."""
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1
    )
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def print_report(y_true: Iterable[int], y_pred: Iterable[int]) -> dict:
    metrics = compute_metrics(y_true, y_pred)
    print(classification_report(y_true, y_pred, digits=4))
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1-score : {metrics['f1']:.4f}")
    return metrics


def save_metrics(metrics: dict, output_path: Union[str, Path]) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {output_path}")
