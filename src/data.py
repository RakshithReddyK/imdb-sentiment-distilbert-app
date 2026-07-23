"""Shared data loading for the IMDB sentiment project.

Both the classical baseline and the DistilBERT fine-tuning path load the same
raw-text IMDB dataset through Hugging Face `datasets`, so there is a single
consistent data path instead of two (the baseline notebook previously used
`tf.keras.datasets.imdb`, a separately pre-tokenized copy of the corpus).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from datasets import Dataset, load_dataset


@dataclass
class IMDBSplit:
    texts: list
    labels: list


def load_imdb_split(
    split: str,
    sample_size: Optional[int] = None,
    seed: int = 42,
) -> IMDBSplit:
    """Load one split ("train" or "test") of the IMDB dataset as plain lists.

    Args:
        split: "train" or "test".
        sample_size: if given, shuffle with `seed` and take only this many
            examples. Useful for fast experimentation; leave as None to use
            the full 25,000-example split.
        seed: shuffle seed, used only when sample_size is set.
    """
    dataset: Dataset = load_dataset("imdb", split=split)

    if sample_size is not None:
        dataset = dataset.shuffle(seed=seed).select(
            range(min(sample_size, len(dataset)))
        )

    return IMDBSplit(texts=list(dataset["text"]), labels=list(dataset["label"]))


def load_imdb(
    train_sample_size: Optional[int] = None,
    test_sample_size: Optional[int] = None,
    seed: int = 42,
):
    """Convenience helper returning (train, test) IMDBSplit tuples."""
    train = load_imdb_split("train", sample_size=train_sample_size, seed=seed)
    test = load_imdb_split("test", sample_size=test_sample_size, seed=seed)
    return train, test
