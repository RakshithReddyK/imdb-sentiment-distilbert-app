"""Train the TF-IDF + Logistic Regression baseline on IMDB.

This is the production/reproducible counterpart to
`notebooks/01_imdb_tfidf_baseline.ipynb`. Run it directly, e.g.:

    python -m src.train_baseline --max-features 20000 --output-dir artifacts/baseline

By default it trains on the full 25,000-example train split and evaluates on
the full 25,000-example test split (no subsampling), matching the numbers
reported in the README.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.data import load_imdb
from src.evaluate import print_report, save_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-features", type=int, default=20000)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument(
        "--train-samples",
        type=int,
        default=None,
        help="Subsample the training split to this many examples (default: full 25k).",
    )
    parser.add_argument(
        "--test-samples",
        type=int,
        default=None,
        help="Subsample the test split to this many examples (default: full 25k).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/baseline",
        help="Directory to save the fitted vectorizer, classifier, and metrics.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading IMDB dataset...")
    train, test = load_imdb(
        train_sample_size=args.train_samples,
        test_sample_size=args.test_samples,
        seed=args.seed,
    )
    print(f"Train samples: {len(train.texts)} | Test samples: {len(test.texts)}")

    vectorizer = TfidfVectorizer(max_features=args.max_features, stop_words="english")
    X_train = vectorizer.fit_transform(train.texts)
    X_test = vectorizer.transform(test.texts)

    clf = LogisticRegression(max_iter=args.max_iter, n_jobs=-1)

    start = time.time()
    clf.fit(X_train, train.labels)
    train_seconds = time.time() - start
    print(f"Training took {train_seconds:.1f}s")

    y_pred = clf.predict(X_test)
    metrics = print_report(test.labels, y_pred)
    metrics["train_samples"] = len(train.texts)
    metrics["test_samples"] = len(test.texts)
    metrics["train_seconds"] = train_seconds

    save_metrics(metrics, output_dir / "metrics.json")
    joblib.dump(vectorizer, output_dir / "vectorizer.joblib")
    joblib.dump(clf, output_dir / "classifier.joblib")
    print(f"Saved vectorizer and classifier to {output_dir}")


if __name__ == "__main__":
    main()
