"""Zero/few-shot LLM sentiment classification, benchmarked against the
fine-tuned DistilBERT model.

Motivation: a fine-tuned transformer is the "classic" 2023-era portfolio
project. What differentiates this repo for 2026 GenAI-flavored roles is
showing you also know how to evaluate a general-purpose LLM as a zero-shot
or few-shot baseline for the *same* task, with no task-specific training at
all -- and to be honest about the cost/latency/accuracy tradeoffs between
"prompt an LLM" and "fine-tune a small model" in a real system.

This script calls the Claude API (via the official `anthropic` Python SDK)
to classify a sample of IMDB test reviews as Positive/Negative, computes the
same accuracy/precision/recall/F1 metrics used elsewhere in this repo, and
prints a comparison table against a DistilBERT metrics.json (if provided).

Requires an API key:

    export ANTHROPIC_API_KEY=sk-ant-...
    python -m src.llm_compare --num-samples 50 --mode zero-shot

Without a key set, the script exits with a clear message instead of
crashing -- see the README for example expected output.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

from src.data import load_imdb_split
from src.evaluate import compute_metrics, save_metrics

ZERO_SHOT_SYSTEM = (
    "You classify movie review sentiment. Respond with exactly one word: "
    "either \"Positive\" or \"Negative\". No punctuation, no explanation."
)

FEW_SHOT_EXAMPLES = [
    (
        "This film wasted two hours of my life. Wooden acting, a plot that "
        "goes nowhere, and dialogue that made me cringe throughout.",
        "Negative",
    ),
    (
        "A genuinely moving story with career-best performances. I was "
        "hooked from the first scene and thought about it for days after.",
        "Positive",
    ),
    (
        "Competent but forgettable. Nothing here is actively bad, it's just "
        "not memorable in any way.",
        "Negative",
    ),
]


def build_messages(review_text: str, mode: str) -> list:
    messages = []
    if mode == "few-shot":
        for example_text, example_label in FEW_SHOT_EXAMPLES:
            messages.append({"role": "user", "content": example_text})
            messages.append({"role": "assistant", "content": example_label})
    messages.append({"role": "user", "content": review_text})
    return messages


def parse_label(raw_text: str) -> Optional[int]:
    """Map the model's text response to an IMDB label (1=positive, 0=negative)."""
    normalized = raw_text.strip().lower()
    if normalized.startswith("pos"):
        return 1
    if normalized.startswith("neg"):
        return 0
    return None


def classify_review(
    client, model: str, review_text: str, mode: str, max_chars: int
) -> Optional[int]:
    truncated = review_text[:max_chars]
    messages = build_messages(truncated, mode)

    response = client.messages.create(
        model=model,
        max_tokens=16,
        system=ZERO_SHOT_SYSTEM,
        messages=messages,
    )

    if response.stop_reason == "refusal":
        return None

    text = "".join(block.text for block in response.content if block.type == "text")
    return parse_label(text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--num-samples", type=int, default=50, help="Number of IMDB test reviews to classify."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="claude-opus-4-8",
        help="Claude model ID to use (default: claude-opus-4-8).",
    )
    parser.add_argument(
        "--mode",
        choices=["zero-shot", "few-shot"],
        default="zero-shot",
        help="zero-shot: no examples. few-shot: 3 labeled examples prepended.",
    )
    parser.add_argument(
        "--max-chars", type=int, default=2000, help="Truncate each review to this many characters."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sleep", type=float, default=0.0, help="Seconds to sleep between API calls."
    )
    parser.add_argument(
        "--distilbert-metrics",
        type=str,
        default=None,
        help="Path to a metrics.json from src/train_distilbert.py for a side-by-side comparison.",
    )
    parser.add_argument("--output-dir", type=str, default="artifacts/llm_compare")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print(
            "ANTHROPIC_API_KEY is not set. This script calls the Claude API live "
            "and requires a key -- see .env.example and the README's "
            "'LLM comparison' section for setup instructions and example "
            "expected output.",
            file=sys.stderr,
        )
        return 1

    try:
        import anthropic
    except ImportError:
        print(
            "The 'anthropic' package is not installed. Run: pip install anthropic",
            file=sys.stderr,
        )
        return 1

    client = anthropic.Anthropic(api_key=api_key)

    print(f"Loading {args.num_samples} IMDB test samples (seed={args.seed})...")
    test = load_imdb_split("test", sample_size=args.num_samples, seed=args.seed)

    y_true, y_pred = [], []
    unparsed = 0

    for i, (text, label) in enumerate(zip(test.texts, test.labels)):
        try:
            pred = classify_review(client, args.model, text, args.mode, args.max_chars)
        except anthropic.RateLimitError as exc:
            retry_after = int(exc.response.headers.get("retry-after", "10")) if exc.response else 10
            print(f"Rate limited, sleeping {retry_after}s...", file=sys.stderr)
            time.sleep(retry_after)
            continue
        except anthropic.AuthenticationError:
            print("Invalid ANTHROPIC_API_KEY.", file=sys.stderr)
            return 1
        except anthropic.APIConnectionError as exc:
            print(f"Network error calling Claude API: {exc}", file=sys.stderr)
            return 1
        except anthropic.APIStatusError as exc:
            print(f"API error ({exc.status_code}): {exc.message}", file=sys.stderr)
            continue

        if pred is None:
            unparsed += 1
            continue

        y_true.append(label)
        y_pred.append(pred)

        if (i + 1) % 10 == 0:
            print(f"  ...classified {i + 1}/{args.num_samples}")

        if args.sleep:
            time.sleep(args.sleep)

    if not y_true:
        print("No reviews were successfully classified.", file=sys.stderr)
        return 1

    print(f"\nParsed {len(y_true)}/{args.num_samples} responses ({unparsed} unparsed/refused).")
    metrics = compute_metrics(y_true, y_pred)
    metrics["model"] = args.model
    metrics["mode"] = args.mode
    metrics["num_samples"] = len(y_true)
    metrics["unparsed"] = unparsed

    print(f"\nClaude ({args.model}, {args.mode}) on {len(y_true)} IMDB test reviews:")
    print(f"  Accuracy : {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall   : {metrics['recall']:.4f}")
    print(f"  F1-score : {metrics['f1']:.4f}")

    if args.distilbert_metrics:
        db_path = Path(args.distilbert_metrics)
        if db_path.exists():
            db_metrics = json.loads(db_path.read_text())
            print("\nComparison:")
            print(f"{'Metric':<12}{'DistilBERT':<14}{'Claude (' + args.mode + ')':<18}")
            for key in ("accuracy", "precision", "recall", "f1"):
                print(f"{key:<12}{db_metrics.get(key, float('nan')):<14.4f}{metrics[key]:<18.4f}")
            note = db_metrics.get("subsampled")
            if note:
                print(
                    "\nNote: DistilBERT numbers above were trained on a "
                    f"{db_metrics.get('train_samples', '?')}-example subsample -- "
                    "see the README Limitations section before treating this as apples-to-apples."
                )
        else:
            print(f"\n(--distilbert-metrics path not found: {db_path})")

    save_metrics(metrics, Path(args.output_dir) / f"claude_{args.mode}_metrics.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
