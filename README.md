# IMDB Sentiment Analysis Platform – DistilBERT + FastAPI + Streamlit

End-to-end NLP project that fine-tunes a DistilBERT model on the IMDB movie review
dataset and serves it via a FastAPI backend with a Streamlit frontend, benchmarked
against a classical TF-IDF baseline and a zero/few-shot Claude baseline.

The project demonstrates **full ML lifecycle ownership**:

- classical ML baseline (TF-IDF + Logistic Regression)
- transformer fine-tuning (DistilBERT)
- LLM-as-baseline comparison (Claude zero/few-shot classification)
- production-style API (FastAPI, async, threadpool-offloaded inference)
- user-facing web app (Streamlit)
- tests, CI, and containerized deployment

---

## Architecture

```
                       ┌────────────────────┐
                       │   notebooks/        │  EDA / narrative walkthroughs
                       │   (exploratory)      │  (kept as-is; not the prod path)
                       └────────────────────┘

┌──────────────┐   HTTP    ┌──────────────────┐   loads   ┌─────────────────────┐
│  Streamlit    │ ────────▶│  FastAPI service  │◀─────────│ distilbert-imdb-model│
│  app/app.py   │  /predict │  api/main.py      │  (async,  │ (fine-tuned weights, │
│  (API_URL env)│  /health  │  threadpool infer)│  lazy)    │  not committed)       │
└──────────────┘           └──────────────────┘           └─────────────────────┘
                                                                       ▲
                                                             produced by
                                                                       │
                       ┌───────────────────────────────────────────────────┐
                       │                     src/                          │
                       │  data.py            shared IMDB loader (HF        │
                       │                      `datasets`, one data path)   │
                       │  train_baseline.py   TF-IDF + LogisticRegression  │
                       │  train_distilbert.py DistilBERT fine-tuning       │
                       │  evaluate.py         shared metrics helpers       │
                       │  llm_compare.py       Claude zero/few-shot vs.     │
                       │                      the fine-tuned model         │
                       └───────────────────────────────────────────────────┘
```

- **`src/`** is the production training/evaluation path — CLI scripts with
  argparse options (epochs, batch size, sample size), not hardcoded notebook
  cells. This is what you'd actually run to reproduce or retrain.
- **`notebooks/`** are kept for exploratory data analysis and narrative
  walkthroughs. They are not the source of truth for reproducing metrics.
- **`api/`** is a FastAPI service with an `async` `/predict` route that
  offloads the CPU/GPU-bound tokenization + forward pass to a worker thread
  via `fastapi.concurrency.run_in_threadpool`, so a slow inference call
  doesn't block the event loop for other concurrent requests. Basic request
  logging is included. The model/tokenizer load lazily behind a FastAPI
  dependency, which is also what makes the route unit-testable without
  downloading real weights (see `tests/`).
- **`app/`** is a thin Streamlit UI that calls the API over HTTP, configured
  via the `API_URL` environment variable (see `.env.example`) instead of a
  hardcoded URL.

---

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt          # runtime deps
pip install -r requirements-dev.txt      # + pytest, ruff (dev/CI only)

cp .env.example .env                     # then edit if needed
```

`requirements.txt` is pinned. `requirements-dev.txt` layers pytest/ruff on top
via `-r requirements.txt`, so installing it alone is sufficient for local dev.

### Model weights

Fine-tuned DistilBERT weights are **not committed** to the repo (they exceed
GitHub's size-friendly limits for a portfolio repo). To get a local model:

```bash
python -m src.train_distilbert --train-samples 8000 --test-samples 4000 --epochs 2
# writes to ./distilbert-imdb-model/ by default
```

Both `api/main.py` (via `MODEL_PATH`, default `distilbert-imdb-model`) and the
training script look for/write to that folder.

### Run the API

```bash
uvicorn api.main:app --reload --port 8000
# GET  http://127.0.0.1:8000/health
# POST http://127.0.0.1:8000/predict   {"text": "this movie was great"}
```

### Run the Streamlit app

```bash
export API_URL=http://127.0.0.1:8000     # default if unset
streamlit run app/app.py
```

### Run with Docker Compose

```bash
docker compose up --build
# API:       http://localhost:8000
# Streamlit: http://localhost:8501
```

This expects `./distilbert-imdb-model/` to exist locally (mounted read-only
into the API container) — train it first with the command above, or bake it
into `Dockerfile.api` for a self-contained image.

### Run tests

```bash
pytest tests/ -v
```

Tests use `fastapi.testclient.TestClient` with the model/tokenizer dependency
overridden by a stub (`tests/conftest.py`) — no network access, downloaded
weights, or GPU required. CI (`.github/workflows/ci.yml`) runs `ruff check .`
and this suite on every push/PR.

---

## Models

### 1. Baseline – TF-IDF + Logistic Regression

Production path: `src/train_baseline.py` (`python -m src.train_baseline`).
Exploratory notebook: `notebooks/01_imdb_tfidf_baseline.ipynb`.

- Vectorizer: `TfidfVectorizer` (max 20,000 features, English stopwords removed)
- Classifier: `LogisticRegression`
- Trained and evaluated on the **full** 25,000 / 25,000 train/test split

**Results (full split, `python -m src.train_baseline`):**

| Metric    | Score      |
| --------- | ---------- |
| Accuracy  | **88.00%** |
| Precision | **87.87%** |
| Recall    | **88.16%** |
| F1-score  | **88.02%** |

### 2. Fine-Tuned Transformer – DistilBERT

Production path: `src/train_distilbert.py` (`python -m src.train_distilbert`).
Exploratory notebook: `notebooks/02_imdb_distilbert_finetuning.ipynb`.

- Model: `distilbert-base-uncased` fine-tuned for binary classification
- Framework: PyTorch + Hugging Face Transformers
- **Default run subsamples to 8,000 train / 4,000 test examples** for fast
  experimentation on a laptop CPU/GPU — see **Limitations** below before
  comparing directly to the baseline.

**Results (8k/4k subsample, 2 epochs):**

| Metric    | Score      |
| --------- | ---------- |
| Accuracy  | **89.93%** |
| Precision | **88.07%** |
| Recall    | **92.19%** |
| F1-score  | **90.08%** |

> To train on the full 25k/25k split instead (directly comparable to the
> baseline, but much slower on CPU):
> `python -m src.train_distilbert --train-samples -1 --test-samples -1`

The fine-tuned model and tokenizer are saved to `distilbert-imdb-model/`
(gitignored — see **Model weights** above).

### 3. LLM Zero/Few-Shot Baseline – Claude

Production path: `src/llm_compare.py` (`python -m src.llm_compare`).

This is the part that differentiates this project from a plain BERT
fine-tuning tutorial: instead of only comparing two *trained* models, it also
benchmarks a general-purpose LLM prompted with **no task-specific training**
against the fine-tuned DistilBERT model, using the Claude API
(`anthropic` Python SDK).

- **Zero-shot**: the model is asked to classify a review as Positive/Negative
  with only a system-prompt instruction, no examples.
- **Few-shot**: the same prompt, preceded by 3 labeled example reviews.

**Requires a live API key** (`ANTHROPIC_API_KEY`) to run — see `.env.example`.
Without a key, the script exits with a clear message instead of crashing:

```bash
$ python -m src.llm_compare --num-samples 50
ANTHROPIC_API_KEY is not set. This script calls the Claude API live and
requires a key -- see .env.example and the README's 'LLM comparison'
section for setup instructions and example expected output.
```

**Example run and expected output (with a key set):**

```bash
$ export ANTHROPIC_API_KEY=sk-ant-...
$ python -m src.llm_compare --num-samples 50 --mode zero-shot \
    --distilbert-metrics distilbert-imdb-model/metrics.json

Loading 50 IMDB test samples (seed=42)...
  ...classified 10/50
  ...classified 20/50
  ...classified 30/50
  ...classified 40/50
  ...classified 50/50

Parsed 50/50 responses (0 unparsed/refused).

Claude (claude-opus-4-8, zero-shot) on 50 IMDB test reviews:
  Accuracy : 0.9600
  Precision: 0.9615
  Recall   : 0.9600
  F1-score : 0.9600

Comparison:
Metric      DistilBERT    Claude (zero-shot)
accuracy    0.8993        0.9600
precision   0.8807        0.9615
recall      0.9219        0.9600
f1          0.9008        0.9600

Note: DistilBERT numbers above were trained on a 8000-example subsample --
see the README Limitations section before treating this as apples-to-apples.
```

(Numbers above are illustrative of the expected shape of the output, not a
claim of a live measured result — see **Limitations**.) A capable general
LLM with no task-specific training will usually match or exceed a small
fine-tuned model like DistilBERT on a well-known, relatively easy benchmark
like IMDB sentiment; the interesting engineering question in a real system is
the **cost/latency tradeoff** — a fine-tuned 66M-parameter model running
locally is dramatically cheaper and faster per request at scale than an API
call to a frontier LLM, even if the LLM is more accurate zero-shot. This
script gives you the numbers to make that tradeoff concrete rather than
asserted.

---

## Limitations

- **DistilBERT/baseline comparison is not apples-to-apples by default.**
  The baseline (`src/train_baseline.py`) trains on the full 25k/25k IMDB
  split; the DistilBERT script (`src/train_distilbert.py`) defaults to an
  8k/4k subsample for fast iteration. The subsample confound means
  DistilBERT's reported numbers above are not a controlled comparison against
  the baseline — retrain DistilBERT with `--train-samples -1 --test-samples -1`
  for a same-data comparison (much slower on CPU).
- **No batching in the API.** `/predict` accepts one review per request. For
  production throughput you'd want a `/predict_batch` route and dynamic
  request batching (e.g. via a queue) rather than one forward pass per HTTP
  request.
- **No authentication/rate limiting** on the FastAPI service — fine for a
  local/demo deployment, not for a public endpoint.
- **Max sequence length is 256 tokens** (both DistilBERT training and
  inference); longer reviews are truncated, which can lose information for
  very long reviews.
- **The LLM comparison is not free** — it costs real API tokens and the
  numbers shown in this README are illustrative of expected output shape,
  not a guaranteed live benchmark (model behavior and pricing can change;
  see `src/llm_compare.py` for the exact prompts and parsing used).
- **Single-GPU/CPU training only** — `src/train_distilbert.py` doesn't
  support multi-GPU/distributed training.

---

## Repository structure

```
api/                    FastAPI inference service
  main.py               async /predict + /health, lazy model loading, logging
app/                    Streamlit frontend
  app.py                calls the API via API_URL env var
src/                    production training/evaluation/comparison scripts
  data.py               shared IMDB loader (Hugging Face `datasets`)
  train_baseline.py     TF-IDF + LogisticRegression, CLI args
  train_distilbert.py   DistilBERT fine-tuning, CLI args
  evaluate.py           shared metrics helpers
  llm_compare.py        Claude zero/few-shot baseline vs. DistilBERT
notebooks/              exploratory notebooks (EDA / narrative walkthroughs)
tests/                  pytest suite for the API (model mocked/stubbed)
.github/workflows/ci.yml   lint (ruff) + pytest on push/PR
Dockerfile.api           container for the FastAPI service
Dockerfile.app           container for the Streamlit app
docker-compose.yml       runs both services together
.env.example              API_URL / MODEL_PATH / ANTHROPIC_API_KEY template
requirements.txt          pinned runtime dependencies
requirements-dev.txt      + pytest/ruff for local dev and CI
```
