# IMDB Sentiment Analysis Platform – DistilBERT + FastAPI + Streamlit

End-to-end NLP project that fine-tunes a DistilBERT model on the IMDB movie review dataset and serves it via a FastAPI backend with a Streamlit frontend.

The repository contains training notebooks and serving/UI code:
- classical ML baseline (TF–IDF + Logistic Regression)
- transformer fine-tuning (DistilBERT)
- production-style API (FastAPI)
- user-facing web app (Streamlit)

---

## Problem

Classify movie reviews from the IMDB dataset as **Positive** or **Negative**.

- Dataset: IMDB reviews (binary sentiment, 50,000 samples)
- Input: Free-text review (string)
- Output: Sentiment label + confidence

---

## Models

###  Baseline – TF–IDF + Logistic Regression

Implemented in `notebooks/imdb_baseline.ipynb`.

- Vectorizer: `TfidfVectorizer` with max features and English stopword removal
- Classifier: `LogisticRegression`

**Results:**

| Metric     | Score   |
|-----------|---------|
| Accuracy  | **88.12%** |
| F1-score  | **88.14%** |

---

### Fine-Tuned Transformer – DistilBERT

Implemented in `notebooks/imdb_distilbert.ipynb`.

- Model: `distilbert-base-uncased` fine-tuned for binary classification
- Framework: PyTorch + Hugging Face Transformers

**Results on test set:**

| Metric     | Score    |
|-----------|----------|
| Accuracy  | **89.93%** |
| Precision | **88.07%** |
| Recall    | **92.19%** |
| F1-score  | **90.08%** |

> DistilBERT improves F1 by ~2 percentage points over the classical baseline.

The fine-tuned model and tokenizer are saved in:

```text
distilbert-imdb-model/
```
> **Note:** Model weights are not committed to the repo due to GitHub size limits.  
> To run this project, fine-tune DistilBERT using `notebooks/imdb_distilbert.ipynb` and save it as:
>
> ```python
> model.save_pretrained("distilbert-imdb-model")
> tokenizer.save_pretrained("distilbert-imdb-model")
> ```
>
> The FastAPI and Streamlit apps will automatically load from this folder.


## Reproduction status

The tables above are previously reported notebook results; they were not rerun in the October 2026 portfolio audit. Fine-tuned weights are not committed, and the API loads them at import time, so it cannot start until training has produced `distilbert-imdb-model/`. The notebook paths above now match the checked-in filenames. No API load test, production deployment, or model-card evaluation is claimed here.
