# IMDB Sentiment Analysis Platform – DistilBERT + FastAPI + Streamlit

End-to-end NLP project that fine-tunes a DistilBERT model on the IMDB movie review dataset and serves it via a FastAPI backend with a Streamlit frontend.

The project demonstrates **full ML lifecycle ownership**:
- classical ML baseline (TF–IDF + Logistic Regression)
- transformer fine-tuning (DistilBERT)
- production-style API (FastAPI)
- user-facing web app (Streamlit)

---

## 🔍 Problem

Classify movie reviews from the IMDB dataset as **Positive** or **Negative**.

- Dataset: IMDB reviews (binary sentiment, 50,000 samples)
- Input: Free-text review (string)
- Output: Sentiment label + confidence

---

## 🧠 Models

### 1️⃣ Baseline – TF–IDF + Logistic Regression

Implemented in `notebooks/01_imdb_tfidf_baseline.ipynb`.

- Vectorizer: `TfidfVectorizer` with max features and English stopword removal
- Classifier: `LogisticRegression`

**Results:**

| Metric     | Score   |
|-----------|---------|
| Accuracy  | **88.12%** |
| F1-score  | **88.14%** |

---

### 2️⃣ Fine-Tuned Transformer – DistilBERT

Implemented in `notebooks/02_imdb_distilbert_finetuning.ipynb`.

- Model: `distilbert-base-uncased` fine-tuned for binary classification
- Framework: PyTorch + Hugging Face Transformers

**Results on test set:**

| Metric     | Score    |
|-----------|----------|
| Accuracy  | **89.93%** |
| Precision | **88.07%** |
| Recall    | **92.19%** |
| F1-score  | **90.08%** |

> ✅ DistilBERT improves F1 by ~2 percentage points over the classical baseline.

The fine-tuned model and tokenizer are saved in:

```text
distilbert-imdb-model/

