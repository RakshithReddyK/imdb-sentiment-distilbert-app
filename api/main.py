import logging
import os
import time
from functools import lru_cache
from typing import Tuple

import numpy as np
import torch
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_PATH = os.environ.get("MODEL_PATH", "distilbert-imdb-model")

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("imdb_sentiment_api")

app = FastAPI(title="IMDB Sentiment API", version="1.0.0")


class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    label: str
    confidence: float


# ---------- Model loading ---------- #
#
# The model/tokenizer are loaded lazily behind a FastAPI dependency (rather
# than at import time) so that:
#   1. importing this module doesn't require the model weights to be on disk
#      or a GPU to be present, and
#   2. tests can override `get_pipeline` with a stub via
#      `app.dependency_overrides`, without downloading real DistilBERT
#      weights or needing a GPU.


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@lru_cache(maxsize=1)
def load_pipeline() -> Tuple[AutoTokenizer, AutoModelForSequenceClassification, torch.device]:
    device = get_device()
    logger.info("Loading model from %s onto device=%s", MODEL_PATH, device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.to(device)
    model.eval()
    return tokenizer, model, device


def get_pipeline() -> Tuple[AutoTokenizer, AutoModelForSequenceClassification, torch.device]:
    """FastAPI dependency returning (tokenizer, model, device).

    Overridden in tests via `app.dependency_overrides[get_pipeline]`.
    """
    return load_pipeline()


def run_inference(
    text: str, tokenizer, model, device: torch.device
) -> PredictResponse:
    """CPU/GPU-bound tokenization + forward pass. Runs off the event loop
    (see `run_in_threadpool` usage in the /predict route below)."""
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

    pred_idx = int(np.argmax(probs))
    confidence = float(probs[pred_idx])

    # IMDB labels: 0 = negative, 1 = positive
    label = "Positive" if pred_idx == 1 else "Negative"
    return PredictResponse(label=label, confidence=confidence)


# ---------- Routes ---------- #


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
async def predict(
    req: PredictRequest,
    request: Request,
    pipeline: Tuple[AutoTokenizer, AutoModelForSequenceClassification, torch.device] = Depends(
        get_pipeline
    ),
):
    text = req.text.strip()
    start = time.perf_counter()

    if not text:
        logger.info("predict rejected: empty text (client=%s)", request.client)
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    tokenizer, model, device = pipeline

    # Tokenization + the forward pass are CPU/GPU-bound and block the
    # thread they run on; offload them to a worker thread so the event
    # loop stays free to handle other requests concurrently.
    result = await run_in_threadpool(run_inference, text, tokenizer, model, device)

    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "predict label=%s confidence=%.4f chars=%d elapsed_ms=%.1f",
        result.label,
        result.confidence,
        len(text),
        elapsed_ms,
    )
    return result
