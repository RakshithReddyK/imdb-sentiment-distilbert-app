from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

MODEL_PATH = "distilbert-imdb-model"

app = FastAPI(title="IMDB Sentiment API", version="1.0.0")


class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    label: str
    confidence: float


# ---------- Model loading ---------- #

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


device = get_device()
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(device)
model.eval()


# ---------- Routes ---------- #

@app.get("/health")
def health():
    return {"status": "ok", "device": str(device)}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

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
