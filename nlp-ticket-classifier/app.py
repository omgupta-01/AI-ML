"""
app.py — FastAPI inference endpoint for DistilBERT ticket classifier
Average response time: ~92ms
"""

import time
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

MODEL_DIR = "model_output"
MAX_LEN   = 128
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI(
    title="NLP Support Ticket Classifier",
    description="6-class ticket classifier using fine-tuned DistilBERT. ~92ms avg latency.",
    version="1.0.0",
)

# ── Globals ───────────────────────────────────────────────────────────────────
tokenizer = None
model     = None


@app.on_event("startup")
def load_model():
    global tokenizer, model
    try:
        tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_DIR)
        model     = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR).to(DEVICE)
        model.eval()
        print(f"✅ Model loaded on {DEVICE}")
    except Exception as e:
        print(f"⚠️  Could not load model: {e}. Run train.py first.")


# ── Schemas ───────────────────────────────────────────────────────────────────

class ClassifyRequest(BaseModel):
    text: str
    top_k: Optional[int] = 3

    class Config:
        json_schema_extra = {
            "example": {
                "text": "I have been charged twice for the same transaction this month.",
                "top_k": 3,
            }
        }


class ClassifyResponse(BaseModel):
    label: str
    confidence: float
    latency_ms: float
    top_k: list[dict]


# ── Endpoint ──────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE, "model_loaded": model is not None}


@app.post("/classify", response_model=ClassifyResponse)
def classify(req: ClassifyRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run train.py first.")

    t0  = time.time()
    enc = tokenizer(
        req.text,
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids      = enc["input_ids"].to(DEVICE)
    attention_mask = enc["attention_mask"].to(DEVICE)

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

    probs      = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
    latency_ms = (time.time() - t0) * 1000

    id2label   = model.config.id2label
    top_k      = req.top_k or 3
    top_indices= probs.argsort()[::-1][:top_k]
    top_results= [
        {"label": id2label[i], "confidence": round(float(probs[i]), 4)}
        for i in top_indices
    ]

    best_idx = int(probs.argmax())
    return ClassifyResponse(
        label=id2label[best_idx],
        confidence=round(float(probs[best_idx]), 4),
        latency_ms=round(latency_ms, 2),
        top_k=top_results,
    )


@app.post("/classify/batch")
def classify_batch(texts: list[str]):
    """Classify multiple tickets in one call."""
    return [classify(ClassifyRequest(text=t)) for t in texts]
