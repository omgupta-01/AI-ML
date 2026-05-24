# 🎫 NLP Support Ticket Classifier

A fine-tuned **DistilBERT** model for 6-class support ticket classification, served via **FastAPI** with ~92ms average response time. Trained on 15,000 labeled financial support tickets.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?logo=huggingface)
![PyTorch](https://img.shields.io/badge/PyTorch-2.3-red?logo=pytorch)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green?logo=fastapi)

---

## 🏷️ Classes

| Label | Description |
|-------|-------------|
| `billing` | Payment issues, invoices, refunds |
| `account` | Login, password, profile management |
| `technical` | Bugs, crashes, API errors |
| `compliance` | KYC, regulatory, legal documents |
| `onboarding` | New user setup, account activation |
| `general` | General inquiries, product questions |

---

## 🎯 Results

| Metric | Value |
|--------|-------|
| Training samples | 15,000 |
| Classes | 6 |
| Validation accuracy | ~94% |
| Avg inference latency | **~92ms** |
| Manual routing reduction | **40%** |

---

## ⚙️ Setup

```bash
git clone https://github.com/omgupta/nlp-ticket-classifier.git
cd nlp-ticket-classifier
pip install -r requirements.txt
```

### 1. Generate sample data

```bash
python generate_sample_data.py
```

### 2. Fine-tune DistilBERT

```bash
python train.py
```

Training runs for 4 epochs. Best model saved to `model_output/`.  
Estimated time: ~15 min on GPU, ~60 min on CPU.

### 3. Start the API

```bash
uvicorn app:app --reload
```

---

## 📡 API Usage

### Single classification

```bash
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "I have been charged twice for the same transaction.", "top_k": 3}'
```

**Response:**
```json
{
  "label": "billing",
  "confidence": 0.9712,
  "latency_ms": 91.4,
  "top_k": [
    {"label": "billing",   "confidence": 0.9712},
    {"label": "account",   "confidence": 0.0183},
    {"label": "technical", "confidence": 0.0062}
  ]
}
```

### Batch classification

```bash
curl -X POST http://localhost:8000/classify/batch \
  -H "Content-Type: application/json" \
  -d '["Cannot log in", "Invoice is wrong", "App crashes"]'
```

---

## 🏗️ Model Architecture

```
Input Text
    │
    ▼
DistilBertTokenizerFast (max_len=128)
    │
    ▼
DistilBERT Base (66M params, distilled from BERT-base)
    │
    ▼
[CLS] representation → Linear(768 → 6) → Softmax
    │
    ▼
6-class probabilities
```

**Fine-tuning details:**
- Base model: `distilbert-base-uncased`
- Optimizer: AdamW (lr=2e-5, weight_decay=0.01)
- Schedule: Linear warmup (10%) + linear decay
- Batch size: 32 | Epochs: 4 | Max seq length: 128

---

## 📁 Project Structure

```
nlp-ticket-classifier/
├── train.py                   # Fine-tuning script
├── app.py                     # FastAPI inference server
├── generate_sample_data.py    # Synthetic data generator
├── requirements.txt
├── data/
│   └── tickets.csv            # Auto-generated
├── model_output/              # Saved model (after training)
└── outputs/
    └── training_history.json
```

---

## 🛠️ Tech Stack

`Python` · `PyTorch` · `HuggingFace Transformers` · `DistilBERT` · `FastAPI` · `Uvicorn` · `Pandas` · `Scikit-learn`

