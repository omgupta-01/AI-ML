"""
train.py — Fine-tune DistilBERT on support ticket classification
Stack: HuggingFace Transformers · PyTorch · FastAPI
Classes: billing | account | technical | compliance | onboarding | general
"""

import os
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# ── Config ────────────────────────────────────────────────────────────────────

LABEL2ID = {
    "billing":    0,
    "account":    1,
    "technical":  2,
    "compliance": 3,
    "onboarding": 4,
    "general":    5,
}
ID2LABEL   = {v: k for k, v in LABEL2ID.items()}
NUM_LABELS = len(LABEL2ID)

MODEL_NAME   = "distilbert-base-uncased"
MAX_LEN      = 128
BATCH_SIZE   = 32
EPOCHS       = 4
LR           = 2e-5
WARMUP_RATIO = 0.1
OUTPUT_DIR   = "model_output"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

print(f"[Config] Device: {DEVICE}")


# ── Dataset ───────────────────────────────────────────────────────────────────

class TicketDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts     = texts
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "labels":         torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ── Training ──────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss, total_correct = 0, 0

    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss    = outputs.loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss    += loss.item()
        preds          = torch.argmax(outputs.logits, dim=1)
        total_correct += (preds == labels).sum().item()

    return total_loss / len(loader), total_correct / len(loader.dataset)


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds   = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return all_labels, all_preds


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    # Load data
    df = pd.read_csv("data/tickets.csv")
    print(f"[Data] {len(df):,} tickets loaded")
    print(df["label"].value_counts())

    texts  = df["text"].tolist()
    labels = [LABEL2ID[l] for l in df["label"].tolist()]

    X_train, X_val, y_train, y_val = train_test_split(
        texts, labels, test_size=0.15, random_state=42, stratify=labels
    )

    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

    train_ds = TicketDataset(X_train, y_train, tokenizer, MAX_LEN)
    val_ds   = TicketDataset(X_val,   y_val,   tokenizer, MAX_LEN)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # Model
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    ).to(DEVICE)

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps   = len(train_loader) * EPOCHS
    warmup_steps  = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    best_acc = 0.0
    history  = []

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, DEVICE)
        val_labels, val_preds = evaluate(model, val_loader, DEVICE)
        val_acc = accuracy_score(val_labels, val_preds)

        elapsed = time.time() - t0
        print(f"Epoch {epoch}/{EPOCHS} | loss={train_loss:.4f} | "
              f"train_acc={train_acc:.4f} | val_acc={val_acc:.4f} | {elapsed:.0f}s")

        history.append({"epoch": epoch, "train_loss": train_loss,
                         "train_acc": train_acc, "val_acc": val_acc})

        if val_acc > best_acc:
            best_acc = val_acc
            model.save_pretrained(OUTPUT_DIR)
            tokenizer.save_pretrained(OUTPUT_DIR)
            print(f"  ✅ Best model saved (val_acc={best_acc:.4f})")

    # Final report
    print("\n── Classification Report ─────────────────────────")
    print(classification_report(val_labels, val_preds, target_names=list(LABEL2ID.keys())))

    Path("outputs").mkdir(exist_ok=True)
    with open("outputs/training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n✅ Done. Best val accuracy: {best_acc:.4f}")
    print(f"   Model saved → {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
