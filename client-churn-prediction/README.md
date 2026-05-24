# 📉 Client Churn Prediction — Financial Services

An end-to-end machine learning pipeline that predicts client churn for financial services using **XGBoost**, **SHAP explainability**, and a **FastAPI** inference endpoint.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green?logo=fastapi)
![SHAP](https://img.shields.io/badge/SHAP-explainability-purple)

---

## 🎯 Results

| Metric | Value |
|--------|-------|
| Accuracy | **84%** |
| ROC-AUC | ~0.91 |
| Churn reduction (business) | ~15% |
| Dataset size | 10M+ rows |

---

## 🚀 Features

- **End-to-end pipeline** — EDA → Feature Engineering → Model Training → Evaluation → Deployment
- **XGBoost + GridSearchCV** — Hyperparameter tuning with 5-fold stratified cross-validation
- **SHAP explainability** — Feature importance visualizations for every prediction
- **FastAPI REST endpoint** — Real-time churn probability scores with risk levels (LOW / MEDIUM / HIGH)
- **Sample data generator** — Synthetic dataset included for immediate demo

---

## 🏗️ Pipeline Architecture

```
Raw CSV Data
     │
     ▼
EDA Summary → Feature Engineering (tenure, ratios, encoding)
     │
     ▼
Train/Test Split (80/20, stratified)
     │
     ▼
GridSearchCV (XGBoost + StandardScaler Pipeline)
     │
     ├── Confusion Matrix
     ├── Classification Report
     ├── SHAP Summary Plot
     └── Model saved → outputs/churn_model.pkl
                              │
                              ▼
                      FastAPI /predict endpoint
```

---

## ⚙️ Setup

```bash
git clone https://github.com/omgupta/client-churn-prediction.git
cd client-churn-prediction
pip install -r requirements.txt
```

### 1. Generate sample data

```bash
python generate_sample_data.py
```

### 2. Train the model

```bash
python train.py
```

Outputs saved to `outputs/`:
- `churn_model.pkl` — trained pipeline
- `confusion_matrix.png`
- `shap_summary.png`

### 3. Start inference API

```bash
uvicorn api:app --reload
```

---

## 📡 API Usage

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "tenure_days": 90,
    "total_transactions": 5,
    "total_value": 15000,
    "avg_txn_value": 3000,
    "txn_per_day": 0.05,
    "days_since_active": 120,
    "product_count": 1,
    "support_tickets": 5,
    "age": 35,
    "segment": 0
  }'
```

**Response:**
```json
{
  "churn_probability": 0.8342,
  "churn_prediction": true,
  "risk_level": "HIGH"
}
```

---

## 📁 Project Structure

```
client-churn-prediction/
├── train.py                  # Full ML pipeline
├── api.py                    # FastAPI inference server
├── generate_sample_data.py   # Synthetic data generator
├── requirements.txt
├── data/
│   └── churn_data.csv        # Auto-generated
└── outputs/
    ├── churn_model.pkl
    ├── confusion_matrix.png
    └── shap_summary.png
```

---

## 🛠️ Tech Stack

`Python` · `XGBoost` · `Scikit-learn` · `SHAP` · `Pandas` · `NumPy` · `FastAPI` · `Matplotlib`

---

## 👤 Author

**Om Gupta** · [omguptabca2020@gmail.com](mailto:omguptabca2020@gmail.com)  
Software Developer — Data & ML @ Bonanza Portfolio Ltd
