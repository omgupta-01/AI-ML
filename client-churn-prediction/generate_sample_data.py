"""
generate_sample_data.py — Creates a synthetic churn dataset for demo/testing
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N = 10_000

df = pd.DataFrame({
    "tenure_days":        np.random.randint(30, 1500, N),
    "total_transactions": np.random.randint(1, 200, N),
    "total_value":        np.random.uniform(5000, 5_000_000, N).round(2),
    "product_count":      np.random.randint(1, 8, N),
    "support_tickets":    np.random.poisson(1.5, N),
    "age":                np.random.randint(22, 70, N),
    "segment":            np.random.choice(["retail", "HNI", "corporate"], N),
})

df["avg_txn_value"]    = (df["total_value"] / (df["total_transactions"] + 1)).round(2)
df["txn_per_day"]      = (df["total_transactions"] / (df["tenure_days"] + 1)).round(4)
df["days_since_active"]= np.random.randint(0, 180, N)

# Churn logic: short tenure + few txns + many tickets → higher churn probability
churn_score = (
    (df["tenure_days"] < 180).astype(int) * 0.3
    + (df["total_transactions"] < 10).astype(int) * 0.25
    + (df["support_tickets"] > 3).astype(int) * 0.2
    + (df["days_since_active"] > 90).astype(int) * 0.25
)
df["churn"] = (churn_score + np.random.uniform(0, 0.3, N) > 0.55).astype(int)

import os
os.makedirs("data", exist_ok=True)
df.to_csv("data/churn_data.csv", index=False)
print(f"Saved data/churn_data.csv  ({N:,} rows)")
print(f"Churn rate: {df['churn'].mean():.1%}")
