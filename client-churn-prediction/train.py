"""
train.py — End-to-end ML pipeline for client churn prediction
Stack: Scikit-learn · XGBoost · SHAP · Pandas
"""

import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
)
from xgboost import XGBClassifier


# ── 1. Load Data ───────────────────────────────────────────────────────────────

def load_data(path: str = "data/churn_data.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"[Data] Loaded {df.shape[0]:,} rows × {df.shape[1]} cols")
    return df


# ── 2. EDA Summary ─────────────────────────────────────────────────────────────

def eda_summary(df: pd.DataFrame) -> None:
    print("\n── EDA Summary ──────────────────────────────────")
    print(df.dtypes.value_counts().rename("dtype count"))
    print(f"\nMissing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    print(f"\nChurn distribution:\n{df['churn'].value_counts(normalize=True).round(3)}")
    print("─────────────────────────────────────────────────\n")


# ── 3. Feature Engineering ─────────────────────────────────────────────────────

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Date-based features
    if "signup_date" in df.columns and "last_activity_date" in df.columns:
        df["signup_date"]        = pd.to_datetime(df["signup_date"])
        df["last_activity_date"] = pd.to_datetime(df["last_activity_date"])
        df["tenure_days"]        = (df["last_activity_date"] - df["signup_date"]).dt.days
        df["days_since_active"]  = (pd.Timestamp.today() - df["last_activity_date"]).dt.days
        df.drop(columns=["signup_date", "last_activity_date"], inplace=True)

    # Ratio features
    if "total_transactions" in df.columns and "tenure_days" in df.columns:
        df["txn_per_day"] = df["total_transactions"] / (df["tenure_days"] + 1)

    if "total_value" in df.columns and "total_transactions" in df.columns:
        df["avg_txn_value"] = df["total_value"] / (df["total_transactions"] + 1)

    # Encode categoricals
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    if "churn" in cat_cols:
        cat_cols.remove("churn")
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    # Fill missing
    df.fillna(df.median(numeric_only=True), inplace=True)

    print(f"[FE] Feature matrix: {df.shape[0]:,} rows × {df.shape[1]} cols")
    return df


# ── 4. Train / Evaluate ────────────────────────────────────────────────────────

def train_and_evaluate(df: pd.DataFrame) -> dict:
    X = df.drop(columns=["churn"])
    y = df["churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[Split] Train={len(X_train):,}  Test={len(X_test):,}")

    # Hyperparameter grid
    param_grid = {
        "clf__n_estimators":    [200, 400],
        "clf__max_depth":       [4, 6],
        "clf__learning_rate":   [0.05, 0.1],
        "clf__subsample":       [0.8],
        "clf__colsample_bytree":[0.8],
    }

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    XGBClassifier(
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
        )),
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    gs = GridSearchCV(pipeline, param_grid, cv=cv, scoring="roc_auc", n_jobs=-1, verbose=1)

    print("[Train] Running GridSearchCV …")
    gs.fit(X_train, y_train)

    best_model = gs.best_estimator_
    print(f"[Train] Best params: {gs.best_params_}")

    # Evaluate
    y_pred  = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print(f"\n── Results ───────────────────────────────────────")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  ROC-AUC  : {auc:.4f}")
    print(f"\n{classification_report(y_test, y_pred)}")

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("outputs/confusion_matrix.png", dpi=150)
    plt.close()

    # Save model
    joblib.dump(best_model, "outputs/churn_model.pkl")
    print("\n[Save] Model saved → outputs/churn_model.pkl")

    return {"model": best_model, "X_train": X_train, "X_test": X_test,
            "y_test": y_test, "features": list(X.columns)}


# ── 5. SHAP Explainability ─────────────────────────────────────────────────────

def explain_with_shap(model_pipeline, X_train: pd.DataFrame, X_test: pd.DataFrame) -> None:
    print("[SHAP] Computing feature importance …")
    clf        = model_pipeline.named_steps["clf"]
    scaler     = model_pipeline.named_steps["scaler"]
    X_train_sc = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns)
    X_test_sc  = pd.DataFrame(scaler.transform(X_test),  columns=X_test.columns)

    explainer  = shap.TreeExplainer(clf)
    shap_vals  = explainer.shap_values(X_test_sc)

    # Summary plot
    plt.figure()
    shap.summary_plot(shap_vals, X_test_sc, show=False)
    plt.tight_layout()
    plt.savefig("outputs/shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[SHAP] Summary plot saved → outputs/shap_summary.png")


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    os.makedirs("outputs", exist_ok=True)

    df       = load_data()
    eda_summary(df)
    df       = feature_engineering(df)
    results  = train_and_evaluate(df)
    explain_with_shap(
        results["model"],
        results["X_train"],
        results["X_test"],
    )
    print("\n✅ Pipeline complete. Check outputs/ folder.")
