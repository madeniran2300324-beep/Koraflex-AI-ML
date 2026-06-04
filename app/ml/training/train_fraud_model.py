"""Day 6 — Train the supervised fraud model (XGBoost) + anomaly model (IsolationForest).

Run:
    python -m app.ml.training.train_fraud_model

Writes artifacts to app/ml/artifacts/.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from app.ml.training.bias_audit import bias_report
from app.ml.training.synthetic import FEATURE_ORDER, make_dataset

ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


def train() -> dict:
    df = make_dataset()
    X = df[FEATURE_ORDER].values
    y = df["is_fraud"].values
    demo = df["demographic_group"].values

    X_train, X_test, y_train, y_test, demo_train, demo_test = train_test_split(
        X, y, demo, test_size=0.2, random_state=42, stratify=y
    )

    pos_weight = float((y_train == 0).sum() / max(1, (y_train == 1).sum()))

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            scale_pos_weight=pos_weight,
            tree_method="hist",
            eval_metric="aucpr",
            n_jobs=-1,
            random_state=42,
        )),
    ])
    pipeline.fit(X_train, y_train)

    proba = pipeline.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)

    metrics = {
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "pr_auc": float(average_precision_score(y_test, proba)),
        "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
        "report": classification_report(y_test, preds, output_dict=True),
        "trained_at": datetime.utcnow().isoformat(),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }

    bias = bias_report(y_true=y_test, y_pred=preds, sensitive=demo_test)

    version = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    joblib.dump(pipeline, ARTIFACT_DIR / "fraud_xgb.joblib")
    (ARTIFACT_DIR / "fraud_xgb.json").write_text(json.dumps({
        "version": version,
        "feature_order": FEATURE_ORDER,
        "metrics": metrics,
        "bias_audit": bias,
    }, indent=2))

    # --- Unsupervised anomaly model on legit-only data ---
    legit = df[df["is_fraud"] == 0][FEATURE_ORDER].values
    iforest = IsolationForest(
        n_estimators=200, contamination=0.05, random_state=42, n_jobs=-1
    )
    iforest.fit(legit)
    joblib.dump(iforest, ARTIFACT_DIR / "anomaly_iforest.joblib")
    (ARTIFACT_DIR / "anomaly_iforest.json").write_text(json.dumps({
        "version": version,
        "feature_order": FEATURE_ORDER,
        "trained_on": "legit-only synthetic",
    }, indent=2))

    print(json.dumps({"version": version, "metrics": metrics, "bias": bias}, indent=2))
    return metrics


if __name__ == "__main__":
    train()
