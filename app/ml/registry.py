"""Day 6/7 — Model registry: loads and serves trained artifacts.

Keeps inference dependency-light: pure joblib + numpy. Training is in
`app/ml/training/`.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from app.core.logging import get_logger

log = get_logger(__name__)

ARTIFACT_DIR = Path(__file__).parent / "artifacts"


class FraudModel:
    """Thin wrapper around an XGBoost/sklearn classifier."""

    def __init__(self, pipeline, feature_order: list[str], version: str):
        self.pipeline = pipeline
        self.feature_order = feature_order
        self.version = version

    def _vec(self, features: dict) -> np.ndarray:
        return np.array([[float(features.get(f, 0.0)) for f in self.feature_order]])

    def predict_proba(self, features: dict) -> float:
        x = self._vec(features)
        proba = self.pipeline.predict_proba(x)[0, 1]
        return float(proba)

    def top_contributing_features(self, features: dict, k: int = 3) -> list[tuple[str, float]]:
        """Cheap "reason codes" using feature_importances_ * normalized value."""
        clf = self.pipeline.named_steps.get("clf") if hasattr(self.pipeline, "named_steps") else self.pipeline
        if not hasattr(clf, "feature_importances_"):
            return []
        importances = clf.feature_importances_
        x = self._vec(features)[0]
        contribs = []
        for name, imp, val in zip(self.feature_order, importances, x):
            contribs.append((name, float(imp * val)))
        contribs.sort(key=lambda t: abs(t[1]), reverse=True)
        return contribs[:k]


class AnomalyModel:
    """IsolationForest wrapper → 0..1 score (higher = more anomalous)."""

    def __init__(self, model, feature_order: list[str], version: str):
        self.model = model
        self.feature_order = feature_order
        self.version = version

    def score(self, features: dict) -> float:
        x = np.array([[float(features.get(f, 0.0)) for f in self.feature_order]])
        # decision_function: higher = more normal. Convert to 0..1 anomaly score.
        df = self.model.decision_function(x)[0]
        return float(max(0.0, min(1.0, 0.5 - df)))


class ModelRegistry:
    _models: dict[str, Any] = {}

    @classmethod
    def load_all(cls) -> None:
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        cls._models = {}

        fraud_path = ARTIFACT_DIR / "fraud_xgb.joblib"
        fraud_meta = ARTIFACT_DIR / "fraud_xgb.json"
        if fraud_path.exists() and fraud_meta.exists():
            meta = json.loads(fraud_meta.read_text())
            pipeline = joblib.load(fraud_path)
            cls._models["fraud_xgb"] = FraudModel(pipeline, meta["feature_order"], meta["version"])
            log.info("model_loaded", name="fraud_xgb", version=meta["version"])

        anom_path = ARTIFACT_DIR / "anomaly_iforest.joblib"
        anom_meta = ARTIFACT_DIR / "anomaly_iforest.json"
        if anom_path.exists() and anom_meta.exists():
            meta = json.loads(anom_meta.read_text())
            model = joblib.load(anom_path)
            cls._models["anomaly_iforest"] = AnomalyModel(model, meta["feature_order"], meta["version"])
            log.info("model_loaded", name="anomaly_iforest", version=meta["version"])

        if not cls._models:
            log.warning("no_models_found", hint="Run python -m app.ml.training.train_fraud_model")

    @classmethod
    def get(cls, name: str):
        return cls._models.get(name)
