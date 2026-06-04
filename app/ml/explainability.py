"""Model explainability — SHAP-based reason codes.

FIXES the "approximate reason codes" gap:
- Previously, top_contributing_features() used feature_importances_ × value,
  which is a global importance proxy, not the actual contribution for a given
  prediction (SHAP is correct; the old method is not).
- shap was already in requirements.txt but unused at inference time.

This module provides:
    explain_prediction(model_wrapper, features) → list of (feature, shap_value)

It is called from scoring.py when a model is available and the decision is
"review" or "block" — i.e., only when a human will actually read the reasons.
For "allow" decisions we stay on the cheap path to preserve latency budget.

Thread-safety: TreeExplainer is stateless for prediction; safe to share.
"""
from __future__ import annotations

import numpy as np

from app.core.logging import get_logger

log = get_logger(__name__)


def explain_prediction(
    model_wrapper,           # app.ml.registry.FraudModel
    features: dict,
    k: int = 5,
) -> list[tuple[str, float]]:
    """
    Compute per-feature SHAP contributions for a single prediction.

    Returns
    -------
    List of (feature_name, shap_value) sorted by |shap_value| descending,
    truncated to the top-k contributors.

    Falls back to the cheap feature_importances_ method if SHAP is unavailable
    or if the underlying classifier type is unsupported.
    """
    try:
        import shap  # imported lazily — not needed at module load time
    except ImportError:
        log.warning("shap_not_installed", hint="pip install shap")
        return model_wrapper.top_contributing_features(features, k=k)

    clf = None
    if hasattr(model_wrapper, "pipeline") and hasattr(model_wrapper.pipeline, "named_steps"):
        clf = model_wrapper.pipeline.named_steps.get("clf")

    if clf is None or not hasattr(clf, "get_booster"):
        # Not an XGBoost model — fall back gracefully
        return model_wrapper.top_contributing_features(features, k=k)

    x = model_wrapper._vec(features)  # shape (1, n_features)

    try:
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(x)  # shape (1, n_features)
        contributions = shap_values[0]           # shape (n_features,)
    except Exception as exc:
        log.warning("shap_failed", error=str(exc))
        return model_wrapper.top_contributing_features(features, k=k)

    pairs = list(zip(model_wrapper.feature_order, contributions))
    pairs.sort(key=lambda t: abs(t[1]), reverse=True)
    return [(name, float(val)) for name, val in pairs[:k]]


def format_reason_codes(
    shap_pairs: list[tuple[str, float]],
    threshold: float = 0.01,
) -> list[str]:
    """
    Convert SHAP (feature, value) pairs into human-readable reason strings.

    Only includes features whose |shap| ≥ threshold to avoid noise.

    Examples
    --------
    "tx_count_5m increased risk by +0.18 (4 transactions in 5 min)"
    "device_unique_users_24h increased risk by +0.14 (device shared by 7 users today)"
    """
    FRIENDLY: dict[str, str] = {
        "tx_count_5m":              "rapid transaction burst (5 min window)",
        "tx_count_1h":              "high transaction count (1 hr window)",
        "tx_amount_1h":             "large transaction volume (1 hr window)",
        "device_unique_users_24h":  "device shared across multiple accounts today",
        "device_is_known":          "unrecognised device",
        "device_account_age_days":  "very new device with short account history",
        "ip_unique_users_1h":       "IP shared by many users this hour",
        "ip_tx_count_1h":           "high IP-level transaction rate",
        "high_value_new_user":      "large purchase by a brand-new account",
        "is_new_user":              "account is less than 7 days old",
        "amount":                   "transaction amount",
        "user_age_days":            "account age",
    }

    reasons = []
    for feature, shap_val in shap_pairs:
        if abs(shap_val) < threshold:
            continue
        direction = "increased" if shap_val > 0 else "decreased"
        label = FRIENDLY.get(feature, feature.replace("_", " "))
        reasons.append(
            f"{label} {direction} risk by {shap_val:+.2f}"
        )
    return reasons
