"""Day 10 — Bias audit: ensure fraud model isn't unfairly targeting groups."""
from __future__ import annotations

import numpy as np


def bias_report(y_true: np.ndarray, y_pred: np.ndarray, sensitive: np.ndarray) -> dict:
    """Demographic parity + equal opportunity (TPR) gaps across groups."""
    groups = np.unique(sensitive)
    per_group = {}
    for g in groups:
        mask = sensitive == g
        yt, yp = y_true[mask], y_pred[mask]
        n = int(mask.sum())
        pos = int(yt.sum())
        flagged = int(yp.sum())
        tp = int(((yp == 1) & (yt == 1)).sum())
        fp = int(((yp == 1) & (yt == 0)).sum())
        per_group[str(g)] = {
            "n": n,
            "actual_fraud_rate": pos / n if n else 0,
            "flagged_rate": flagged / n if n else 0,
            "tpr": tp / pos if pos else 0,
            "fpr": fp / max(1, (yt == 0).sum()),
        }

    flagged_rates = [v["flagged_rate"] for v in per_group.values()]
    tprs = [v["tpr"] for v in per_group.values()]
    return {
        "per_group": per_group,
        "demographic_parity_gap": max(flagged_rates) - min(flagged_rates),
        "equal_opportunity_gap": max(tprs) - min(tprs),
        "passes": (max(flagged_rates) - min(flagged_rates)) < 0.10
                  and (max(tprs) - min(tprs)) < 0.10,
    }
