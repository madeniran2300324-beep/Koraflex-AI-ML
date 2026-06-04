"""Synthetic dataset generator for the baseline fraud model.

FIXES the "synthetic distribution doesn't match real Mono data" gap:
- Added the four new device-fingerprint features introduced in velocity.py
  so FEATURE_ORDER stays in sync between training and inference.
- Added a NOTE block documenting this as a known limitation and the migration
  path to real labeled data.

NOTE — Synthetic data caveat
----------------------------
This generator produces plausible distributions but cannot replicate the exact
statistical properties of real Mono transaction data. The model trained here
should be treated as a *warm-start baseline* that is replaced as soon as
ops has labeled ≥500 confirmed fraud / legit pairs via the feedback API
(/v1/feedback). Use export_training_set() in services/fraud/feedback.py to
export real labeled data and re-run train_fraud_model.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_ORDER = [
    # --- original features ---
    "amount",
    "user_age_days",
    "account_age_minutes",
    "is_new_user",
    "high_value_new_user",
    "tx_count_5m",
    "tx_amount_5m",
    "tx_count_1h",
    "tx_amount_1h",
    "tx_count_24h",
    "tx_amount_24h",
    "device_tx_count_1h",
    "ip_tx_count_1h",
    # --- NEW: device fingerprint features (must match velocity.py) ---
    "device_unique_users_24h",
    "device_is_known",
    "device_account_age_days",
    "ip_unique_users_1h",
]


def make_dataset(n: int = 20_000, fraud_ratio: float = 0.06, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_fraud = int(n * fraud_ratio)
    n_legit = n - n_fraud

    # ---- Legitimate ----
    legit = pd.DataFrame({
        "amount":               rng.lognormal(mean=9.5, sigma=0.9, size=n_legit).clip(500, 1_500_000),
        "user_age_days":        rng.integers(8, 720, size=n_legit),
        "account_age_minutes":  rng.integers(1440, 1_000_000, size=n_legit),
        "tx_count_5m":          rng.poisson(0.2, size=n_legit),
        "tx_count_1h":          rng.poisson(0.6, size=n_legit),
        "tx_count_24h":         rng.poisson(2.5, size=n_legit),
        "device_tx_count_1h":   rng.poisson(0.8, size=n_legit),
        "ip_tx_count_1h":       rng.poisson(1.0, size=n_legit),
        # NEW device features — legitimate users have low device sharing
        "device_unique_users_24h": rng.integers(1, 2, size=n_legit),   # almost always 1
        "device_is_known":         rng.choice([0, 1], size=n_legit, p=[0.15, 0.85]),
        "device_account_age_days": rng.integers(30, 720, size=n_legit),
        "ip_unique_users_1h":      rng.integers(1, 3, size=n_legit),
    })
    legit["tx_amount_5m"]  = legit["amount"] * legit["tx_count_5m"]  * rng.uniform(0.5, 1.2, n_legit)
    legit["tx_amount_1h"]  = legit["amount"] * legit["tx_count_1h"]  * rng.uniform(0.5, 1.2, n_legit)
    legit["tx_amount_24h"] = legit["amount"] * legit["tx_count_24h"] * rng.uniform(0.5, 1.2, n_legit)
    legit["is_fraud"] = 0

    # ---- Fraud (heavier tails, velocity, new accounts, big tickets, device sharing) ----
    fraud = pd.DataFrame({
        "amount":               rng.lognormal(mean=11.5, sigma=0.6, size=n_fraud).clip(20_000, 5_000_000),
        "user_age_days":        rng.integers(0, 14, size=n_fraud),
        "account_age_minutes":  rng.integers(5, 3 * 1440, size=n_fraud),
        "tx_count_5m":          rng.poisson(3.5, size=n_fraud),
        "tx_count_1h":          rng.poisson(6.0, size=n_fraud),
        "tx_count_24h":         rng.poisson(10.0, size=n_fraud),
        "device_tx_count_1h":   rng.poisson(5.0, size=n_fraud),
        "ip_tx_count_1h":       rng.poisson(6.5, size=n_fraud),
        # NEW device features — fraudsters share devices and IPs
        "device_unique_users_24h": rng.integers(3, 15, size=n_fraud),
        "device_is_known":         rng.choice([0, 1], size=n_fraud, p=[0.75, 0.25]),
        "device_account_age_days": rng.integers(0, 10, size=n_fraud),
        "ip_unique_users_1h":      rng.integers(4, 20, size=n_fraud),
    })
    fraud["tx_amount_5m"]  = fraud["amount"] * fraud["tx_count_5m"]  * rng.uniform(0.8, 1.5, n_fraud)
    fraud["tx_amount_1h"]  = fraud["amount"] * fraud["tx_count_1h"]  * rng.uniform(0.8, 1.5, n_fraud)
    fraud["tx_amount_24h"] = fraud["amount"] * fraud["tx_count_24h"] * rng.uniform(0.8, 1.5, n_fraud)
    fraud["is_fraud"] = 1

    df = (
        pd.concat([legit, fraud], ignore_index=True)
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )
    df["is_new_user"]        = (df["user_age_days"] <= 7).astype(int)
    df["high_value_new_user"] = (
        (df["is_new_user"] == 1) & (df["amount"] > 200_000)
    ).astype(int)

    # Demographic column for bias audit — NOT a model feature
    df["demographic_group"] = rng.choice(["A", "B", "C", "D"], size=len(df), p=[0.4, 0.3, 0.2, 0.1])

    # Ensure FEATURE_ORDER columns all exist
    for col in FEATURE_ORDER:
        if col not in df.columns:
            raise ValueError(f"make_dataset is missing column '{col}' — update FEATURE_ORDER or the generator")

    return df
