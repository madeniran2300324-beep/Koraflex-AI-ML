"""Day 2 — Automated data profiling: stats, distributions, correlations."""
from __future__ import annotations

import pandas as pd


def profile_dataframe(df: pd.DataFrame) -> dict:
    numeric = df.select_dtypes(include="number")
    categorical = df.select_dtypes(include=["object", "category", "bool"])

    profile = {
        "shape": {"rows": len(df), "columns": df.shape[1]},
        "columns": {},
        "missing": df.isna().mean().round(4).to_dict(),
        "correlations": numeric.corr().round(3).to_dict() if not numeric.empty else {},
    }

    for col in df.columns:
        col_data = df[col]
        info: dict = {
            "dtype": str(col_data.dtype),
            "n_unique": int(col_data.nunique(dropna=True)),
            "missing_pct": float(col_data.isna().mean()),
        }
        if col in numeric.columns:
            info.update({
                "mean": float(col_data.mean()),
                "std": float(col_data.std()),
                "min": float(col_data.min()),
                "p25": float(col_data.quantile(0.25)),
                "p50": float(col_data.quantile(0.50)),
                "p75": float(col_data.quantile(0.75)),
                "max": float(col_data.max()),
            })
        elif col in categorical.columns:
            top = col_data.value_counts(dropna=True).head(5).to_dict()
            info["top_values"] = {str(k): int(v) for k, v in top.items()}
        profile["columns"][col] = info

    return profile
