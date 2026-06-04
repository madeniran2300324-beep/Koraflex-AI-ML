"""Day 1 — Statistical anomaly detection for incoming bank/Mono transaction data.

FIXES:
- Previous file had the pure math (robust z-score + IQR) but was never wired into
  a usable service: no field-level scanning, no persistence, no alerting.
- This version adds:
    * scan_dataframe()  — iterates every numeric column and flags anomalous rows
    * scan_mono_batch() — entry point called from the API after each Mono webhook
    * persists results to MongoDB (anomaly_events collection)
    * fires a Slack/email alert when the anomaly rate exceeds threshold
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from app.core.alerts import send_alert
from app.core.config import settings
from app.core.logging import get_logger
from app.db.mongo import db

log = get_logger(__name__)

# Raise an alert if more than this fraction of records are anomalous
ANOMALY_RATE_ALERT_THRESHOLD = 0.10


# ---------------------------------------------------------------------------
# Core statistical helpers (unchanged from original, kept co-located)
# ---------------------------------------------------------------------------

def robust_zscore(series: pd.Series) -> pd.Series:
    """Median Absolute Deviation z-score — robust to outliers."""
    med = series.median()
    mad = (series - med).abs().median()
    if mad == 0:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return 0.6745 * (series - med) / mad


def detect_anomalies(series: pd.Series, z_thresh: float = 3.5) -> pd.DataFrame:
    """Flag values that exceed the MAD z-score threshold OR fall outside IQR fences."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return pd.DataFrame(columns=["value", "robust_z", "below_iqr", "above_iqr"])
    z = robust_zscore(s)
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    mask = (z.abs() > z_thresh) | (s < lo) | (s > hi)
    return pd.DataFrame({
        "value": s[mask],
        "robust_z": z[mask].round(3),
        "below_iqr": s[mask] < lo,
        "above_iqr": s[mask] > hi,
    })


# ---------------------------------------------------------------------------
# NEW: field-level scan across a whole DataFrame
# ---------------------------------------------------------------------------

def scan_dataframe(
    df: pd.DataFrame,
    z_thresh: float = 3.5,
    numeric_cols: list[str] | None = None,
) -> dict[str, Any]:
    """
    Run anomaly detection over every numeric column (or a specified subset).

    Returns
    -------
    {
        "total_records": int,
        "anomaly_rate": float,          # fraction of rows with ≥1 anomalous field
        "anomalous_record_count": int,
        "by_field": {
            "<col>": {
                "n_anomalies": int,
                "anomaly_rate": float,
                "rows": [{"index": ..., "value": ..., "robust_z": ...}, ...]
            }
        }
    }
    """
    cols = numeric_cols or list(df.select_dtypes(include="number").columns)
    total = len(df)
    anomalous_indices: set = set()
    by_field: dict[str, Any] = {}

    for col in cols:
        flagged = detect_anomalies(df[col], z_thresh=z_thresh)
        if flagged.empty:
            by_field[col] = {"n_anomalies": 0, "anomaly_rate": 0.0, "rows": []}
            continue

        anomalous_indices.update(flagged.index.tolist())
        rows = [
            {"index": int(idx), "value": float(row["value"]), "robust_z": float(row["robust_z"])}
            for idx, row in flagged.iterrows()
        ]
        by_field[col] = {
            "n_anomalies": len(flagged),
            "anomaly_rate": round(len(flagged) / max(total, 1), 4),
            "rows": rows[:100],  # cap payload size
        }

    anomalous_record_count = len(anomalous_indices)
    return {
        "total_records": total,
        "anomalous_record_count": anomalous_record_count,
        "anomaly_rate": round(anomalous_record_count / max(total, 1), 4),
        "by_field": by_field,
    }


# ---------------------------------------------------------------------------
# NEW: Mono batch entry point — persist + alert
# ---------------------------------------------------------------------------

async def scan_mono_batch(
    records: list[dict],
    source: str = "mono",
    z_thresh: float = 3.5,
) -> dict[str, Any]:
    """
    Called after each Mono webhook batch lands.

    1. Runs scan_dataframe() over the batch.
    2. Persists the summary to MongoDB (anomaly_events collection).
    3. Fires an alert if the anomaly rate exceeds ANOMALY_RATE_ALERT_THRESHOLD.
    4. Returns the summary dict.
    """
    if not records:
        return {"total_records": 0, "anomalous_record_count": 0, "anomaly_rate": 0.0, "by_field": {}}

    df = pd.DataFrame(records)
    summary = scan_dataframe(df, z_thresh=z_thresh)
    summary["source"] = source
    summary["scanned_at"] = datetime.now(timezone.utc).isoformat()

    # Persist
    await db().anomaly_events.insert_one({
        **summary,
        "scanned_at": datetime.now(timezone.utc),
    })

    log.info(
        "mono_anomaly_scan",
        source=source,
        total=summary["total_records"],
        anomalous=summary["anomalous_record_count"],
        rate=summary["anomaly_rate"],
    )

    # Alert if rate is alarming
    if summary["anomaly_rate"] > ANOMALY_RATE_ALERT_THRESHOLD:
        top_fields = sorted(
            summary["by_field"].items(),
            key=lambda kv: kv[1]["n_anomalies"],
            reverse=True,
        )[:3]
        field_summary = ", ".join(f"{k}({v['n_anomalies']})" for k, v in top_fields)
        await send_alert(
            title="High anomaly rate in Mono bank data",
            message=(
                f"Source: {source} | "
                f"Rate: {summary['anomaly_rate']:.1%} "
                f"({summary['anomalous_record_count']}/{summary['total_records']} records). "
                f"Top anomalous fields: {field_summary}."
            ),
            severity="warning",
        )

    return summary
