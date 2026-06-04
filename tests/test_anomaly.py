import pandas as pd

from app.services.data_quality.anomaly import detect_anomalies


def test_detects_obvious_outlier():
    s = pd.Series([10, 11, 12, 9, 10, 11, 12, 10, 9, 1000])
    out = detect_anomalies(s)
    assert 1000 in out["value"].values
