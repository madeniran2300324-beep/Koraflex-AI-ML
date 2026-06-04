import pandas as pd

from app.services.data_quality.validation import (
    TRANSACTION_RULES,
    USER_RULES,
    validate_dataframe,
    validate_record,
)


def test_valid_transaction_passes():
    rec = {
        "transaction_id": "tx1",
        "user_id": "u1",
        "amount": 1500.0,
        "currency": "NGN",
        "merchant_id": "m1",
        "timestamp": "2026-01-01T00:00:00",
    }
    report = validate_record(rec, TRANSACTION_RULES)
    assert report.failed == 0


def test_missing_required_fails():
    rec = {"transaction_id": "tx1"}
    report = validate_record(rec, TRANSACTION_RULES)
    assert report.failed == 1
    assert any(i.rule == "required" for i in report.issues)


def test_user_email_format():
    rec = {"user_id": "u1", "email": "not-an-email", "phone": "08012345678", "full_name": "Ada"}
    report = validate_record(rec, USER_RULES)
    assert report.failed == 1
    assert any(i.field == "email" for i in report.issues)


def test_batch_validation():
    df = pd.DataFrame([
        {"transaction_id": "tx1", "user_id": "u1", "amount": 100, "currency": "NGN",
         "merchant_id": "m", "timestamp": "2026-01-01"},
        {"transaction_id": "tx2", "user_id": "u2", "amount": -5, "currency": "XXX",
         "merchant_id": "m", "timestamp": "bad"},
    ])
    report = validate_dataframe(df, TRANSACTION_RULES)
    assert report.passed == 1
    assert report.failed == 1
