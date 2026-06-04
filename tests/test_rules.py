from app.schemas.common import TransactionPayload
from app.services.fraud.rules import evaluate_rules


def test_high_value_new_user_triggers():
    tx = TransactionPayload(
        transaction_id="t", user_id="u", amount=500_000, merchant_id="m",
        user_age_days=2,
    )
    feats = {"high_value_new_user": 1, "tx_count_5m": 0, "device_tx_count_1h": 0, "ip_tx_count_1h": 0}
    score, reasons, rules = evaluate_rules(tx, feats)
    assert score >= 40
    assert "new_user_high_value" in rules


def test_velocity_triggers():
    tx = TransactionPayload(transaction_id="t", user_id="u", amount=1000, merchant_id="m")
    feats = {"tx_count_5m": 5, "device_tx_count_1h": 7, "ip_tx_count_1h": 9}
    score, reasons, rules = evaluate_rules(tx, feats)
    assert "velocity_5m" in rules and "device_velocity" in rules and "ip_velocity" in rules
