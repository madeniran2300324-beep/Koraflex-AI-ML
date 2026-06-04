# Fraud Model Card — KoraFlex (MVP)

## Intended use
Score every checkout transaction with a 0–100 risk score so the BNPL engine can:
- auto-allow low risk,
- send medium risk to manual review,
- auto-block high risk.

## Architecture
- **Rule layer** (`app/services/fraud/rules.py`): fast deterministic checks
  (velocity, new-user high-value, invalid amount). Sub-millisecond.
- **Supervised model**: XGBoost classifier, ~13 features (amount, velocity counts &
  sums over 5m/1h/24h per user/device/IP, account age, new-user flags).
  Trained with `scale_pos_weight` to handle ~6% positive class.
- **Anomaly model**: IsolationForest trained on legitimate-only data.
- **Fusion**: `final = max(rule_score, round(100 * (0.7*ml + 0.3*anomaly)))`.

## Performance (synthetic baseline)
Run `python -m app.ml.training.train_fraud_model` to print metrics
(ROC-AUC, PR-AUC, confusion matrix, classification report) and the bias audit.

## Inputs (feature order)
`amount, user_age_days, account_age_minutes, is_new_user, high_value_new_user,
 tx_count_5m, tx_amount_5m, tx_count_1h, tx_amount_1h, tx_count_24h, tx_amount_24h,
 device_tx_count_1h, ip_tx_count_1h`

## Latency target
< 500 ms end-to-end (`/v1/transactions/score`). Velocity features come from Redis
(O(1) Redis ops); model inference is a single vectorized predict_proba call.

## Reason codes (explainability)
We surface the top-3 features by `feature_importance × normalized_value`. For full
SHAP attributions, swap in `shap.TreeExplainer` (dep already pinned) inside
`FraudModel.top_contributing_features`.

## Feedback loop
`POST /v1/feedback` writes confirmed labels to `fraud_feedback`. The retraining
job calls `export_training_set()` to join scores with labels.

## Bias audit (Day 10)
`app/ml/training/bias_audit.py` reports per-group flagged rate + TPR/FPR and
fails the build if demographic-parity or equal-opportunity gap > 10%.
