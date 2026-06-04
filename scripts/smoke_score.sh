#!/usr/bin/env bash
set -e
curl -s -X POST http://localhost:8001/v1/transactions/score \
  -H 'Content-Type: application/json' \
  -d '{
    "transaction_id": "tx_demo_1",
    "user_id": "user_demo_1",
    "amount": 350000,
    "currency": "NGN",
    "merchant_id": "merch_demo",
    "user_age_days": 2,
    "account_age_minutes": 90,
    "ip": "102.89.10.1",
    "device_id": "dev_demo"
  }' | jq .
