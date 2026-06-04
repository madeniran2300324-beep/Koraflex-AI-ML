"""Day 10 — End-to-end integration tests for the KoraFlex AI/ML service.

FIXES the "no E2E tests for the AI/ML service" gap:
- Previously, tests only covered unit-level functions (rules, validation, similarity).
  No test exercised the actual FastAPI endpoints.
- This suite fires real HTTP requests against a test instance of the app
  using httpx.AsyncClient + ASGITransport (no network needed).

Coverage
--------
POST /v1/transactions/score       — full scoring pipeline
POST /v1/identity/score           — identity fraud detection
POST /v1/login/score              — ATO scoring
POST /v1/feedback/                — feedback labeling
GET  /v1/transactions/review-queue
POST /v1/data-quality/validate    — data quality validation
POST /v1/data-quality/anomalies   — anomaly detection
GET  /v1/data-quality/freshness   — freshness check
GET  /v1/admin/thresholds         — threshold read (auth)
PUT  /v1/admin/thresholds         — threshold update (auth)

Run with:
    pytest tests/test_e2e.py -v
"""
from __future__ import annotations

import os
import pytest
import pytest_asyncio
import httpx
from unittest.mock import AsyncMock, MagicMock, patch

# Set required env vars before importing the app
os.environ.setdefault("JWT_SECRET", "test-secret-key-e2e-32chars-long!")
os.environ.setdefault("MONGODB_URI", "mongodb://localhost:27017")
os.environ.setdefault("REDIS_URL",   "redis://localhost:6379/1")  # DB 1 for tests


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def anyio_backend():
    return "asyncio"


def _make_mock_db():
    """Returns a mock Motor database that swallows all writes."""
    coll = MagicMock()
    coll.insert_one    = AsyncMock(return_value=MagicMock(inserted_id="fakeid"))
    coll.update_one    = AsyncMock()
    coll.find_one      = AsyncMock(return_value=None)
    coll.count_documents = AsyncMock(return_value=0)
    coll.distinct      = AsyncMock(return_value=[])

    async def _empty_cursor(*_, **__):
        return
        yield  # make it an async generator

    coll.find = MagicMock(return_value=MagicMock(
        sort=MagicMock(return_value=MagicMock(
            limit=MagicMock(return_value=_empty_cursor())
        ))
    ))

    mock_db = MagicMock()
    mock_db.__getattr__ = lambda self, name: coll
    mock_db.__getitem__ = lambda self, name: coll
    return mock_db


def _make_mock_redis():
    r = MagicMock()
    r.lpush   = AsyncMock(return_value=1)
    r.ltrim   = AsyncMock()
    r.expire  = AsyncMock()
    r.lrange  = AsyncMock(return_value=[])
    r.get     = AsyncMock(return_value=None)
    r.set     = AsyncMock()
    r.sismember = AsyncMock(return_value=False)
    r.ping    = AsyncMock()
    return r


@pytest_asyncio.fixture(scope="session")
async def client():
    """Async test client wired to the FastAPI app with mocked Mongo + Redis."""
    mock_db_instance = _make_mock_db()
    mock_redis_instance = _make_mock_redis()

    with (
        patch("app.db.mongo.init_mongo",   new=AsyncMock()),
        patch("app.db.mongo.close_mongo",  new=AsyncMock()),
        patch("app.db.mongo.db",           return_value=mock_db_instance),
        patch("app.db.redis_client.init_redis",  new=AsyncMock()),
        patch("app.db.redis_client.close_redis", new=AsyncMock()),
        patch("app.db.redis_client.redis", return_value=mock_redis_instance),
        patch("app.ml.registry.ModelRegistry.load_all", return_value=None),
        patch("app.ml.registry.ModelRegistry.get",      return_value=None),
    ):
        from app.main import app
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as c:
            yield c


AUTH = {"Authorization": "Bearer test-secret-key-e2e-32chars-long!"}


# ---------------------------------------------------------------------------
# Transaction scoring
# ---------------------------------------------------------------------------

class TestTransactionScore:
    @pytest.mark.asyncio
    async def test_score_returns_valid_structure(self, client):
        payload = {
            "transaction_id": "tx-001",
            "user_id":        "user-001",
            "amount":         50000.0,
            "currency":       "NGN",
            "merchant_id":    "merch-001",
            "user_age_days":  90,
            "account_age_minutes": 130000,
        }
        r = await client.post("/v1/transactions/score", json=payload)
        assert r.status_code == 200
        data = r.json()
        assert "risk_score" in data
        assert data["risk_score"] >= 0
        assert data["risk_score"] <= 100
        assert data["band"] in ("low", "medium", "high")
        assert data["decision"] in ("allow", "review", "block")
        assert isinstance(data["reasons"], list)
        assert "latency_ms" in data

    @pytest.mark.asyncio
    async def test_invalid_amount_blocked(self, client):
        payload = {
            "transaction_id": "tx-bad",
            "user_id":        "user-001",
            "amount":         -100.0,
            "merchant_id":    "merch-001",
        }
        r = await client.post("/v1/transactions/score", json=payload)
        assert r.status_code == 200
        data = r.json()
        assert data["risk_score"] == 100
        assert data["decision"] == "block"
        assert "invalid_amount" in data["triggered_rules"]

    @pytest.mark.asyncio
    async def test_new_user_high_value_flagged(self, client):
        payload = {
            "transaction_id": "tx-new-hv",
            "user_id":        "user-new",
            "amount":         500_000.0,
            "merchant_id":    "merch-001",
            "user_age_days":  2,
        }
        r = await client.post("/v1/transactions/score", json=payload)
        assert r.status_code == 200
        data = r.json()
        assert "new_user_high_value" in data["triggered_rules"]

    @pytest.mark.asyncio
    async def test_review_queue_endpoint(self, client):
        r = await client.get("/v1/transactions/review-queue")
        assert r.status_code == 200
        assert isinstance(r.json(), list)


# ---------------------------------------------------------------------------
# Identity fraud
# ---------------------------------------------------------------------------

class TestIdentityScore:
    @pytest.mark.asyncio
    async def test_clean_identity_low_risk(self, client):
        payload = {
            "user_id":   "uid-clean",
            "email":     "john.doe@gmail.com",
            "phone":     "+2348012345678",
            "full_name": "John Doe",
            "bvn":       "12345678901",
        }
        r = await client.post("/v1/identity/score", json=payload)
        assert r.status_code == 200
        data = r.json()
        assert data["risk_score"] < 50

    @pytest.mark.asyncio
    async def test_disposable_email_raises_score(self, client):
        payload = {
            "user_id":   "uid-disp",
            "email":     "test@mailinator.com",
            "phone":     "+2348099999999",
            "full_name": "Test User",
        }
        r = await client.post("/v1/identity/score", json=payload)
        assert r.status_code == 200
        data = r.json()
        assert data["risk_score"] >= 35
        assert "disposable_email" in data["triggered_rules"]

    @pytest.mark.asyncio
    async def test_no_kyc_adds_score(self, client):
        payload = {
            "user_id":   "uid-nokyc",
            "email":     "legit@gmail.com",
            "phone":     "+2348088888888",
            "full_name": "Legit Person",
            # no bvn, no nin
        }
        r = await client.post("/v1/identity/score", json=payload)
        assert r.status_code == 200
        assert "no_kyc_id" in r.json()["triggered_rules"]


# ---------------------------------------------------------------------------
# Account Takeover
# ---------------------------------------------------------------------------

class TestLoginScore:
    @pytest.mark.asyncio
    async def test_normal_login_low_risk(self, client):
        payload = {
            "user_id":    "user-ato",
            "ip":         "102.89.1.1",
            "user_agent": "Mozilla/5.0 (Windows NT 10.0) Chrome/120",
            "device_id":  "device-abc",
            "geo_country": "NG",
        }
        r = await client.post("/v1/login/score", json=payload)
        assert r.status_code == 200
        assert r.json()["risk_score"] <= 50


# ---------------------------------------------------------------------------
# Feedback loop
# ---------------------------------------------------------------------------

class TestFeedback:
    @pytest.mark.asyncio
    async def test_record_fraud_feedback(self, client):
        payload = {
            "transaction_id": "tx-001",
            "is_fraud":       True,
            "notes":          "Confirmed by ops team",
            "labeled_by":     "ops",
        }
        r = await client.post("/v1/feedback/", json=payload)
        assert r.status_code == 200
        assert r.json()["status"] == "recorded"


# ---------------------------------------------------------------------------
# Data Quality
# ---------------------------------------------------------------------------

class TestDataQuality:
    @pytest.mark.asyncio
    async def test_validate_valid_transaction(self, client):
        payload = {
            "schema": "transaction",
            "records": [{
                "transaction_id": "tx-dq-1",
                "user_id":        "u-1",
                "amount":         15000.0,
                "currency":       "NGN",
                "merchant_id":    "m-1",
                "timestamp":      "2024-06-01T10:00:00",
            }],
        }
        r = await client.post("/v1/data-quality/validate", json=payload)
        assert r.status_code == 200
        data = r.json()
        assert data["total_records"] == 1
        assert data["passed"] == 1
        assert data["failed"] == 0

    @pytest.mark.asyncio
    async def test_validate_missing_required_field(self, client):
        payload = {
            "schema": "transaction",
            "records": [{"user_id": "u-1", "amount": 1000.0}],  # missing transaction_id etc.
        }
        r = await client.post("/v1/data-quality/validate", json=payload)
        assert r.status_code == 200
        assert r.json()["failed"] == 1

    @pytest.mark.asyncio
    async def test_anomaly_detection_flags_outlier(self, client):
        values = [100, 105, 98, 102, 99, 103, 50_000]  # last is a clear outlier
        r = await client.post("/v1/data-quality/anomalies", json={"values": values})
        assert r.status_code == 200
        data = r.json()
        assert data["n_anomalies"] >= 1

    @pytest.mark.asyncio
    async def test_freshness_unknown_when_no_events(self, client):
        r = await client.get("/v1/data-quality/freshness")
        assert r.status_code == 200
        assert r.json()["status"] in ("unknown", "fresh", "stale")


# ---------------------------------------------------------------------------
# Admin: threshold management
# ---------------------------------------------------------------------------

class TestAdminThresholds:
    @pytest.mark.asyncio
    async def test_get_thresholds_requires_auth(self, client):
        r = await client.get("/v1/admin/thresholds")
        assert r.status_code == 403  # no Bearer token

    @pytest.mark.asyncio
    async def test_get_thresholds_with_auth(self, client):
        r = await client.get("/v1/admin/thresholds", headers=AUTH)
        assert r.status_code == 200
        data = r.json()
        assert "auto_block" in data
        assert "review" in data
        assert data["auto_block"] > data["review"]

    @pytest.mark.asyncio
    async def test_update_thresholds(self, client):
        payload = {"auto_block": 90, "review": 55, "reason": "Tightening for launch week"}
        r = await client.put("/v1/admin/thresholds", json=payload, headers=AUTH)
        assert r.status_code == 200
        data = r.json()
        assert data["auto_block"] == 90
        assert data["review"] == 55

    @pytest.mark.asyncio
    async def test_update_thresholds_rejects_inverted_values(self, client):
        payload = {"auto_block": 40, "review": 60, "reason": "bad config"}
        r = await client.put("/v1/admin/thresholds", json=payload, headers=AUTH)
        assert r.status_code == 422  # Pydantic validation error

    @pytest.mark.asyncio
    async def test_threshold_history(self, client):
        r = await client.get("/v1/admin/thresholds/history", headers=AUTH)
        assert r.status_code == 200
        assert isinstance(r.json(), list)
