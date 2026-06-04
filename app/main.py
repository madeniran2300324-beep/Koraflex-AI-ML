"""KoraFlex AI/ML FastAPI entry point.

CHANGES vs original:
- CORSMiddleware now uses settings.ALLOWED_ORIGINS instead of ["*"]
- dashboard_router added under /v1/dashboards
- LATENCY_BUDGET_MS surfaced in startup log for observability
"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Histogram, make_asgi_app

from app.api import (
    data_quality_router,
    feedback_router,
    fraud_router,
    health_router,
    identity_router,
    login_router,
    network_router,
)
from app.api.dashboards import router as dashboard_router
from app.core.config import settings
from app.core.logging import configure_logging, get_logger
from app.db.mongo import close_mongo, init_mongo
from app.db.redis_client import close_redis, init_redis
from app.ml.registry import ModelRegistry

log = get_logger(__name__)

REQUESTS = Counter("koraflex_ai_requests_total", "Requests", ["path", "method", "status"])
LATENCY  = Histogram("koraflex_ai_latency_seconds", "Latency", ["path"])


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging(settings.LOG_LEVEL)
    log.info(
        "startup",
        env=settings.APP_ENV,
        allowed_origins=settings.ALLOWED_ORIGINS,
        latency_budget_ms=settings.LATENCY_BUDGET_MS,
    )
    await init_mongo()
    await init_redis()
    ModelRegistry.load_all()
    yield
    await close_mongo()
    await close_redis()
    log.info("shutdown")


app = FastAPI(
    title="KoraFlex AI/ML Service",
    description="Fraud detection (0-100) + Data Quality engine for the KoraFlex BNPL MVP.",
    version="1.0.0",
    lifespan=lifespan,
)

# SECURITY FIX: locked CORS — no longer allow_origins=["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
)


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - start
    REQUESTS.labels(request.url.path, request.method, response.status_code).inc()
    LATENCY.labels(request.url.path).observe(elapsed)
    response.headers["X-Response-Time-Ms"] = f"{elapsed * 1000:.2f}"
    return response


app.mount("/metrics", make_asgi_app())

app.include_router(health_router)
app.include_router(identity_router,     prefix="/v1/identity",      tags=["identity-fraud"])
app.include_router(login_router,        prefix="/v1/login",          tags=["account-takeover"])
app.include_router(fraud_router,        prefix="/v1/transactions",   tags=["transaction-fraud"])
app.include_router(feedback_router,     prefix="/v1/feedback",       tags=["feedback"])
app.include_router(network_router,      prefix="/v1/fraud",          tags=["network-analysis"])
app.include_router(data_quality_router, prefix="/v1/data-quality",   tags=["data-quality"])
# FIX: dashboards now served inside the app, not as orphaned Streamlit processes
app.include_router(dashboard_router,    prefix="/v1/dashboards",     tags=["dashboards"])
app.include_router(admin_router, prefix="/v1/admin", tags=["admin"])
