"""Core settings — REPLACES app/core/config.py

FIXES TWO security gaps:
1. JWT_SECRET had a "change-me" default — anyone who deployed without setting
   the env var would have a publicly known secret.
   FIX: JWT_SECRET is now a required field with no default; the app will refuse
   to start (ValidationError) if it is not set.

2. CORS was allow_origins=["*"] — any origin could make credentialed requests.
   FIX: Added ALLOWED_ORIGINS (comma-separated, env-driven).
   In main.py, allow_origins=settings.ALLOWED_ORIGINS is used instead of ["*"].

Also adds LATENCY_BUDGET_MS used by scoring.py's timeout enforcement.
"""
from __future__ import annotations

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    
    APP_ENV:  str = "development"
    APP_PORT: int = 8001
    LOG_LEVEL: str = "INFO"

    MONGODB_URI: str = "mongodb://localhost:27017"
    MONGODB_DB:  str = "koraflex"

    REDIS_URL: str = "redis://localhost:6379/0"

    MONO_FRESHNESS_MAX_AGE_MIN: int = 60

    FRAUD_AUTO_BLOCK_THRESHOLD: int = 70
    FRAUD_REVIEW_THRESHOLD:     int = 35

    # Latency SLA for the fraud scoring pipeline (ms).
    # Falls back to rules-only if the ML pipeline exceeds this budget.
    LATENCY_BUDGET_MS: int = 480

    SLACK_WEBHOOK_URL: str = ""
    ALERT_EMAIL: str = "ops@koraflex.com"

    # SECURITY FIX 1: No default — the app will crash at startup if unset.
    JWT_SECRET: str = Field(..., description="Required: set in environment / .env")
    JWT_ALG:    str = "HS256"

    # SECURITY FIX 2: Locked CORS origins.
    # Set to a comma-separated list in the environment, e.g.:
    #   ALLOWED_ORIGINS=https://app.koraflex.com,https://admin.koraflex.com
    # Defaults to localhost only for local development.
    ALLOWED_ORIGINS_STR: str = Field(
        default="http://localhost:3000,http://localhost:5173",
        alias="ALLOWED_ORIGINS",
    )

    @field_validator("JWT_SECRET")
    @classmethod
    def _jwt_secret_must_be_set(cls, v: str) -> str:
        if not v or v.lower() in ("change-me", "secret", "changeme", ""):
            raise ValueError(
                "JWT_SECRET must be set to a strong random value in the environment. "
                "Generate one with: python -c \"import secrets; print(secrets.token_hex(32))\""
            )
        return v

    @property
    def ALLOWED_ORIGINS(self) -> list[str]:
        return [o.strip() for o in self.ALLOWED_ORIGINS_STR.split(",") if o.strip()]

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        populate_by_name=True,
    )


settings = Settings()
