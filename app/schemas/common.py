from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, EmailStr, Field


RiskBand = Literal["low", "medium", "high"]
Decision = Literal["allow", "review", "block"]


class IdentityPayload(BaseModel):
    user_id: str
    email: EmailStr
    phone: str
    full_name: str
    bvn: Optional[str] = None
    nin: Optional[str] = None
    dob: Optional[str] = None
    ip: Optional[str] = None
    device_id: Optional[str] = None


class LoginPayload(BaseModel):
    user_id: str
    ip: str
    user_agent: str
    device_id: str
    geo_country: Optional[str] = None
    geo_city: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class TransactionPayload(BaseModel):
    transaction_id: str
    user_id: str
    amount: float
    currency: str = "NGN"
    merchant_id: str
    product_category: Optional[str] = None
    payment_method_id: Optional[str] = None
    ip: Optional[str] = None
    device_id: Optional[str] = None
    user_age_days: int = 0
    account_age_minutes: int = 0
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ScoreResponse(BaseModel):
    risk_score: int  # 0-100
    band: RiskBand
    decision: Decision
    reasons: list[str] = []
    triggered_rules: list[str] = []
    model_version: str
    latency_ms: float


class FeedbackPayload(BaseModel):
    transaction_id: str
    is_fraud: bool
    notes: Optional[str] = None
    labeled_by: str = "ops"
