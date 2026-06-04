from .health import router as health_router
from .identity import router as identity_router
from .login import router as login_router
from .fraud import router as fraud_router
from .feedback import router as feedback_router
from .network import router as network_router
from .data_quality import router as data_quality_router
from app.api.dashboards import router as dashboard_router
from app.api.admin import router as admin_router

__all__ = [
    "health_router",
    "identity_router",
    "login_router",
    "fraud_router",
    "feedback_router",
    "network_router",
    "data_quality_router",
]
