"""Slack / email alerting for data-quality + fraud spikes."""
from __future__ import annotations

import httpx

from app.core.config import settings
from app.core.logging import get_logger

log = get_logger(__name__)


async def send_alert(title: str, message: str, severity: str = "warning") -> None:
    payload = {
        "text": f"[KoraFlex AI][{severity.upper()}] {title}\n{message}"
    }
    log.warning("alert", title=title, severity=severity, message=message)
    if not settings.SLACK_WEBHOOK_URL:
        return
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            await client.post(settings.SLACK_WEBHOOK_URL, json=payload)
    except Exception as exc:  # pragma: no cover
        log.error("slack_alert_failed", error=str(exc))
