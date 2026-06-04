"""Dashboard integration for KoraFlex AI — mounts both Streamlit apps as
sub-processes and proxies them behind a FastAPI router with Bearer-token auth.

FIXES the "orphaned Streamlit scripts" gap:
- Previously, data_quality_dashboard.py and fraud_analytics_dashboard.py
  were standalone scripts unreachable from the deployed service.
- This module adds a /v1/dashboards/* router that:
    1. Lazily spawns each Streamlit process on first request (8501, 8502).
    2. Reverse-proxies requests via httpx so a single Render service serves
       the dashboards without extra infrastructure.
    3. Gates every request behind the same JWT_SECRET used by the main app,
       so the dashboards are not publicly accessible.

Usage (already wired in main.py patch below):
    app.include_router(dashboard_router, prefix="/v1/dashboards", tags=["dashboards"])
"""
from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.config import settings
from app.core.logging import get_logger

log = get_logger(__name__)
router = APIRouter()
security = HTTPBearer()

# ---------------------------------------------------------------------------
# Process registry  {name: (proc, port)}
# ---------------------------------------------------------------------------

_procs: dict[str, tuple[subprocess.Popen, int]] = {}  # type: ignore[type-arg]

DASHBOARDS_DIR = Path(__file__).resolve().parents[3] / "dashboards"

DASHBOARD_MAP: dict[str, tuple[str, int]] = {
    "data-quality": ("data_quality_dashboard.py", 8501),
    "fraud":        ("fraud_analytics_dashboard.py", 8502),
}


def _spawn(name: str) -> int:
    """Start the Streamlit process if not already running. Returns port."""
    if name in _procs:
        proc, port = _procs[name]
        if proc.poll() is None:   # still alive
            return port

    filename, port = DASHBOARD_MAP[name]
    script = DASHBOARDS_DIR / filename
    if not script.exists():
        raise FileNotFoundError(f"Dashboard script not found: {script}")

    env_extra = {
        "MONGODB_URI": settings.MONGODB_URI,
        "MONGODB_DB": settings.MONGODB_DB,
    }
    import os
    env = {**os.environ, **env_extra}

    proc = subprocess.Popen(
        [
            sys.executable, "-m", "streamlit", "run", str(script),
            "--server.port", str(port),
            "--server.headless", "true",
            "--server.enableCORS", "false",
            "--server.enableXsrfProtection", "false",
        ],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    _procs[name] = (proc, port)
    log.info("dashboard_spawned", name=name, port=port, pid=proc.pid)
    return port


async def _wait_ready(port: int, retries: int = 20, delay: float = 0.5) -> None:
    """Poll until Streamlit is serving on the given port."""
    for _ in range(retries):
        try:
            async with httpx.AsyncClient() as client:
                r = await client.get(f"http://localhost:{port}/healthz", timeout=2.0)
                if r.status_code < 500:
                    return
        except Exception:
            pass
        await asyncio.sleep(delay)
    raise TimeoutError(f"Dashboard on port {port} did not start in time")


# ---------------------------------------------------------------------------
# Auth dependency
# ---------------------------------------------------------------------------

def _verify_token(creds: HTTPAuthorizationCredentials = Depends(security)) -> None:
    """Simple shared-secret check — same secret as the main JWT config."""
    if creds.credentials != settings.JWT_SECRET:
        raise HTTPException(status_code=401, detail="Invalid dashboard token")


# ---------------------------------------------------------------------------
# Reverse-proxy route
# ---------------------------------------------------------------------------

@router.api_route(
    "/{dashboard_name}/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD", "PATCH"],
    dependencies=[Depends(_verify_token)],
)
async def proxy_dashboard(
    dashboard_name: str,
    path: str,
    request: Request,
) -> Response:
    if dashboard_name not in DASHBOARD_MAP:
        raise HTTPException(404, f"Unknown dashboard '{dashboard_name}'. "
                                 f"Available: {list(DASHBOARD_MAP)}")

    # Spawn lazily
    try:
        port = _spawn(dashboard_name)
        await _wait_ready(port)
    except (FileNotFoundError, TimeoutError) as exc:
        raise HTTPException(503, str(exc))

    target_url = f"http://localhost:{port}/{path}"
    params = dict(request.query_params)

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            proxied = await client.request(
                method=request.method,
                url=target_url,
                params=params,
                headers={k: v for k, v in request.headers.items() if k.lower() != "host"},
                content=await request.body(),
            )
        except httpx.RequestError as exc:
            raise HTTPException(502, f"Dashboard unreachable: {exc}")

    return Response(
        content=proxied.content,
        status_code=proxied.status_code,
        headers=dict(proxied.headers),
        media_type=proxied.headers.get("content-type"),
    )


# ---------------------------------------------------------------------------
# Status endpoint (no auth required — used by health checks)
# ---------------------------------------------------------------------------

@router.get("/status")
async def dashboard_status() -> dict:
    """Returns the run-state of each dashboard process."""
    result = {}
    for name, (filename, port) in DASHBOARD_MAP.items():
        if name in _procs:
            proc, _ = _procs[name]
            state = "running" if proc.poll() is None else "crashed"
        else:
            state = "not_started"
        result[name] = {"state": state, "port": port, "script": filename}
    return result
