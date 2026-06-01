import os
import time
import logging
import httpx
from mcp.server.fastmcp import Context

logger = logging.getLogger("mcp.config")

# Harness config cache: project_id -> (cfg_dict_or_error_str, fetched_at_epoch).
# 5-minute TTL — short enough to auto-recover if a user links Harness mid-session,
# long enough to keep tool invocations cheap.
_HARNESS_CACHE_TTL = 300
_harness_cfg_cache: dict[str, tuple] = {}


def get_config(project_id: str = None) -> tuple[str, str, str]:
    """Return (project_id, api_url, api_key) from args/env."""
    pid = project_id or os.environ.get("PROJECT_ID")
    url = os.environ.get("API_URL", "http://localhost:8000")
    key = os.environ.get("API_KEY")
    return pid, url, key


def validate_config(project_id: str, api_key: str) -> str | None:
    """Return error string if config is invalid, None if OK."""
    if not project_id or not api_key:
        return "Error: PROJECT_ID and API_KEY must be set (via env var or argument)."
    return None


def get_tech_stack() -> tuple[str, str]:
    """Return (frontend_requirements, backend_requirements) from env."""
    frontend = os.environ.get("FRONTEND_REQUIREMENTS", "")
    backend = os.environ.get("BACKEND_REQUIREMENTS", "")
    return frontend, backend


def get_client(ctx: Context) -> httpx.AsyncClient:
    """Get the shared httpx client from lifespan context."""
    return ctx.request_context.lifespan_context["client"]


async def fetch_harness_config(
    client: httpx.AsyncClient,
    api_url: str,
    api_key: str,
    project_id: str,
) -> dict | str:
    """
    Fetch Harness config from the backend for the project's owner.
    Cached per project_id for 5 minutes so tool calls stay cheap.

    Returns the config dict on success, or a human-readable error string on failure.
    Backend resolves harness_pat + account_id + org_id + project_id from the project
    owner's stored credentials; base_url comes from the backend's HARNESS_BASE_URL.
    """
    now = time.time()
    cached = _harness_cfg_cache.get(project_id)
    if cached and (now - cached[1]) < _HARNESS_CACHE_TTL:
        return cached[0]

    try:
        resp = await client.post(
            f"{api_url}/api/orchestration/harness-config-internal",
            headers={"X-API-Key": api_key},
            json={"project_id": project_id},
            timeout=15.0,
        )
    except Exception as e:
        return f"Error fetching Harness config from backend: {type(e).__name__}: {e}"

    if resp.status_code != 200:
        # Surface the backend's detail message verbatim so users see
        # "Project owner has not linked a Harness account..." etc.
        try:
            detail = resp.json().get("detail") or resp.text[:300]
        except Exception:
            detail = resp.text[:300]
        return f"Error: Backend returned {resp.status_code} — {detail}"

    cfg = resp.json()
    _harness_cfg_cache[project_id] = (cfg, now)
    return cfg


def validate_harness_config(cfg) -> str | None:
    """
    Return error string if the Harness config didn't load, None if OK.

    Accepts either the dict returned by fetch_harness_config, or the error
    string fetch_harness_config returned on failure (which gets passed through).
    """
    if isinstance(cfg, str):
        return cfg
    if not cfg or not cfg.get("api_key"):
        return "Error: Harness config is missing or incomplete (no PAT)."
    return None
