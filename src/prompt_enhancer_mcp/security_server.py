import logging
import warnings
import httpx
from contextlib import asynccontextmanager
from mcp.server.fastmcp import FastMCP, Context
from .config import get_config, validate_config, get_client

warnings.filterwarnings("ignore", message="Unverified HTTPS request")

logger = logging.getLogger("mcp.security")


# ─── Lifespan (shared HTTP client) ────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app):
    """Shared httpx client for the lifetime of the server."""
    async with httpx.AsyncClient(timeout=120.0, verify=False) as client:
        logger.info("Security server: HTTP client created")
        yield {"client": client}
    logger.info("Security server: HTTP client closed")


mcp = FastMCP("security", lifespan=lifespan)


# ─── Tool 1: prepare_security_scan ────────────────────────────────────────────

@mcp.tool()
async def prepare_security_scan(
    ctx: Context,
    scope: str = "",
    project_id: str = None,
) -> str:
    """
    STEP 1: Start the security scan (SAST + dependency CVEs + secrets). Call this
    when the user asks to check security, find vulnerabilities, scan for secrets,
    or review the code they just wrote for security issues.

    Returns a MANDATORY workflow: install/run semgrep (SAST), trivy (dependency
    CVEs) and gitleaks (secrets) in the developer's environment, REPORT each
    finding to the developer in chat (secrets shown as type + location + [redacted]
    only), then call submit_security_findings with the RAW reports. The backend
    parses, redacts secrets on ingest, and scores — it never sees a raw secret it
    is supposed to store.

    Args:
        scope: What to scan. Dependency + secret scans are whole-project by nature.
        project_id: Optional project ID. Defaults to PROJECT_ID env var.
    """
    project_id, api_url, api_key = get_config(project_id)
    if err := validate_config(project_id, api_key):
        return err

    logger.info("Preparing security scan (scope=%s, project=%s)", scope or "(whole)", project_id)
    try:
        client = get_client(ctx)
        resp = await client.post(
            f"{api_url}/api/quality/security-prompt-internal",
            headers={"X-API-Key": api_key},
            json={"project_id": project_id, "scope": scope},
            timeout=120.0,
        )
    except Exception as e:
        logger.exception("prepare_security_scan backend call failed")
        return f"Error calling backend: {type(e).__name__}: {str(e)}"

    if resp.status_code != 200:
        logger.error("Backend returned %d — %s", resp.status_code, resp.text[:500])
        return f"Error: Backend returned {resp.status_code} — {resp.text[:500]}"

    return resp.json().get("prompt", "Error: backend returned no prompt.")


# ─── Tool 2: submit_security_findings ─────────────────────────────────────────

@mcp.tool()
async def submit_security_findings(
    ctx: Context,
    sast_sarif: str = "",
    deps_report: str = "",
    deps_format: str = "trivy_json",
    secrets_report: str = "",
    secrets_tool: str = "gitleaks",
    scan_granularity: str = "whole_project",
    scope: str = "",
    commit_sha: str = "",
    language: str = "",
    files_analyzed: list = None,
    unavailable_tools: list = None,
    project_id: str = None,
) -> str:
    """
    STEP 2: Submit the RAW security reports after running the scanners.

    Call AFTER prepare_security_scan and after actually running the tools. Pass RAW
    report contents verbatim. For SECRETS: the backend redacts on ingest to type +
    location only — but you must STILL never print a secret value to the developer.
    Leave a report empty AND add the tool to unavailable_tools if it could not run —
    an unavailable scanner is NOT a clean scan; never fabricate output.

    Args:
        sast_sarif:     RAW semgrep (+bandit/gosec/brakeman) SARIF.
        deps_report:    RAW trivy report.
        deps_format:    "trivy_json" (preferred) or "sarif".
        secrets_report: RAW gitleaks SARIF, or trufflehog JSON (--json).
        secrets_tool:   "gitleaks" (default) or "trufflehog".
        scan_granularity: usually "whole_project" (deps + secrets are whole-project).
        scope, commit_sha, language, files_analyzed, unavailable_tools, project_id.
    """
    project_id, api_url, api_key = get_config(project_id)
    if err := validate_config(project_id, api_key):
        return err

    if not any([(sast_sarif or "").strip(), (deps_report or "").strip(),
                (secrets_report or "").strip()]):
        return ("Error: no reports provided. Run semgrep / trivy / gitleaks first "
                "and pass their raw output — do not fabricate findings.")

    logger.info("Submitting security findings (secrets_tool=%s) for project %s",
                secrets_tool, project_id)
    try:
        client = get_client(ctx)
        resp = await client.post(
            f"{api_url}/api/quality/security-internal",
            headers={"X-API-Key": api_key},
            json={
                "project_id": project_id, "scope": scope, "commit_sha": commit_sha,
                "language": language, "scan_granularity": scan_granularity,
                "sast_sarif": sast_sarif, "deps_report": deps_report, "deps_format": deps_format,
                "secrets_report": secrets_report, "secrets_tool": secrets_tool,
                "files_analyzed": files_analyzed or [], "unavailable_tools": unavailable_tools or [],
            },
            timeout=120.0,
        )
    except Exception as e:
        logger.exception("submit_security_findings backend call failed")
        return f"Error calling backend: {type(e).__name__}: {str(e)}"

    if resp.status_code != 200:
        logger.error("Backend returned %d — %s", resp.status_code, resp.text[:500])
        return f"Error: Backend returned {resp.status_code} — {resp.text[:500]}"

    data = resp.json()
    gates = data.get("gates", {})
    attributed = data.get("attributed")

    def g(k):
        v = gates.get(k)
        return v if v else "not measured"

    return (
        f"Security findings recorded for project {project_id}.\n"
        f"- SAST gate:    {g('security_sast')}\n"
        f"- Deps gate:    {g('security_deps')}\n"
        f"- Secrets gate: {g('security_secrets')}\n"
        f"- Attributed to project owner: {'yes' if attributed else 'no (unattributed)'}\n\n"
        f"On the Pair Programming KPI dashboard, security is FINDING-LED: the card "
        f"leads with critical/high counts and a pass/fail gate, not a percentage. "
        f"Secrets were redacted on ingest (type + location only). NOTE: SCA has no "
        f"reachability and SAST/secrets are potential issues needing human triage — "
        f"and a well-formed report is a format check, not proof the scan ran."
    )


# ─── Entrypoint ────────────────────────────────────────────────────────────────

def main():
    mcp.run()


if __name__ == "__main__":
    main()
