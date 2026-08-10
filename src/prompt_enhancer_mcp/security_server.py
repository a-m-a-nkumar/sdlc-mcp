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
    Run the COMPLETE security suite. Call this whenever the user asks to check
    security, find vulnerabilities, scan for secrets, or review the code they just
    wrote for security issues — this ONE tool runs everything.

    The returned MANDATORY workflow runs an AUTOMATIC CHAIN in the developer's own
    environment: install/run semgrep (SAST), trivy (dependency CVEs) and gitleaks
    (secrets), plus an AI-as-judge REVIEW (you score security_posture as a soft
    second opinion, deferring to the scanners), printing a descriptive read-out of
    each scanner as it goes (secrets shown as type + location + [redacted] only),
    then AUTO-SUBMITS everything via submit_security_findings + submit_quality_review
    (no permission prompt). The backend parses, redacts secrets on ingest, and
    scores — it never sees a raw secret it is supposed to store.

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


# ─── Tool 3: submit_quality_review (AI-as-judge — called by the chain) ─────────

@mcp.tool()
async def submit_quality_review(
    review_json: str,
    ctx: Context,
    judge_model: str = "",
    scope: str = "",
    commit_sha: str = "",
    project_id: str = None,
) -> str:
    """
    Submit the AI-as-judge rubric JSON. The prepare_security_scan workflow tells YOU
    (the IDE agent) to score security_posture against the project's standards and call
    this tool automatically at the end of the chain — there is no separate prepare step
    to call first.

    review_json MUST be the JSON schema described in the workflow (score_total + a
    per-dimension breakdown; leave dimensions this server doesn't own as null). It is
    stored as a self-reported AI-judge score, kept separate from the scanner findings
    (never averaged with them) — it is a soft second opinion, not a scanner result.

    Args:
        review_json: the rubric JSON, as a string.
        judge_model: YOUR model identity as the judge (e.g. "claude-code",
                     "github-copilot", "cursor") — so the dashboard shows which AI
                     reviewed, not a hardcoded label.
        scope, commit_sha, project_id.
    """
    project_id, api_url, api_key = get_config(project_id)
    if err := validate_config(project_id, api_key):
        return err
    if not (review_json or "").strip():
        return "Error: review_json is empty. Produce the rubric JSON first, then submit it."
    try:
        client = get_client(ctx)
        resp = await client.post(
            f"{api_url}/api/quality/llm-review-internal",
            headers={"X-API-Key": api_key},
            json={"project_id": project_id, "scope": scope, "commit_sha": commit_sha,
                  "review_json": review_json, "judge_model": judge_model},
            timeout=120.0,
        )
    except Exception as e:
        logger.exception("submit_quality_review backend call failed")
        return f"Error calling backend: {type(e).__name__}: {str(e)}"
    if resp.status_code != 200:
        logger.error("Backend returned %d — %s", resp.status_code, resp.text[:500])
        return f"Error: Backend returned {resp.status_code} — {resp.text[:500]}"
    data = resp.json()
    sc = data.get("llm_score")
    sc_txt = f"{sc}/100" if sc is not None else "not scored"
    attributed = data.get("attributed")
    return (
        f"AI security review recorded for project {project_id}.\n"
        f"- security_posture (self-reported): {sc_txt}\n"
        f"- Attributed to project owner: {'yes' if attributed else 'no (unattributed)'}\n\n"
        f"This is a SOFT second opinion (AI grading AI) — the dashboard labels it "
        f"self-reported and keeps it separate from the scanner findings, which are "
        f"the authoritative security signal."
    )


# ─── Entrypoint ────────────────────────────────────────────────────────────────

def main():
    try:
        from .config import fire_install_beacon
        from . import __version__ as _v
        fire_install_beacon("security", _v)
    except Exception:
        pass
    mcp.run()


if __name__ == "__main__":
    main()
