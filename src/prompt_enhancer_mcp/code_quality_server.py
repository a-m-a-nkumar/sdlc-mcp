import logging
import warnings
import httpx
from contextlib import asynccontextmanager
from mcp.server.fastmcp import FastMCP, Context
from .config import get_config, validate_config, get_client

warnings.filterwarnings("ignore", message="Unverified HTTPS request")

logger = logging.getLogger("mcp.code-quality")


# ─── Lifespan (shared HTTP client) ────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app):
    """Shared httpx client for the lifetime of the server."""
    async with httpx.AsyncClient(timeout=120.0, verify=False) as client:
        logger.info("Code-quality server: HTTP client created")
        yield {"client": client}
    logger.info("Code-quality server: HTTP client closed")


mcp = FastMCP("code-quality", lifespan=lifespan)


# ─── Tool 1: prepare_quality_scan ─────────────────────────────────────────────

@mcp.tool()
async def prepare_quality_scan(
    ctx: Context,
    scan_mode: str = "",
    scope: str = "",
    project_id: str = None,
) -> str:
    """
    STEP 1: Start the code-quality scan (complexity + lint/quality). Call this
    when the user asks to check code quality, complexity, maintainability, or to
    review the code they just wrote against standards.

    The returned MANDATORY workflow first RESOLVES SCOPE with the developer
    (whole repo vs the files they just changed — discovered via `git diff` and
    confirmed by the developer, never hand-listed), then runs lizard / jscpd /
    radon / the language linter, REPORTS the findings to the developer in chat,
    and finally calls submit_quality_metrics. Everything runs in the developer's
    environment — the backend only parses and scores what you submit.

    Args:
        scan_mode: Optional override — "changed" or "whole_repo". If empty, the
                   served workflow asks the developer in-chat which they want.
        scope: Optional explicit scope. If empty, resolved in STEP 1a.
        project_id: Optional project ID. Defaults to PROJECT_ID env var.
    """
    project_id, api_url, api_key = get_config(project_id)
    if err := validate_config(project_id, api_key):
        return err

    logger.info("Preparing quality scan (scan_mode=%s, scope=%s, project=%s)",
                scan_mode or "(ask)", scope or "(resolve)", project_id)
    try:
        client = get_client(ctx)
        resp = await client.post(
            f"{api_url}/api/quality/quality-prompt-internal",
            headers={"X-API-Key": api_key},
            json={"project_id": project_id, "scan_mode": scan_mode, "scope": scope},
            timeout=120.0,
        )
    except Exception as e:
        logger.exception("prepare_quality_scan backend call failed")
        return f"Error calling backend: {type(e).__name__}: {str(e)}"

    if resp.status_code != 200:
        logger.error("Backend returned %d — %s", resp.status_code, resp.text[:500])
        return f"Error: Backend returned {resp.status_code} — {resp.text[:500]}"

    return resp.json().get("prompt", "Error: backend returned no prompt.")


# ─── Tool 2: submit_quality_metrics ───────────────────────────────────────────

@mcp.tool()
async def submit_quality_metrics(
    ctx: Context,
    lizard_csv: str = "",
    radon_json: str = "",
    jscpd_json: str = "",
    lint_json: str = "",
    lint_tool: str = "",
    scan_granularity: str = "file_scope",
    scope: str = "",
    commit_sha: str = "",
    language: str = "",
    files_analyzed: list = None,
    unavailable_tools: list = None,
    project_id: str = None,
) -> str:
    """
    STEP 2: Submit the RAW static-analysis reports after running the tools.

    Call AFTER prepare_quality_scan and after actually running the tools. Pass RAW
    tool output verbatim — the backend re-parses it into separate complexity and
    quality scores. Leave a field empty AND add the tool to unavailable_tools if
    it could not run — never fabricate output.

    Args:
        lizard_csv: raw `lizard --csv` output.
        radon_json: raw `radon mi -j` output (Python).
        jscpd_json: raw `jscpd --reporters json` output.
        lint_json:  raw linter JSON (eslint -f json / ruff --output-format=json).
        lint_tool:  which linter produced lint_json.
        scan_granularity: "file_scope" (changed files) or "whole_project".
                          Report honestly — do not label a whole-repo scan as
                          if it measured only the changed files.
        scope, commit_sha, language, files_analyzed, unavailable_tools, project_id.
    """
    project_id, api_url, api_key = get_config(project_id)
    if err := validate_config(project_id, api_key):
        return err

    if not any([(lizard_csv or "").strip(), (radon_json or "").strip(),
                (jscpd_json or "").strip(), (lint_json or "").strip()]):
        return ("Error: no reports provided. Run lizard and/or the language linter "
                "first and pass their raw output — do not fabricate metrics.")

    logger.info("Submitting quality metrics (granularity=%s) for project %s",
                scan_granularity, project_id)
    try:
        client = get_client(ctx)
        resp = await client.post(
            f"{api_url}/api/quality/metrics-internal",
            headers={"X-API-Key": api_key},
            json={
                "project_id": project_id, "scope": scope, "commit_sha": commit_sha,
                "language": language, "scan_granularity": scan_granularity,
                "lizard_csv": lizard_csv, "radon_json": radon_json,
                "jscpd_json": jscpd_json, "lint_json": lint_json, "lint_tool": lint_tool,
                "files_analyzed": files_analyzed or [], "unavailable_tools": unavailable_tools or [],
            },
            timeout=120.0,
        )
    except Exception as e:
        logger.exception("submit_quality_metrics backend call failed")
        return f"Error calling backend: {type(e).__name__}: {str(e)}"

    if resp.status_code != 200:
        logger.error("Backend returned %d — %s", resp.status_code, resp.text[:500])
        return f"Error: Backend returned {resp.status_code} — {resp.text[:500]}"

    data = resp.json()
    cs = data.get("complexity_score")
    qs = data.get("quality_score")
    cs_txt = f"{cs}/100" if cs is not None else "not measured"
    qs_txt = f"{qs}/100" if qs is not None else "not measured"
    attributed = data.get("attributed")
    return (
        f"Code-quality scores recorded for project {project_id} "
        f"(granularity: {scan_granularity}).\n"
        f"- Complexity: {cs_txt}\n"
        f"- Quality:    {qs_txt}\n"
        f"- Attributed to project owner: {'yes' if attributed else 'no (unattributed)'}\n\n"
        f"These are on the project's Pair Programming KPI dashboard. NOTE: server-side "
        f"parsing is a format check, not proof the tools ran; complexity and quality "
        f"are heuristic 0-100 scores (scorer v1), kept as separate cards (never averaged)."
    )


# ─── Tool 3: prepare_quality_review (Copilot-as-judge) ────────────────────────

@mcp.tool()
async def prepare_quality_review(
    ctx: Context,
    scope: str = "",
    project_id: str = None,
) -> str:
    """
    STEP 1 (review): Start a Copilot-as-judge code review of the code just written.
    Call this when the user asks for a qualitative review, a code critique, or "how
    good is this code" against the project's standards.

    Returns a rubric prompt (five dimensions) with THIS project's own standards
    injected by the backend. YOU do the judging (free — no backend LLM): produce the
    JSON, show the developer a short summary, then call submit_quality_review with
    the JSON. This is a SELF-REPORTED review (AI grading AI) and is labelled as such.

    Args:
        scope: What was reviewed. project_id: defaults to PROJECT_ID env var.
    """
    project_id, api_url, api_key = get_config(project_id)
    if err := validate_config(project_id, api_key):
        return err
    try:
        client = get_client(ctx)
        resp = await client.post(
            f"{api_url}/api/quality/llm-review-prompt-internal",
            headers={"X-API-Key": api_key},
            json={"project_id": project_id, "scope": scope},
            timeout=120.0,
        )
    except Exception as e:
        logger.exception("prepare_quality_review backend call failed")
        return f"Error calling backend: {type(e).__name__}: {str(e)}"
    if resp.status_code != 200:
        logger.error("Backend returned %d — %s", resp.status_code, resp.text[:500])
        return f"Error: Backend returned {resp.status_code} — {resp.text[:500]}"
    return resp.json().get("prompt", "Error: backend returned no prompt.")


# ─── Tool 4: submit_quality_review ────────────────────────────────────────────

@mcp.tool()
async def submit_quality_review(
    review_json: str,
    ctx: Context,
    scope: str = "",
    commit_sha: str = "",
    project_id: str = None,
) -> str:
    """
    STEP 2 (review): Submit the rubric JSON you produced from prepare_quality_review.

    review_json MUST be the exact JSON schema from the rubric prompt (score_total +
    a per-dimension breakdown). It is stored as a self-reported Copilot-judge score,
    kept separate from the tool-based scores (never averaged with them).

    Args:
        review_json: the rubric JSON, as a string.
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
                  "review_json": review_json},
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
        f"Copilot review recorded for project {project_id}.\n"
        f"- Overall (self-reported): {sc_txt}\n"
        f"- Attributed to project owner: {'yes' if attributed else 'no (unattributed)'}\n\n"
        f"This is the SOFTEST signal (AI grading AI, uncontrolled model version) — the "
        f"dashboard labels it self-reported and keeps it separate from the tool-based scores."
    )


# ─── Entrypoint ────────────────────────────────────────────────────────────────

def main():
    mcp.run()


if __name__ == "__main__":
    main()
