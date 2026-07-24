import logging
import warnings
import httpx
from contextlib import asynccontextmanager
from mcp.server.fastmcp import FastMCP, Context
from .config import get_config, validate_config, get_client

warnings.filterwarnings("ignore", message="Unverified HTTPS request")

logger = logging.getLogger("mcp.unit-test")


# ─── Lifespan (shared HTTP client) ────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app):
    """Shared httpx client for the lifetime of the server."""
    async with httpx.AsyncClient(timeout=120.0, verify=False) as client:
        logger.info("Unit-test server: HTTP client created")
        yield {"client": client}
    logger.info("Unit-test server: HTTP client closed")


mcp = FastMCP("unit-test", lifespan=lifespan)


# ─── Tool 1: prepare_unit_tests ───────────────────────────────────────────────

@mcp.tool()
async def prepare_unit_tests(
    ctx: Context,
    scope: str = "",
    project_id: str = None,
) -> str:
    """
    Run the COMPLETE unit-test suite. Call this whenever the user asks to generate
    unit tests, add tests, measure coverage, run mutation testing, or check how well
    the code they just wrote is tested — this ONE tool runs everything; there is no
    separate mutation tool to call.

    The returned MANDATORY workflow runs an AUTOMATIC CHAIN in the developer's own
    environment: detect the language, author meaningful tests, run them WITH coverage,
    run MUTATION testing (the fabrication-resistant signal), and do an AI-as-judge
    REVIEW (you score testability + correctness against the project's standards),
    printing a descriptive read-out of each tool as it goes, then AUTO-SUBMITS
    everything via submit_test_results + submit_mutation_results + submit_quality_review
    (no permission prompt). The backend only parses and scores what you submit.

    Args:
        scope: What to test/measure. If empty, measure the files just
               authored/changed. Pass a subpath to target a module.
        project_id: Optional project ID. Defaults to PROJECT_ID env var.
    """
    project_id, api_url, api_key = get_config(project_id)
    if err := validate_config(project_id, api_key):
        return err

    logger.info("Preparing unit-test workflow (scope=%s, project=%s)", scope or "(auto)", project_id)
    try:
        client = get_client(ctx)
        resp = await client.post(
            f"{api_url}/api/quality/unit-test-prompt-internal",
            headers={"X-API-Key": api_key},
            json={"project_id": project_id, "scope": scope},
            timeout=120.0,
        )
    except Exception as e:
        logger.exception("prepare_unit_tests backend call failed")
        return f"Error calling backend: {type(e).__name__}: {str(e)}"

    if resp.status_code != 200:
        logger.error("Backend returned %d — %s", resp.status_code, resp.text[:500])
        return f"Error: Backend returned {resp.status_code} — {resp.text[:500]}"

    return resp.json().get("prompt", "Error: backend returned no prompt.")


# ─── Tool 2: submit_test_results ──────────────────────────────────────────────

@mcp.tool()
async def submit_test_results(
    coverage_report: str,
    report_format: str,
    ctx: Context,
    tests_passed: int = 0,
    tests_failed: int = 0,
    scope: str = "",
    commit_sha: str = "",
    framework: str = "",
    files_analyzed: list = None,
    unavailable_tools: list = None,
    project_id: str = None,
) -> str:
    """
    STEP 2: Submit the RAW coverage report after actually running the tests.

    Only call this AFTER prepare_unit_tests AND after you have really run the test
    suite with coverage. Pass the RAW report file contents verbatim — the backend
    re-parses them; do NOT summarize, round, or reconstruct. A metric you could
    not measure must be omitted / listed in unavailable_tools — never invent one.

    Args:
        coverage_report: RAW coverage report file contents.
        report_format:   lcov | cobertura_xml | jest_json_summary.
        tests_passed / tests_failed: counts from the run.
        scope:      files/paths measured.
        commit_sha: output of `git rev-parse HEAD`.
        framework:  detected test framework (pytest, jest, ...).
        files_analyzed:    list of files measured.
        unavailable_tools: tools that could not be installed/run.
        project_id: Optional; defaults to PROJECT_ID env var.
    """
    project_id, api_url, api_key = get_config(project_id)
    if err := validate_config(project_id, api_key):
        return err

    if not (coverage_report or "").strip():
        return ("Error: coverage_report is empty. Run the tests WITH coverage first "
                "and pass the raw report — do not fabricate a coverage number.")

    logger.info("Submitting coverage (%s, %d chars) for project %s",
                report_format, len(coverage_report), project_id)
    try:
        client = get_client(ctx)
        resp = await client.post(
            f"{api_url}/api/quality/test-report-internal",
            headers={"X-API-Key": api_key},
            json={
                "project_id": project_id, "scope": scope, "commit_sha": commit_sha,
                "coverage_report": coverage_report, "report_format": report_format,
                "tests_passed": tests_passed, "tests_failed": tests_failed,
                "framework": framework, "files_analyzed": files_analyzed or [],
                "unavailable_tools": unavailable_tools or [],
            },
            timeout=120.0,
        )
    except Exception as e:
        logger.exception("submit_test_results backend call failed")
        return f"Error calling backend: {type(e).__name__}: {str(e)}"

    if resp.status_code != 200:
        logger.error("Backend returned %d — %s", resp.status_code, resp.text[:500])
        return f"Error: Backend returned {resp.status_code} — {resp.text[:500]}"

    data = resp.json()
    pct = data.get("coverage_pct")
    pct_txt = f"{pct}%" if pct is not None else "not measured"
    attributed = data.get("attributed")
    return (
        f"Coverage recorded for project {project_id}.\n"
        f"- Line coverage: {pct_txt}\n"
        f"- Tests: {tests_passed} passed, {tests_failed} failed\n"
        f"- Attributed to project owner: {'yes' if attributed else 'no (unattributed)'}\n\n"
        f"This is now on the project's Pair Programming KPI dashboard. NOTE: the "
        f"backend parses the report format — it does not prove the tests ran. The "
        f"authoritative signal remains your CI coverage gate."
    )


# ─── Tool 3: submit_mutation_results (called by the chain) ────────────────────

@mcp.tool()
async def submit_mutation_results(
    ctx: Context,
    mutation_report: str = "",
    report_format: str = "stryker_json",
    killed: int = 0,
    survived: int = 0,
    total: int = 0,
    tool: str = "",
    scope: str = "",
    commit_sha: str = "",
    language: str = "",
    scan_granularity: str = "whole_project",
    files_analyzed: list = None,
    unavailable_tools: list = None,
    project_id: str = None,
) -> str:
    """
    Submit the mutation result. The prepare_unit_tests workflow runs mutation testing
    as part of its chain and calls this tool automatically — there is no separate
    prepare step to call first.

    Prefer the StrykerJS JSON (report_format="stryker_json", mutation_report=<raw json>);
    otherwise pass killed, survived and total counts from `mutmut results`. A run that
    could not complete is "unavailable" — never fabricate a score.

    Args:
        mutation_report: raw StrykerJS mutation-report.json (if used).
        report_format: "stryker_json".
        killed / survived / total: counts (e.g. from mutmut).
        tool, scope, commit_sha, language, scan_granularity, files_analyzed,
        unavailable_tools, project_id.
    """
    project_id, api_url, api_key = get_config(project_id)
    if err := validate_config(project_id, api_key):
        return err
    if not (mutation_report or "").strip() and (int(killed or 0) + int(survived or 0) + int(total or 0)) == 0:
        return ("Error: no mutation result. Run mutation testing first and pass the raw "
                "report or the killed/survived/total counts — do not fabricate a score.")
    try:
        client = get_client(ctx)
        resp = await client.post(
            f"{api_url}/api/quality/mutation-report-internal",
            headers={"X-API-Key": api_key},
            json={
                "project_id": project_id, "scope": scope, "commit_sha": commit_sha,
                "language": language, "scan_granularity": scan_granularity,
                "mutation_report": mutation_report, "report_format": report_format,
                "killed": killed, "survived": survived, "total": total, "tool": tool,
                "files_analyzed": files_analyzed or [], "unavailable_tools": unavailable_tools or [],
            },
            timeout=120.0,
        )
    except Exception as e:
        logger.exception("submit_mutation_results backend call failed")
        return f"Error calling backend: {type(e).__name__}: {str(e)}"
    if resp.status_code != 200:
        return f"Error: Backend returned {resp.status_code} — {resp.text[:500]}"
    data = resp.json()
    pct = data.get("mutation_pct")
    pct_txt = f"{pct}%" if pct is not None else "not measured"
    return (
        f"Mutation score recorded for project {project_id}.\n"
        f"- Mutation score: {pct_txt}\n"
        f"- Attributed to project owner: {'yes' if data.get('attributed') else 'no (unattributed)'}\n\n"
        f"Mutation testing is the strongest objective signal — a surviving mutant is a "
        f"real gap your tests don't catch. Strengthen those tests and re-run."
    )


# ─── Tool 4: submit_quality_review (AI-as-judge — called by the chain) ─────────

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
    Submit the AI-as-judge rubric JSON. The prepare_unit_tests workflow tells YOU (the
    IDE agent) to score testability + correctness against the project's standards and
    call this tool automatically at the end of the chain — there is no separate prepare
    step to call first.

    review_json MUST be the JSON schema described in the workflow (score_total + a
    per-dimension breakdown; leave dimensions this server doesn't own as null). It is
    stored as a self-reported AI-judge score, kept separate from the tool-based scores
    (never averaged with them).

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
        f"AI review recorded for project {project_id}.\n"
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
