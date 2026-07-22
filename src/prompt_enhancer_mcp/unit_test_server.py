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
    STEP 1: Start the unit-test + coverage workflow. Call this when the user asks
    to generate unit tests, add tests, measure test coverage, or check how well
    the code they just wrote is tested.

    Returns a MANDATORY workflow prompt the IDE AI must follow: detect the
    language, author meaningful tests, run them WITH coverage, REPORT the result
    to the developer in chat, then call submit_test_results with the RAW coverage
    report. Everything runs in the developer's own environment — the backend only
    parses and scores what you submit.

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


# ─── Tool 3: prepare_mutation_test ────────────────────────────────────────────

@mcp.tool()
async def prepare_mutation_test(
    ctx: Context,
    scope: str = "",
    project_id: str = None,
) -> str:
    """
    STEP 1 (mutation): Start mutation testing — the ONE fabrication-resistant signal
    (the tests must actually KILL injected mutants, which can't be faked by typing a
    number). Call this when the user asks how strong/thorough their tests are, or for
    mutation testing / a mutation score.

    Returns a workflow: install/run the mutation tool (mutmut / StrykerJS / ...) WITH
    the existing test suite (slow), report the score + surviving mutants to the
    developer, then call submit_mutation_results.

    Args:
        scope: module(s) to stress-test. project_id: defaults to PROJECT_ID env var.
    """
    project_id, api_url, api_key = get_config(project_id)
    if err := validate_config(project_id, api_key):
        return err
    try:
        client = get_client(ctx)
        resp = await client.post(
            f"{api_url}/api/quality/mutation-prompt-internal",
            headers={"X-API-Key": api_key},
            json={"project_id": project_id, "scope": scope},
            timeout=120.0,
        )
    except Exception as e:
        logger.exception("prepare_mutation_test backend call failed")
        return f"Error calling backend: {type(e).__name__}: {str(e)}"
    if resp.status_code != 200:
        return f"Error: Backend returned {resp.status_code} — {resp.text[:500]}"
    return resp.json().get("prompt", "Error: backend returned no prompt.")


# ─── Tool 4: submit_mutation_results ──────────────────────────────────────────

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
    STEP 2 (mutation): Submit the mutation result after the run actually completed.

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


# ─── Entrypoint ────────────────────────────────────────────────────────────────

def main():
    mcp.run()


if __name__ == "__main__":
    main()
