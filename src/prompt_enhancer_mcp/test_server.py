import logging
import httpx
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
from mcp.server.fastmcp import FastMCP, Context
from .config import get_config, validate_config, get_client

logger = logging.getLogger("mcp.test-workflow")


# ─── Workflow State Management ─────────────────────────────────────────────────

STEP_ORDER = ["not_started", "pages_listed", "prompt_fetched", "submitted"]


@dataclass
class WorkflowState:
    """Tracks one test-generation workflow from discovery to submission."""
    project_id: str = ""
    step: str = "not_started"       # not_started → pages_listed → prompt_fetched → submitted
    # Step 1 results
    pages: list = field(default_factory=list)         # [{id, title}, ...]
    selected_page_id: str = ""
    selected_page_title: str = ""
    # Step 2 results
    session_id: str = ""
    prompt: str = ""
    scenario_count: int = 0
    # Step 3 results
    submitted: bool = False
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


# One workflow per project_id. If a project starts a new workflow, the old one is replaced.
_workflows: dict[str, WorkflowState] = {}

MAX_WORKFLOW_AGE = timedelta(hours=2)


def _get_workflow(project_id: str) -> WorkflowState | None:
    """Get active workflow for a project, or None if expired/missing."""
    state = _workflows.get(project_id)
    if state and (datetime.utcnow() - state.created_at) > MAX_WORKFLOW_AGE:
        logger.info("Workflow for %s expired, clearing", project_id)
        del _workflows[project_id]
        return None
    return state


def _set_workflow(project_id: str, state: WorkflowState) -> None:
    """Store/update workflow state."""
    state.updated_at = datetime.utcnow()
    _workflows[project_id] = state


def _enforce_step(project_id: str, required_step: str) -> str | None:
    """
    Return error string if workflow is not ready for the required step.
    Returns None if the step is allowed to proceed.
    """
    state = _get_workflow(project_id)

    if not state:
        if required_step == "pages_listed":
            return None
        return (
            "Error: No active workflow found. "
            "Start by calling list_test_scenario_pages() first."
        )

    current_idx = STEP_ORDER.index(state.step)
    required_idx = STEP_ORDER.index(required_step)

    if required_idx <= current_idx + 1:
        return None

    skipped_step = STEP_ORDER[current_idx + 1]

    STEP_HINTS = {
        "pages_listed": "Call list_test_scenario_pages() first.",
        "prompt_fetched": "Call get_test_prompt() with a page ID first.",
        "submitted": "Generate Gherkin and get user confirmation first.",
    }

    return (
        f"Error: Cannot skip steps. "
        f"Current step: '{state.step}'. "
        f"You need to complete '{skipped_step}' before '{required_step}'. "
        f"{STEP_HINTS.get(skipped_step, '')}"
    )


def _resolve_page(state: WorkflowState, page_selection: str) -> dict | None:
    """
    Resolve a page from workflow state by ID, index number, or title (partial match).
    Returns the page dict {id, title} or None if not found.
    """
    if not state or not state.pages:
        return None

    selection = page_selection.strip()

    # Pass 1: Match by exact page ID
    for p in state.pages:
        if selection == str(p["id"]):
            return p

    # Pass 2: Match by index (1-based)
    if selection.isdigit():
        idx = int(selection) - 1
        if 0 <= idx < len(state.pages):
            return state.pages[idx]

    # Pass 3: Match by partial title (case-insensitive)
    for p in state.pages:
        if selection.lower() in p["title"].lower():
            return p

    return None


# ─── Lifespan (shared HTTP client) ────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app):
    """Create a shared httpx client that lives for the entire server lifetime."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        logger.info("Test workflow server: HTTP client created")
        yield {"client": client}
    logger.info("Test workflow server: HTTP client closed")


mcp = FastMCP("test-workflow", lifespan=lifespan)


# ─── Workflow Status Tool ──────────────────────────────────────────────────────

@mcp.tool()
async def get_workflow_status(ctx: Context, project_id: str = None) -> str:
    """
    Check the current state of the test generation workflow.
    Call this if you lost track of the session_id, page_id, or current step.
    Returns all stored workflow data so you can resume where you left off.

    Args:
        project_id: Optional project ID. Defaults to PROJECT_ID env var.
    """
    project_id, _, api_key = get_config(project_id)

    if err := validate_config(project_id, api_key):
        return err

    state = _get_workflow(project_id)

    if not state:
        return (
            "No active workflow found for this project.\n"
            "Start a new workflow by calling list_test_scenario_pages()."
        )

    NEXT_STEPS = {
        "not_started": "Call list_test_scenario_pages() to discover test scenario pages.",
        "pages_listed": "Call get_test_prompt(page_selection=\"<title or number>\") with one of the pages above.",
        "prompt_fetched": "Generate Gherkin using the prompt, show it to the user, then call submit_test_cases() if confirmed.",
        "submitted": "Workflow complete. Start a new one with list_test_scenario_pages() if needed.",
    }

    lines = [
        f"=== Workflow Status ===",
        f"Project: {state.project_id}",
        f"Current step: {state.step}",
        f"Started: {state.created_at.isoformat()}",
        f"Last updated: {state.updated_at.isoformat()}",
    ]

    if state.pages:
        lines.append(f"\n--- Discovered Pages ({len(state.pages)}) ---")
        for i, p in enumerate(state.pages):
            marker = " [SELECTED]" if p["id"] == state.selected_page_id else ""
            lines.append(f"  {i + 1}. Page ID: {p['id']}  |  Title: {p['title']}{marker}")

    if state.session_id:
        lines.append(f"\n--- Session ---")
        lines.append(f"Session ID: {state.session_id}")
        lines.append(f"Page: {state.selected_page_title}")
        lines.append(f"Scenarios: {state.scenario_count}")

    if state.submitted:
        lines.append(f"\n--- Submission ---")
        lines.append(f"Status: Submitted successfully")

    lines.append(f"\n--- Next Step ---")
    lines.append(NEXT_STEPS.get(state.step, "Unknown state. Call list_test_scenario_pages() to restart."))

    return "\n".join(lines)


# ─── Test Generation Workflow Tools ────────────────────────────────────────────

@mcp.tool()
async def list_test_scenario_pages(ctx: Context, project_id: str = None, filter: str = "test scenario") -> str:
    """
    STEP 1: List Confluence pages containing test scenarios for your project.
    Returns page IDs and titles. Use a page title/number with get_test_prompt() next.
    This is always the first step — call this to start a new workflow.

    Args:
        project_id: Optional project ID. Defaults to PROJECT_ID env var.
        filter: Optional filter string to match page titles. Defaults to "test scenario".
    """
    project_id, api_url, api_key = get_config(project_id)

    if err := validate_config(project_id, api_key):
        return err

    if err := _enforce_step(project_id, "pages_listed"):
        return err

    logger.info("Listing test scenario pages (Project: %s, Filter: %s)", project_id, filter)

    try:
        client = get_client(ctx)
        resp = await client.post(
            f"{api_url}/api/test/list-pages-internal",
            headers={"X-API-Key": api_key},
            json={
                "project_id": project_id,
                "filter": filter,
            },
        )

        if resp.status_code != 200:
            return f"Error: Backend returned {resp.status_code} — {resp.text}"

        data = resp.json()
        pages = data.get("pages", [])

        if not pages:
            return f"No pages found matching '{filter}' in this project."

        # ── Save to workflow state ──
        state = WorkflowState(project_id=project_id, step="pages_listed", pages=pages)
        _set_workflow(project_id, state)
        logger.info("Workflow state saved: %d pages found", len(pages))

        lines = [f"Found {len(pages)} test scenario page(s):\n"]
        for i, p in enumerate(pages):
            lines.append(f"  {i + 1}. Page ID: {p['id']}  |  Title: {p['title']}")
        lines.append(f"\nNext: call get_test_prompt(page_selection=\"<title or number>\") to pick a page.")
        lines.append(f"If you lose this data later, call get_workflow_status() to retrieve it.")
        return "\n".join(lines)
    except Exception as e:
        logger.exception("Exception during list_test_scenario_pages")
        return f"Error calling backend: {str(e)}"


@mcp.tool()
async def get_test_prompt(page_selection: str, ctx: Context, project_id: str = None) -> str:
    """
    STEP 2: Fetch a test generation prompt for a page from the discovered list.
    Pass the page title, page number (from the list), or page ID.
    The server resolves the actual page ID from saved workflow state.
    Examples: "Login Flow Scenarios", "1", "12345"

    Args:
        page_selection: Page title, list number, or page ID to select.
        project_id: Optional project ID. Defaults to PROJECT_ID env var.
    """
    project_id, api_url, api_key = get_config(project_id)

    if err := validate_config(project_id, api_key):
        return err

    if err := _enforce_step(project_id, "prompt_fetched"):
        return err

    if not page_selection or not page_selection.strip():
        return "Error: page_selection is required. Pass a page title, number, or ID."

    state = _get_workflow(project_id)
    page = _resolve_page(state, page_selection)

    if not page:
        titles = "\n".join(
            f"  {i + 1}. {p['title']} (ID: {p['id']})"
            for i, p in enumerate(state.pages)
        )
        return (
            f"Page not found for selection: '{page_selection}'.\n"
            f"Available pages:\n{titles}\n\n"
            f"Pass the page number, title, or ID."
        )

    confluence_page_id = page["id"]
    logger.info("Resolved page selection '%s' → page_id=%s", page_selection, confluence_page_id)

    logger.info("Fetching test prompt for page %s (Project: %s)", confluence_page_id, project_id)

    try:
        client = get_client(ctx)
        resp = await client.post(
            f"{api_url}/api/test/parse-scenarios-internal",
            headers={"X-API-Key": api_key},
            json={
                "confluence_page_id": confluence_page_id,
                "project_id": project_id,
            },
            timeout=60.0,
        )

        if resp.status_code != 200:
            logger.error("Backend returned %d", resp.status_code)
            return f"Error: Backend returned {resp.status_code} — {resp.text}"

        data = resp.json()
        session_id = data.get("session_id", "")
        prompt = data.get("prompt", "")
        page_title = data.get("page_title", "")
        scenarios = data.get("scenarios", [])

        logger.info("Got prompt (%d chars), %d scenarios, session=%s", len(prompt), len(scenarios), session_id)

        # ── Save to workflow state ──
        state.step = "prompt_fetched"
        state.selected_page_id = confluence_page_id
        state.selected_page_title = page_title
        state.session_id = session_id
        state.prompt = prompt
        state.scenario_count = len(scenarios)
        _set_workflow(project_id, state)
        logger.info("Workflow state updated: session=%s, page=%s", session_id, page_title)

        return (
            f"SESSION_ID: {session_id}\n"
            f"PAGE: {page_title}\n"
            f"SCENARIOS: {len(scenarios)}\n\n"
            f"--- PROMPT START ---\n"
            f"{prompt}\n"
            f"--- PROMPT END ---\n\n"
            f"INSTRUCTIONS:\n"
            f"1. Use the prompt above to generate Gherkin .feature files.\n"
            f"   Follow ALL instructions in the prompt (tags, format, coverage summary, etc.)\n"
            f"2. AFTER generating, DISPLAY the complete Gherkin output to the user in chat.\n"
            f"   Do NOT auto-submit. The user must review the output first.\n"
            f"3. Ask the user if they want to submit the test cases.\n"
            f"4. ONLY if the user confirms (e.g. says 'submit', 'yes', 'send it', 'push'), call:\n"
            f"   submit_test_cases(gherkin=\"<the generated output>\")\n"
            f"   The session_id is stored on the server — you do NOT need to pass it.\n"
            f"   Do NOT call submit_test_cases without explicit user confirmation."
        )
    except Exception as e:
        logger.exception("Exception during get_test_prompt")
        return f"Error calling backend: {str(e)}"


@mcp.tool()
async def submit_test_cases(gherkin: str, ctx: Context, project_id: str = None) -> str:
    """
    STEP 3: Submit generated Gherkin test cases to the frontend via SSE.
    The session_id is automatically read from saved workflow state — you don't need to pass it.
    Only call this when the user explicitly confirms submission.

    Args:
        gherkin: The complete Gherkin .feature file output (all features concatenated)
        project_id: Optional project ID. Defaults to PROJECT_ID env var.
    """
    project_id, api_url, api_key = get_config(project_id)

    if err := validate_config(project_id, api_key):
        return err

    if err := _enforce_step(project_id, "submitted"):
        return err

    if not gherkin.strip():
        return "Error: gherkin parameter cannot be empty."

    state = _get_workflow(project_id)
    session_id = state.session_id
    logger.info("Using session_id from workflow state: %s", session_id)

    logger.info("Submitting test cases (%d chars, session=%s)", len(gherkin), session_id)

    try:
        client = get_client(ctx)
        resp = await client.post(
            f"{api_url}/api/test/submit-gherkin-internal",
            headers={"X-API-Key": api_key},
            json={
                "project_id": project_id,
                "gherkin": gherkin,
                "session_id": session_id,
            },
        )

        if resp.status_code != 200:
            logger.error("Backend returned %d", resp.status_code)
            return f"Error: Backend returned {resp.status_code} — {resp.text}"

        data = resp.json()
        logger.info("Submit successful: %s (project: %s)", data.get('status'), project_id)

        # ── Update workflow state ──
        state.step = "submitted"
        state.submitted = True
        _set_workflow(project_id, state)

        return f"Test cases submitted successfully to project {project_id}. Session: {data.get('session_id')}. The frontend SSE listener at /api/test/listen/{project_id} will receive the Gherkin output."
    except Exception as e:
        logger.exception("Exception during submit_test_cases")
        return f"Error calling backend: {str(e)}"


def main():
    mcp.run()


if __name__ == "__main__":
    main()
