import logging
import warnings
import httpx
from contextlib import asynccontextmanager
from mcp.server.fastmcp import FastMCP, Context
from .config import get_config, validate_config, get_client

warnings.filterwarnings("ignore", message="Unverified HTTPS request")

logger = logging.getLogger("mcp.code-documentation")


# ─── Lifespan (shared HTTP client) ────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app):
    """Shared httpx client for the lifetime of the server."""
    async with httpx.AsyncClient(timeout=120.0, verify=False) as client:
        logger.info("Code documentation server: HTTP client created")
        yield {"client": client}
    logger.info("Code documentation server: HTTP client closed")


mcp = FastMCP("code-documentation", lifespan=lifespan)


# ─── Tool 1: prepare_code_documentation ───────────────────────────────────────

@mcp.tool()
async def prepare_code_documentation(
    ctx: Context,
    scope: str = "whole repository",
    project_id: str = None,
) -> str:
    """
    STEP 1: Start the code documentation workflow.
    Call this when the user asks to generate code documentation, code
    summary, repo documentation, or any phrase that means "document
    this codebase and publish it".

    Returns a structured workflow prompt that the IDE AI must follow:
    explore the repo, generate Markdown, show it to the user, get
    explicit approval, then call push_code_documentation.

    Args:
        scope: What to document (e.g. "whole repository", "src/api module",
               "authentication flow"). Becomes part of the Confluence page title.
        project_id: Optional project ID. Defaults to PROJECT_ID env var.
    """
    project_id, _, api_key = get_config(project_id)
    if err := validate_config(project_id, api_key):
        return err

    logger.info("Preparing code documentation workflow (scope=%s, project=%s)", scope, project_id)

    return (
        f"CODE DOCUMENTATION TASK\n"
        f"Scope: {scope}\n"
        f"Project: {project_id}\n\n"
        f"===================================================================\n"
        f"MANDATORY workflow — follow these steps in order. Do NOT skip or\n"
        f"reorder. Do NOT call push_code_documentation until Step 4.\n"
        f"===================================================================\n\n"
        f"STEP 1 — EXPLORE THE CODEBASE\n"
        f"  - Read the repo structure (top-level folders and files).\n"
        f"  - Identify entry points, frameworks, and overall architecture.\n"
        f"  - Read the most important files: config, main modules,\n"
        f"    routers/controllers, key services.\n"
        f"  - If the scope is a sub-path, focus only on files inside it.\n\n"
        f"STEP 2 — GENERATE THE MARKDOWN DOCUMENT\n"
        f"  Produce a comprehensive Markdown document with these sections:\n"
        f"    1. Overview         — what the project/scope does (2-3 sentences).\n"
        f"    2. Architecture     — major components and how they connect.\n"
        f"    3. Tech stack       — languages, frameworks, key dependencies.\n"
        f"    4. Directory layout — annotated tree of important paths.\n"
        f"    5. Key modules      — one paragraph per significant module.\n"
        f"    6. External integrations — DBs, APIs, third-party services.\n"
        f"    7. Configuration    — required env vars and config files.\n"
        f"    8. Build & run      — how to start it locally.\n"
        f"    9. Notable conventions / gotchas — non-obvious behaviour.\n\n"
        f"  Use standard Markdown only: `##` headers, fenced code blocks\n"
        f"  with language tags, bullet lists. No HTML, no embedded images.\n\n"
        f"STEP 3 — SHOW & ASK FOR APPROVAL\n"
        f"  - Display the FULL Markdown to the user in chat, inside a single\n"
        f"    fenced code block. Do NOT summarize, paraphrase, or abbreviate.\n"
        f"  - Then ask, in your own words: \"Should I publish this to your\n"
        f"    project's linked Confluence space?\"\n"
        f"  - WAIT for explicit confirmation (e.g. 'yes', 'publish', 'push').\n"
        f"  - If the user wants edits, revise the document and re-show.\n"
        f"  - Do NOT call push_code_documentation yet.\n\n"
        f"STEP 4 — PUBLISH (only after explicit user approval)\n"
        f"  - Determine the current commit SHA by running: git rev-parse HEAD\n"
        f"    (use your shell/Bash tool in the user's repo directory).\n"
        f"    If git is unavailable, pass an empty string for commit_sha.\n"
        f"  - Call push_code_documentation(\n"
        f"        content=\"<the entire Markdown from Step 2>\",\n"
        f"        scope=\"{scope}\",\n"
        f"        commit_sha=\"<the SHA you just resolved>\"\n"
        f"    )\n"
        f"  - Show the user the returned Confluence page URL.\n\n"
        f"The backend resolves the target Confluence space from project_id —\n"
        f"you do not need to know the space key. The page will be created\n"
        f"under the project's 'Code Documentation' parent page and labelled\n"
        f"'code-documentation' so the SDLC frontend can list it."
    )


# ─── Tool 2: push_code_documentation ──────────────────────────────────────────

@mcp.tool()
async def push_code_documentation(
    content: str,
    ctx: Context,
    scope: str = "whole repository",
    commit_sha: str = "",
    project_id: str = None,
) -> str:
    """
    STEP 2: Publish a finished Markdown code documentation to the project's
    linked Confluence space.

    Only call this AFTER prepare_code_documentation has run AND the user has
    explicitly approved publication. Never call this without user confirmation.

    The Confluence space is resolved from project_id on the backend using the
    project owner's stored Atlassian credentials — you do not need to know
    or pass the space key.

    Args:
        content:    The full Markdown documentation body to publish.
        scope:      What was documented (becomes part of the page title).
        commit_sha: Current git commit SHA. Pass empty string if unknown
                    (the backend will use "unknown" in the title).
        project_id: Optional project ID. Defaults to PROJECT_ID env var.
    """
    project_id, api_url, api_key = get_config(project_id)
    if err := validate_config(project_id, api_key):
        return err

    if not content or not content.strip():
        return "Error: content (the Markdown documentation) cannot be empty."

    logger.info(
        "Publishing code documentation (scope=%s, sha=%s, %d chars) for project %s",
        scope, commit_sha or "(none)", len(content), project_id,
    )

    try:
        client = get_client(ctx)
        resp = await client.post(
            f"{api_url}/api/integrations/code-documentation/push-to-confluence-internal",
            headers={"X-API-Key": api_key},
            json={
                "scope": scope,
                "content": content,
                "commit_sha": commit_sha,
                "project_id": project_id,
            },
            timeout=120.0,
        )
    except Exception as e:
        logger.exception("Exception during push_code_documentation")
        return f"Error calling backend: {type(e).__name__}: {str(e)}"

    if resp.status_code != 200:
        logger.error("Backend returned %d — %s", resp.status_code, resp.text[:500])
        return f"Error: Backend returned {resp.status_code} — {resp.text[:500]}"

    data = resp.json()
    page_url = data.get("web_url") or "(no URL returned)"
    page_id = data.get("page_id", "?")
    title = data.get("title", "?")
    created = data.get("created", True)

    status_line = (
        "Created new Confluence page."
        if created
        else "A page already existed for this scope+commit — returning the existing one."
    )

    return (
        f"{status_line}\n\n"
        f"- Title: {title}\n"
        f"- Page ID: {page_id}\n"
        f"- URL: {page_url}\n"
        f"- Label: code-documentation\n\n"
        f"The page is now visible in the project's Confluence space under the\n"
        f"'Code Documentation' parent page, and will be indexed for future RAG queries."
    )


# ─── Entrypoint ────────────────────────────────────────────────────────────────

def main():
    mcp.run()


if __name__ == "__main__":
    main()
