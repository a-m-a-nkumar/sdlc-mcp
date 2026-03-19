import json
import logging
import httpx
from contextlib import asynccontextmanager
from mcp.server.fastmcp import FastMCP, Context
from .config import get_config, validate_config, get_client

logger = logging.getLogger("mcp.enhance-prompt")


# ─── Lifespan (shared HTTP client) ────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app):
    """Create a shared httpx client that lives for the entire server lifetime."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        logger.info("Enhance server: HTTP client created")
        yield {"client": client}
    logger.info("Enhance server: HTTP client closed")


mcp = FastMCP("enhance-prompt", lifespan=lifespan)


# ─── Streaming helper ─────────────────────────────────────────────────────────

async def _stream_enhance(client: httpx.AsyncClient, api_url: str, api_key: str,
                          project_id: str, task: str) -> str:
    """Shared streaming logic for both the prompt and tool."""
    result = ""
    async with client.stream(
        "POST",
        f"{api_url}/api/orchestration/query-internal",
        headers={"X-API-Key": api_key},
        json={"project_id": project_id, "query": task, "max_chunks": 5, "return_prompt": True}
    ) as r:
        if r.status_code != 200:
            logger.error("Backend returned %d", r.status_code)
            return f"Error: Backend returned {r.status_code}"

        async for line in r.aiter_lines():
            if line.startswith("data: "):
                try:
                    data = json.loads(line[6:])
                    if data.get("type") == "enhanced_prompt":
                        result = data.get("content", "")
                        logger.info("Received enhanced prompt (%d chars)", len(result))
                    elif data.get("type") == "chunk":
                        result += data.get("content", "")
                    elif data.get("type") == "error":
                        result += f"\n[Remote Error: {data.get('message')}]"
                        logger.error("Remote error: %s", data.get('message'))
                except json.JSONDecodeError:
                    continue
    return result


# ─── Prompt ────────────────────────────────────────────────────────────────────

@mcp.prompt()
async def enhance(task: str, project_id: str = None) -> str:
    """Enhance a dev task with context from your project docs"""
    project_id, api_url, api_key = get_config(project_id)

    if err := validate_config(project_id, api_key):
        return err

    logger.info("Enhancing task: %s (Project: %s)", task, project_id)

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            result = await _stream_enhance(client, api_url, api_key, project_id, task)
    except Exception as e:
        logger.exception("Exception during enhance")
        return f"Error calling backend: {str(e)}"

    logger.info("Enhancement complete.")
    return (
        f"<enhanced_prompt>\n{result}\n</enhanced_prompt>\n\n"
        "Display this enhanced prompt to the user, then wait for their next instruction."
    )


# ─── Tool ──────────────────────────────────────────────────────────────────────

@mcp.tool()
async def enhance_task(task: str, ctx: Context, project_id: str = None) -> str:
    """
    Search project documentation and return an enhanced prompt with relevant context.
    Use this to get background info, requirements, or architecture context for a task.

    This is a standalone remote lookup — it queries an API and returns results directly.

    After receiving the result, display it to the user and wait for their instructions.

    Args:
        task: The task or query to enhance
        project_id: Optional project ID/GUID. Defaults to PROJECT_ID env var.
    """
    project_id, api_url, api_key = get_config(project_id)

    if err := validate_config(project_id, api_key):
        return err

    logger.info("Enhancing task: %s (Project: %s)", task, project_id)

    try:
        client = get_client(ctx)
        result = await _stream_enhance(client, api_url, api_key, project_id, task)
    except Exception as e:
        logger.exception("Exception during enhance_task")
        return f"Error calling backend: {str(e)}"

    logger.info("Enhancement complete.")
    return (
        f"<enhanced_prompt>\n{result}\n</enhanced_prompt>\n\n"
        "Display this enhanced prompt to the user, then wait for their next instruction."
    )


def main():
    mcp.run()


if __name__ == "__main__":
    main()
