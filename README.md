# brd-enhancer-mcp

A package that ships seven MCP servers for an AI-assisted SDLC workflow. Each is a
separate executable you can register on its own in your IDE's MCP config:

| Server | Executable | Purpose |
|---|---|---|
| enhance-prompt | `prompt-enhancer-mcp` | Enhance dev tasks with project documentation context |
| test-workflow | `test-workflow-mcp` | Generate & submit Gherkin test cases from Confluence |
| pipeline-analyzer | `pipeline-analyzer-mcp` | Analyze Harness pipeline failures with RAG context from past incidents |
| code-documentation | `code-documentation-mcp` | Generate Markdown code documentation and publish it to Confluence |
| unit-test | `unit-test-mcp` | Generate unit tests, run them with coverage, and report coverage KPIs |
| code-quality | `code-quality-mcp` | Measure code complexity + lint/quality (whole-repo or changed files) + a Copilot-as-judge review |
| security | `security-mcp` | Scan SAST (semgrep) + dependency CVEs (trivy) + secrets (gitleaks); finding-led |

---

## Prerequisites

### ⚠️ Use Official Python (NOT Microsoft Store Python)

Download Python from **https://www.python.org/downloads/**

During installation, make sure to check:
> ✅ **"Add Python to PATH"**

**Verify you have the correct Python:**
```bash
where python
```
| Output | Status |
|--------|--------|
| `C:\Python313\python.exe` | ✅ Official Python — Good |
| `C:\Users\...\WindowsApps\python.exe` | ❌ Microsoft Store Python — Reinstall from python.org |

---

## Installation

### Option A — Global Install (Recommended for most developers)

```bash
pip install git+https://github.com/arushsingh17/mcp.git
```

**Verify installation:**
```bash
pip show brd-enhancer-mcp
where brd-enhancer-mcp        # Windows
which brd-enhancer-mcp        # Mac/Linux
```

---

### Option B — Virtual Environment Install

```bash
# Step 1: Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # Mac/Linux

# Step 2: Install the package
pip install git+https://github.com/arushsingh17/mcp.git

# Step 3: Get the exact executable path (needed for config)
python -c "import shutil; print(shutil.which('brd-enhancer-mcp'))"
```

Example output:
```
C:\Users\YourName\Desktop\myproject\venv\Scripts\brd-enhancer-mcp.exe
```
> Copy this path — you will need it in the config below.

---

## Configuration

### 🌍 Global Install Config

Since `brd-enhancer-mcp` is registered in system PATH, no file path is needed.

**Claude Desktop** → `%APPDATA%\Claude\claude_desktop_config.json` (Windows)
**Claude Desktop** → `~/Library/Application Support/Claude/claude_desktop_config.json` (Mac)
**Claude Code** → `~/.claude.json`

```json
{
    "mcpServers": {
        "brd-enhancer": {
            "command": "brd-enhancer-mcp",
            "env": {
                "API_KEY": "your_api_key",
                "PROJECT_ID": "your_project_id",
                "API_URL": "https://your-backend.com"
            }
        }
    }
}
```

---

### 📦 Virtual Environment Config

Use the full path you got from the command above.

> ⚠️ On Windows replace every `\` with `\\` in the path

```json
{
    "mcpServers": {
        "brd-enhancer": {
            "command": "C:\\Users\\YourName\\Desktop\\myproject\\venv\\Scripts\\brd-enhancer-mcp.exe",
            "env": {
                "API_KEY": "your_api_key",
                "PROJECT_ID": "your_project_id",
                "API_URL": "https://your-backend.com"
            }
        }
    }
}
```

**Mac/Linux venv config:**
```json
{
    "mcpServers": {
        "brd-enhancer": {
            "command": "/Users/yourname/myproject/venv/bin/brd-enhancer-mcp",
            "env": {
                "API_KEY": "your_api_key",
                "PROJECT_ID": "your_project_id",
                "API_URL": "https://your-backend.com"
            }
        }
    }
}
```

---

## Environment Variables

### Shared (all servers — only three vars total)

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `API_KEY` | ✅ Yes | — | Your personal backend API key |
| `PROJECT_ID` | ✅ Yes | — | Your project ID/GUID |
| `API_URL` | ❌ No | `http://localhost:8000` | Backend URL |

### enhance-prompt only (optional tech-stack hints)

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `FRONTEND_REQUIREMENTS` | ❌ No | — | e.g. `react18` — appended to the enhanced prompt |
| `BACKEND_REQUIREMENTS` | ❌ No | — | e.g. `redis,postgresql,express` — appended to the enhanced prompt |

### pipeline-analyzer — no Harness env vars needed

The pipeline-analyzer used to require five `HARNESS_*` env vars. It no longer does — the backend resolves your stored Harness credentials by project owner. Link Harness once in **Settings → Harness** on the SDLC frontend; the MCP picks it up automatically (cached 5 min, so changes propagate within ~5 minutes without a restart).

### code-documentation — uses the project owner's Atlassian credentials

No Confluence env vars needed. The backend uses the project owner's stored Atlassian credentials and the project's linked Confluence space to publish documentation pages.

### Same config shape for all four servers

```json
{
    "mcpServers": {
        "enhance-prompt": {
            "command": "prompt-enhancer-mcp",
            "env": {
                "PROJECT_ID": "your_project_id",
                "API_KEY":    "your_backend_api_key",
                "API_URL":    "https://your-backend.com"
            }
        },
        "test-workflow": {
            "command": "test-workflow-mcp",
            "env": {
                "PROJECT_ID": "your_project_id",
                "API_KEY":    "your_backend_api_key",
                "API_URL":    "https://your-backend.com"
            }
        },
        "pipeline-analyzer": {
            "command": "pipeline-analyzer-mcp",
            "env": {
                "PROJECT_ID": "your_project_id",
                "API_KEY":    "your_backend_api_key",
                "API_URL":    "https://your-backend.com"
            }
        },
        "code-documentation": {
            "command": "code-documentation-mcp",
            "env": {
                "PROJECT_ID": "your_project_id",
                "API_KEY":    "your_backend_api_key",
                "API_URL":    "https://your-backend.com"
            }
        }
    }
}
```

---

## Quick Reference

| Scenario | Command in config |
|----------|-------------------|
| Global install (Official Python) | `"command": "brd-enhancer-mcp"` |
| Global install (Microsoft Store Python) | Full path from `where brd-enhancer-mcp` |
| Virtual environment (any OS) | Full path from `python -c "import shutil; print(shutil.which('brd-enhancer-mcp'))"` |
