# Installation and Setup

This guide covers local setup for development, demo runs, and basic validation.

## Prerequisites

- Python 3.11+
- pip
- Optional: Docker Desktop for containerized runs

## Local Setup

1. Install dependencies.

```bash
pip install -r requirements.txt
```

2. Create environment file.

```bash
copy .env.example .env
```

3. Choose provider mode.

- Mock mode for no-credential local iteration
- Real provider mode (OpenAI, Gemini, Anthropic, or Ollama) for full validation

4. Run the API.

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

5. Open the demo UI.

- http://localhost:8000/demo

## Required Environment Variables

Minimal local env values usually include:

- PROVIDER
- MODEL (optional override)
- LANGSMITH_API_KEY or legacy LANGCHAIN_API_KEY (optional, for tracing)
- LANGSMITH_PROJECT or legacy LANGCHAIN_PROJECT (optional, for trace grouping)

See [.env.example](../.env.example) for full template.

## Smoke Tests

After startup, verify:

- GET /ping
- GET /health
- GET /retrieval/stats
- POST /mcp/tools/list

Then run one end-to-end query from the demo UI.

## Container Path

Run with Docker Compose:

```bash
docker compose up --build
```

## Common Setup Issues

1. Backend does not start:

- Validate Python environment and dependency install
- Ensure chosen provider has required keys in .env

2. LangSmith not showing traces:

- Confirm LANGSMITH_API_KEY is set
- If only LANGCHAIN\_\* is set, ensure compatibility aliases are active in app startup

3. Slow first query:

- Use stream mode in UI for live progress visibility
- Provider latency and iteration count affect total completion time

## Related Docs

- Project overview: [../README.md](../README.md)
- Runtime architecture: [ARCHITECTURE.md](ARCHITECTURE.md)
- Framework comparison: [FRAMEWORK_COMPARISON.md](FRAMEWORK_COMPARISON.md)
- Demo walkthrough: [../demo/README.md](../demo/README.md)
