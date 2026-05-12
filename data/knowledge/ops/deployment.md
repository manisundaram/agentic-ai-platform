# Deployment

This platform supports local development, Docker Compose, and Railway cloud deployment.

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Configure environment
copy .env.example .env   # Windows
cp .env.example .env     # macOS/Linux

# Start the service
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Open the demo
open http://localhost:8000/demo
```

## Docker Compose

```bash
docker compose up --build
```

This starts the FastAPI service with production settings. The demo is accessible at `http://localhost:8000/demo`.

## Railway

Railway configuration is in `deployment/railway.toml`. The service is configured to use the `Dockerfile` at the repo root. Required environment variables must be set in the Railway dashboard:
- `OPENAI_API_KEY`
- `LANGSMITH_API_KEY`
- `LANGSMITH_PROJECT`

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | For OpenAI | - | OpenAI API key |
| `LLM_PROVIDER` | No | `openai` | LLM provider: openai, gemini, anthropic, ollama, mock |
| `EMBEDDING_PROVIDER` | No | `openai` | Embedding provider |
| `LANGSMITH_API_KEY` | For tracing | - | LangSmith authentication |
| `LANGSMITH_PROJECT` | No | `agentic-ai-platform` | LangSmith project name |
| `DEFAULT_COLLECTION_NAME` | No | `default` | ChromaDB collection |
| `CHROMA_PERSIST_DIR` | No | - | Path for persistent ChromaDB |
