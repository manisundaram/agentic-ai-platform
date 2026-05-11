# agentic-ai-platform

Agentic IT Support platform built for production-style workflows, not single-turn chatbot demos.

It combines LangGraph orchestration, Human-In-The-Loop review, LangSmith tracing, LlamaIndex retrieval, and MCP tool contracts behind a FastAPI service.

## What This Project Demonstrates

- Multi-turn reasoning with explicit workflow state
- Human approval gates for risky actions
- Resumeable runs using thread-level checkpoint state
- Real-time observability through logs and trace links
- Evaluation baselines captured in eval artifacts

## Architecture

```mermaid
flowchart LR
		C[Client / Demo UI] --> API[FastAPI Service]

		API --> Q[POST /agent/query]
		API --> R[POST /agent/resume]
		API --> T[GET /agent/trace/{thread_id}]

		Q --> LG[LangGraph Runtime]
		R --> LG
		T --> LG

		LG --> RET[LlamaIndex Retriever]
		RET --> VDB[(ChromaDB)]

		LG --> MCPA[MCP Tool Adapter]
		MCPA --> MCPS[MCP Server]

		API --> OPS[Operational Endpoints]
		OPS --> PING[GET /ping]
		OPS --> HEALTH[GET /health]
		OPS --> DIAG[GET /diagnostics]

		LG -. trace .-> LS[LangSmith]
		API --> EV[Eval Runner]
		EV --> OUT[[evals/results/latest.json]]
```

## Quick Start

1. Install dependencies.

```bash
pip install -r requirements.txt
```

2. Create local environment file.

```bash
copy .env.example .env
```

3. Run the API.

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

4. Open demo UI.

- http://localhost:8000/demo

## Documentation Map

- Demo walkthrough and media guide: [demo/README.md](demo/README.md)
- Architecture details: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- Installation and setup: [docs/INSTALLATION.md](docs/INSTALLATION.md)
- Framework decision story: [docs/FRAMEWORK_COMPARISON.md](docs/FRAMEWORK_COMPARISON.md)
- Current evaluation artifact: [evals/results/latest.json](evals/results/latest.json)

## Why This Stack

- LangGraph: explicit node-based workflows with retries, branching, and interrupts
- LlamaIndex: fast retrieval iteration over document corpora
- MCP: consistent tool discovery/invocation contracts
- LangSmith: runtime traces for debugging and validation

For side-by-side tradeoff reasoning against sibling repos, see [docs/FRAMEWORK_COMPARISON.md](docs/FRAMEWORK_COMPARISON.md).

## Related Repositories

- agents-api: custom ReAct baseline used for orchestration comparison
- semantic-search-api: raw Chroma-focused retrieval baseline
- rag-api: complementary RAG service patterns
- ai-service-kit: shared provider, config, and ops foundation
