# Architecture

This document describes the runtime architecture for agentic-ai-platform and how requests move through the system.

## System Overview

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

## Core Components

- FastAPI service:
  Request ingress, API contracts, demo route hosting, and operational endpoints
- LangGraph runtime:
  Stateful orchestration with retries, conditional routing, and checkpointed thread flow
- Retrieval layer:
  LlamaIndex retriever over Chroma vector store for context grounding
- MCP tool layer:
  Standardized tool discovery and invocation contracts for external capabilities
- Observability layer:
  Structured logs, SSE log stream, and LangSmith trace links

## End-to-End Query Flow

1. Client sends POST /agent/query with query, optional thread_id, and max_iterations.
2. API builds initial graph state and starts LangGraph execution.
3. retrieve_context node gathers retrieval context from LlamaIndex + Chroma.
4. generate_answer node calls provider-backed LLM with normalized conversation state.
5. critique_answer node evaluates quality and decides retry or completion.
6. If risky action is detected, graph pauses for Human-In-The-Loop approval.
7. Client calls POST /agent/resume to approve, reject, or modify continuation.
8. Client reads GET /agent/trace/{thread_id} for persisted state snapshots.

## State and Safety Model

- Thread-scoped state:
  Each run is keyed by thread_id so progression and resume are deterministic
- Human approval gate:
  Sensitive actions can be interrupted and resumed only by explicit decision
- Iteration limit:
  max_iterations bounds retries and prevents unbounded loops

## Observability Model

- Live logs:
  GET /demo/logs streams runtime events to the demo UI via SSE
- Traceability:
  LangSmith trace URL is attached to query responses when configured
- Health visibility:
  /ping, /health, and /diagnostics expose runtime status and dependencies

## Related Docs

- Project overview: [../README.md](../README.md)
- Installation and setup: [INSTALLATION.md](INSTALLATION.md)
- Framework tradeoffs: [FRAMEWORK_COMPARISON.md](FRAMEWORK_COMPARISON.md)
- Demo walkthrough: [../demo/README.md](../demo/README.md)
