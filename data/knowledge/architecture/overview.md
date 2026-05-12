# Platform Overview

agentic-ai-platform is a production-style IT Support AI Agent built on LangGraph, LlamaIndex, MCP, LangSmith, and FastAPI. It demonstrates multi-turn reasoning, retrieval-augmented generation, human-in-the-loop approval, and real-time observability in a single deployable service.

## Core Components

- **LangGraph**: Orchestrates the agent workflow as a typed state graph with four nodes: retrieve_context, generate_answer, critique_answer, and human_review.
- **LlamaIndex**: Handles document loading, chunking, embedding, and retrieval from ChromaDB.
- **MCP (Model Context Protocol)**: Provides standardized tool discovery and invocation. The MCP server exposes search, calculation, and time tools.
- **LangSmith**: Traces every run with span-level observability. Trace URLs are returned in every query response.
- **FastAPI**: Serves the REST API with Pydantic-validated request/response models and Server-Sent Events for real-time log streaming.
- **ai-service-kit**: Shared operational library providing LLM/embedding provider abstraction, health endpoints, structured logging, and secret masking.

## Key Endpoints

- `POST /agent/query`: Submit a query, receive a streaming or non-streaming response with answer, sources, and trace URL.
- `POST /agent/resume`: Resume a HITL-paused run with approve/reject/modify decision.
- `GET /agent/trace/{thread_id}`: Inspect current graph state for a thread.
- `POST /retrieval/index`: Index new documents into ChromaDB.
- `GET /retrieval/stats`: Check collection statistics.
- `GET /demo/logs`: SSE stream of live backend logs.
- `GET /demo`: Serve the demo UI.

## Design Philosophy

The platform is designed to demonstrate engineering judgment: choosing well-supported frameworks where they eliminate boilerplate, building clean abstractions where they don't, and validating every component with evals and tracing.
