from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from ai_service_kit.health import apply_operational_middleware, register_operational_endpoints
from ai_service_kit.logging import Logger, setup_enhanced_logging
from ai_service_kit.providers import BaseLLMProvider, LLMProviderFactory, MockLLMProvider
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from langgraph.types import Command
from langsmith.run_helpers import tracing_context
from pydantic import BaseModel, ConfigDict, Field

from .bootstrap import build_service_context, debug_snapshot
from .config import get_settings
from .graph.builder import get_graph
from .graph.nodes import GraphNodeDependencies
from .logging.sse_handler import LOG_QUEUE, enqueue_demo_log, install_sse_log_handler
from .mcp.client import MCPServerConfig, MCPToolAdapter
from .retrieval import LlamaIndexRetriever, RetrievalConfig


logger = logging.getLogger(__name__)


class AgentQueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1)
    thread_id: str | None = None
    max_iterations: int = Field(default=2, ge=1, le=10)


class AgentQueryResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    thread_id: str
    answer: str
    sources: list[dict[str, Any]]
    tool_calls_made: list[dict[str, Any]]
    iteration_count: int
    requires_human_review: bool = False
    langsmith_trace_url: str | None


class AgentResumeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    thread_id: str = Field(min_length=1)
    human_decision: Literal["approve", "reject", "modify"]
    modified_answer: str | None = None


class AgentTraceResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    thread_id: str
    values: dict[str, Any]
    next_nodes: list[str]
    created_at: str | None


class RetrievalDocumentIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str | None = None
    content: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalIndexRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    documents: list[RetrievalDocumentIn] = Field(min_length=1)


class MCPToolsListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tools: list[dict[str, Any]]


@dataclass(slots=True)
class AppRuntime:
    retriever: LlamaIndexRetriever
    graph: Any
    mcp_adapter: MCPToolAdapter


class _AgentGenerator:
    def __init__(self, llm_provider: BaseLLMProvider) -> None:
        self._llm_provider = llm_provider

    @staticmethod
    def _normalize_message_roles(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        allowed_roles = {"system", "assistant", "user", "function", "tool", "developer"}
        normalized_messages: list[dict[str, Any]] = []
        for message in messages:
            message_role = str(message.get("role", "user"))
            if message_role not in allowed_roles:
                message_role = "system"
            normalized_messages.append({**message, "role": message_role})
        return normalized_messages

    async def generate(self, *, query: str, context: str, messages: list[dict[str, Any]]) -> dict[str, Any]:
        llm_messages = self._normalize_message_roles(list(messages))
        llm_messages.append(
            {
                "role": "user",
                "content": f"Query: {query}\n\nContext:\n{context}",
            }
        )
        settings = get_settings()
        model = settings.llm_provider_config().get("model") or "gpt-4o-mini"
        llm_result = await self._llm_provider.generate(llm_messages, model=model)
        return {
            "answer": llm_result.content,
            "tool_calls": [{"tool": "llm", "provider": settings.resolved_provider("llm"), "model": llm_result.model}],
        }


class _AgentCritic:
    def __init__(self, llm_provider: BaseLLMProvider) -> None:
        self._llm_provider = llm_provider

    async def critique(self, *, query: str, context: str, answer: str) -> dict[str, Any]:
        # Check quality based on grounding and risky keywords.
        # No LLM call needed - the logic is deterministic and fast.
        has_issues = not bool(context.strip()) or not bool(answer.strip())
        risky_phrases = (
            "vpn",
            "mdm",
            "enroll",
            "enrolled",
            "production",
            "prod",
            "admin",
            "privileged",
            "password reset",
            "reset password",
            "disable",
            "mfa",
            "credential",
            "access set up",
        )
        requires_human_review = any(flag in query.lower() for flag in ("legal", "finance", "medical", *risky_phrases))
        return {
            "has_issues": has_issues,
            "feedback": "Needs stronger grounding" if has_issues else "Answer appears grounded",
            "eval_scores": {
                "groundedness": 0.35 if has_issues else 0.9,
                "completeness": 0.4 if has_issues else 0.88,
            },
            "requires_human_review": requires_human_review,
        }


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _first_env(*names: str) -> str:
    for name in names:
        value = os.getenv(name)
        if value and value.strip():
            return value.strip()
    return ""


def _langsmith_project_name() -> str:
    return _first_env("LANGSMITH_PROJECT", "LANGCHAIN_PROJECT") or "agentic-ai-platform"


def _ensure_langsmith_env_aliases() -> None:
    # Keep backward compatibility with older LANGCHAIN_* variable names.
    if not _first_env("LANGSMITH_API_KEY"):
        legacy_key = _first_env("LANGCHAIN_API_KEY")
        if legacy_key:
            os.environ["LANGSMITH_API_KEY"] = legacy_key
    if not _first_env("LANGSMITH_PROJECT"):
        legacy_project = _first_env("LANGCHAIN_PROJECT")
        if legacy_project:
            os.environ["LANGSMITH_PROJECT"] = legacy_project


def _demo_index_path() -> Path:
    return _project_root() / "demo" / "index.html"


def _langsmith_trace_url(thread_id: str) -> str | None:
    api_key = _first_env("LANGSMITH_API_KEY", "LANGCHAIN_API_KEY")
    if not api_key:
        return None
    project = _langsmith_project_name()
    return f"https://smith.langchain.com/?project={project}&query={thread_id}"


def _graph_config(thread_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": thread_id}}


def _build_initial_state(query: str, max_iterations: int) -> dict[str, Any]:
    return {
        "messages": [],
        "query": query,
        "context": "",
        "tool_calls": [],
        "iteration_count": 0,
        "max_iterations": max_iterations,
        "final_answer": "",
        "requires_human_review": False,
        "eval_scores": {},
    }


def _extract_sources(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for call in reversed(tool_calls):
        sources = call.get("sources")
        if isinstance(sources, list):
            return sources
    return []


def _build_query_response(state: dict[str, Any], thread_id: str) -> AgentQueryResponse:
    tool_calls = list(state.get("tool_calls", []))
    return AgentQueryResponse(
        thread_id=thread_id,
        answer=str(state.get("final_answer", "")),
        sources=_extract_sources(tool_calls),
        tool_calls_made=tool_calls,
        iteration_count=int(state.get("iteration_count", 0)),
        requires_human_review=bool(state.get("requires_human_review", False)),
        langsmith_trace_url=_langsmith_trace_url(thread_id),
    )


def _build_runtime() -> AppRuntime:
    settings = get_settings()
    retriever = LlamaIndexRetriever(
        RetrievalConfig(
            collection_name=settings.default_collection_name,
            default_top_k=3,
        )
    )
    llm_config = settings.llm_provider_config()
    llm_provider_name = settings.resolved_provider("llm")
    try:
        llm_provider = LLMProviderFactory().create_provider(llm_provider_name, llm_config)
        logger.info("LLM provider initialised provider=%s model=%s", llm_provider_name, llm_config.get("model"))
    except Exception:
        logger.warning("LLM provider %s unavailable, falling back to mock", llm_provider_name)
        llm_provider = MockLLMProvider({"model": "mock-llm", "seed": 17})
    graph = get_graph(
        GraphNodeDependencies(
            retriever=retriever,
            generator=_AgentGenerator(llm_provider),
            critic=_AgentCritic(llm_provider),
        )
    )
    mcp_adapter = MCPToolAdapter(
        MCPServerConfig(
            command=sys.executable,
            args=["-m", "app.mcp.server"],
            cwd=str(_project_root()),
            read_timeout_seconds=20.0,
        )
    )
    return AppRuntime(retriever=retriever, graph=graph, mcp_adapter=mcp_adapter)


def create_app() -> FastAPI:
    load_dotenv(_project_root() / ".env", override=False)
    settings = get_settings()
    _ensure_langsmith_env_aliases()
    setup_enhanced_logging(service_name=settings.app_name, environment=settings.app_env)
    install_sse_log_handler()

    service_context = build_service_context(settings)

    app = FastAPI(title=settings.app_name, debug=settings.app_debug)
    app.state.settings = settings
    app.state.service_context = service_context
    app.state.runtime = _build_runtime()
    app.state.shutdown_event = asyncio.Event()

    if settings.enable_cors:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=False,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    Logger.info(f"Starting {settings.app_name} v{settings.app_version} in {settings.app_env} mode")

    apply_operational_middleware(
        app,
        enable_cors=False,
        cors_origins=settings.cors_origins,
        enable_logging_middleware=True,
    )

    register_operational_endpoints(
        app,
        context_getter=lambda current_app: current_app.state.service_context,
        settings_snapshot_getter=lambda current_app: current_app.state.settings.masked_debug_config(),
        bootstrap_snapshot_getter=lambda current_app: debug_snapshot(current_app.state.service_context),
    )

    @app.on_event("shutdown")
    async def mark_shutdown():
        app.state.shutdown_event.set()
        logger.info("Application shutdown signaled")

    @app.get("/demo")
    @app.get("/demo/index.html")
    async def demo_index():
        demo_path = _demo_index_path()
        if not demo_path.exists():
            raise HTTPException(status_code=404, detail="Demo UI not found")
        return FileResponse(demo_path)

    @app.get("/demo/logs")
    async def demo_logs(request: Request):
        async def event_stream():
            try:
                yield ": connected\n\n"
                while True:
                    if app.state.shutdown_event.is_set():
                        logger.info("SSE log stream stopping for application shutdown")
                        break

                    if await request.is_disconnected():
                        logger.info("SSE log stream disconnected by client")
                        break

                    try:
                        payload = await asyncio.wait_for(LOG_QUEUE.get(), timeout=15.0)
                        yield f"data: {payload}\n\n"
                    except asyncio.TimeoutError:
                        if await request.is_disconnected():
                            logger.info("SSE log stream disconnected during keep-alive")
                            break
                        yield ": keep-alive\n\n"
            except asyncio.CancelledError:
                logger.info("SSE log stream cancelled during shutdown")
                raise

        logger.info("SSE log stream connected")
        return StreamingResponse(event_stream(), media_type="text/event-stream")

    @app.post("/agent/query", response_model=AgentQueryResponse)
    async def agent_query(payload: AgentQueryRequest, stream: bool = Query(default=False)):
        thread_id = payload.thread_id or f"thread-{uuid4().hex}"
        config = _graph_config(thread_id)
        initial_state = _build_initial_state(payload.query, payload.max_iterations)
        enqueue_demo_log(source="app", level="INFO", message=f"/agent/query received thread_id={thread_id}")
        logger.info("Agent query received thread_id=%s stream=%s", thread_id, stream)

        if stream:
            async def stream_events():
                last_state: dict[str, Any] = initial_state
                with tracing_context(project_name=_langsmith_project_name(), enabled=True):
                    async for chunk in app.state.runtime.graph.astream(initial_state, config=config, stream_mode="values"):
                        if isinstance(chunk, dict):
                            last_state = chunk
                        logger.info(
                            "LangGraph stream update thread_id=%s iteration=%s",
                            thread_id,
                            last_state.get("iteration_count", 0),
                        )
                        yield json.dumps(
                            {
                                "event": "state",
                                "thread_id": thread_id,
                                "iteration_count": int(last_state.get("iteration_count", 0)),
                                "answer": str(last_state.get("final_answer", "")),
                            }
                        ) + "\n"

                final_payload = _build_query_response(last_state, thread_id).model_dump(mode="json")
                yield json.dumps({"event": "final", "thread_id": thread_id, "data": final_payload}) + "\n"

            return StreamingResponse(stream_events(), media_type="application/x-ndjson")

        try:
            with tracing_context(project_name=_langsmith_project_name(), enabled=True):
                state = await app.state.runtime.graph.ainvoke(initial_state, config=config)
        except asyncio.CancelledError:
            logger.warning("Agent query cancelled before graph completion thread_id=%s", thread_id)
            raise
        logger.info("Agent query completed thread_id=%s", thread_id)
        return _build_query_response(state, thread_id)

    @app.post("/agent/resume", response_model=AgentQueryResponse)
    async def agent_resume(payload: AgentResumeRequest):
        config = _graph_config(payload.thread_id)
        note = {"role": "system", "content": f"Human decision: {payload.human_decision}"}
        update: dict[str, Any] = {
            "requires_human_review": False,
            "messages": [note],
        }
        if payload.human_decision == "modify":
            if not payload.modified_answer:
                raise HTTPException(status_code=400, detail="modified_answer is required when human_decision=modify")
            update["final_answer"] = payload.modified_answer
        if payload.human_decision == "reject":
            update["final_answer"] = "Rejected by human reviewer."

        enqueue_demo_log(
            source="hitl",
            level="WARNING",
            message=f"Human decision received thread_id={payload.thread_id} decision={payload.human_decision}",
        )
        logger.warning("HITL decision thread_id=%s decision=%s", payload.thread_id, payload.human_decision)
        with tracing_context(project_name=_langsmith_project_name(), enabled=True):
            state = await app.state.runtime.graph.ainvoke(Command(update=update), config=config)
        logger.info("Agent resume completed thread_id=%s decision=%s", payload.thread_id, payload.human_decision)
        return _build_query_response(state, payload.thread_id)

    @app.get("/agent/trace/{thread_id}", response_model=AgentTraceResponse)
    async def agent_trace(thread_id: str):
        snapshot = await app.state.runtime.graph.aget_state(_graph_config(thread_id))
        if snapshot is None:
            raise HTTPException(status_code=404, detail="Thread not found")
        created_at = getattr(snapshot, "created_at", None)
        if created_at is None:
            created_str = None
        elif hasattr(created_at, "isoformat"):
            created_str = created_at.isoformat()
        else:
            created_str = str(created_at)
        return AgentTraceResponse(
            thread_id=thread_id,
            values=dict(snapshot.values or {}),
            next_nodes=list(snapshot.next or ()),
            created_at=created_str,
        )

    @app.post("/retrieval/index")
    async def retrieval_index(payload: RetrievalIndexRequest):
        logger.info("Retrieval index request received documents=%s", len(payload.documents))
        docs = [document.model_dump(mode="python") for document in payload.documents]
        return await app.state.runtime.retriever.index_documents(docs)

    @app.get("/retrieval/stats")
    async def retrieval_stats():
        logger.info("Retrieval stats requested")
        return await app.state.runtime.retriever.get_collection_stats()

    @app.post("/mcp/tools/list", response_model=MCPToolsListResponse)
    async def mcp_tools_list():
        logger.info("MCP tools list requested")
        tools = await app.state.runtime.mcp_adapter.list_tools()
        return MCPToolsListResponse(tools=[tool.model_dump(mode="json") for tool in tools])

    return app


app = create_app()
