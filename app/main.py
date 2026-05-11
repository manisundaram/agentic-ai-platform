from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from ai_service_kit.health import apply_operational_middleware, register_operational_endpoints
from ai_service_kit.logging import Logger, setup_enhanced_logging
from ai_service_kit.providers import MockLLMProvider
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from langgraph.types import Command
from langsmith.run_helpers import tracing_context
from pydantic import BaseModel, ConfigDict, Field

from .bootstrap import build_service_context, debug_snapshot
from .config import get_settings
from .graph.builder import get_graph
from .graph.nodes import GraphNodeDependencies
from .mcp.client import MCPServerConfig, MCPToolAdapter
from .retrieval import LlamaIndexRetriever, RetrievalConfig


class AgentQueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1)
    thread_id: str | None = None
    max_iterations: int = Field(default=2, ge=1, le=10)


class AgentQueryResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str
    sources: list[dict[str, Any]]
    tool_calls_made: list[dict[str, Any]]
    iteration_count: int
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
    def __init__(self, llm_provider: MockLLMProvider) -> None:
        self._llm_provider = llm_provider

    async def generate(self, *, query: str, context: str, messages: list[dict[str, Any]]) -> dict[str, Any]:
        llm_messages = list(messages)
        llm_messages.append(
            {
                "role": "user",
                "content": f"Query: {query}\n\nContext:\n{context}",
            }
        )
        llm_result = await self._llm_provider.generate(llm_messages, model="mock-llm")
        return {
            "answer": llm_result.content,
            "tool_calls": [{"tool": "llm", "provider": "mock", "model": llm_result.model}],
        }


class _AgentCritic:
    def __init__(self, llm_provider: MockLLMProvider) -> None:
        self._llm_provider = llm_provider

    async def critique(self, *, query: str, context: str, answer: str) -> dict[str, Any]:
        _ = await self._llm_provider.generate(
            [
                {
                    "role": "user",
                    "content": f"Critique answer quality for query={query}",
                }
            ],
            model="mock-llm",
        )
        has_issues = not bool(context.strip()) or not bool(answer.strip())
        requires_human_review = any(flag in query.lower() for flag in ("legal", "finance", "medical"))
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


def _langsmith_project_name() -> str:
    return os.getenv("LANGCHAIN_PROJECT", "agentic-ai-platform")


def _langsmith_trace_url(thread_id: str) -> str | None:
    api_key = os.getenv("LANGCHAIN_API_KEY", "").strip()
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
        answer=str(state.get("final_answer", "")),
        sources=_extract_sources(tool_calls),
        tool_calls_made=tool_calls,
        iteration_count=int(state.get("iteration_count", 0)),
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
    settings = get_settings()
    setup_enhanced_logging(service_name=settings.app_name, environment=settings.app_env)

    service_context = build_service_context(settings)

    app = FastAPI(title=settings.app_name, debug=settings.app_debug)
    app.state.settings = settings
    app.state.service_context = service_context
    app.state.runtime = _build_runtime()

    Logger.info(f"Starting {settings.app_name} v{settings.app_version} in {settings.app_env} mode")

    apply_operational_middleware(
        app,
        enable_cors=settings.enable_cors,
        cors_origins=settings.cors_origins,
        enable_logging_middleware=True,
    )

    register_operational_endpoints(
        app,
        context_getter=lambda current_app: current_app.state.service_context,
        settings_snapshot_getter=lambda current_app: current_app.state.settings.masked_debug_config(),
        bootstrap_snapshot_getter=lambda current_app: debug_snapshot(current_app.state.service_context),
    )

    @app.post("/agent/query", response_model=AgentQueryResponse)
    async def agent_query(payload: AgentQueryRequest, stream: bool = Query(default=False)):
        thread_id = payload.thread_id or f"thread-{uuid4().hex}"
        config = _graph_config(thread_id)
        initial_state = _build_initial_state(payload.query, payload.max_iterations)

        if stream:
            async def stream_events():
                last_state: dict[str, Any] = initial_state
                with tracing_context(project_name=_langsmith_project_name(), enabled=True):
                    async for chunk in app.state.runtime.graph.astream(initial_state, config=config, stream_mode="values"):
                        if isinstance(chunk, dict):
                            last_state = chunk
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

        with tracing_context(project_name=_langsmith_project_name(), enabled=True):
            state = await app.state.runtime.graph.ainvoke(initial_state, config=config)
        return _build_query_response(state, thread_id)

    @app.post("/agent/resume", response_model=AgentQueryResponse)
    async def agent_resume(payload: AgentResumeRequest):
        config = _graph_config(payload.thread_id)
        note = {"role": "human", "content": f"Decision: {payload.human_decision}"}
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

        with tracing_context(project_name=_langsmith_project_name(), enabled=True):
            state = await app.state.runtime.graph.ainvoke(Command(update=update), config=config)
        return _build_query_response(state, payload.thread_id)

    @app.get("/agent/trace/{thread_id}", response_model=AgentTraceResponse)
    async def agent_trace(thread_id: str):
        snapshot = await app.state.runtime.graph.aget_state(_graph_config(thread_id))
        if snapshot is None:
            raise HTTPException(status_code=404, detail="Thread not found")
        created_at = getattr(snapshot, "created_at", None)
        created_str = created_at.isoformat() if created_at else None
        return AgentTraceResponse(
            thread_id=thread_id,
            values=dict(snapshot.values or {}),
            next_nodes=list(snapshot.next or ()),
            created_at=created_str,
        )

    @app.post("/retrieval/index")
    async def retrieval_index(payload: RetrievalIndexRequest):
        docs = [document.model_dump(mode="python") for document in payload.documents]
        return await app.state.runtime.retriever.index_documents(docs)

    @app.get("/retrieval/stats")
    async def retrieval_stats():
        return await app.state.runtime.retriever.get_collection_stats()

    @app.post("/mcp/tools/list", response_model=MCPToolsListResponse)
    async def mcp_tools_list():
        tools = await app.state.runtime.mcp_adapter.list_tools()
        return MCPToolsListResponse(tools=[tool.model_dump(mode="json") for tool in tools])

    return app


app = create_app()
