from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

from fastapi.testclient import TestClient
from langgraph.types import Command

from app.main import AppRuntime, create_app


class StubGraph:
    def __init__(self) -> None:
        self._states: dict[str, dict[str, Any]] = {}

    async def ainvoke(self, payload: Any, config: dict[str, Any]) -> dict[str, Any]:
        thread_id = config["configurable"]["thread_id"]
        if isinstance(payload, Command):
            current = dict(self._states.get(thread_id, {}))
            if payload.update:
                current.update(dict(payload.update))
            self._states[thread_id] = current
            return current

        state = dict(payload)
        state.update(
            {
                "context": "stub context",
                "final_answer": "stub answer",
                "iteration_count": 1,
                "tool_calls": [
                    {
                        "tool": "llamaindex_retrieval",
                        "sources": [{"id": "doc-1", "score": 0.99}],
                    }
                ],
            }
        )
        self._states[thread_id] = state
        return state

    async def astream(self, payload: Any, config: dict[str, Any], stream_mode: str = "values"):
        final_state = await self.ainvoke(payload, config)
        yield final_state

    async def aget_state(self, config: dict[str, Any]):
        thread_id = config["configurable"]["thread_id"]
        if thread_id not in self._states:
            return None
        return SimpleNamespace(values=self._states[thread_id], next=(), created_at=datetime.now(UTC))


@dataclass(slots=True)
class StubRetriever:
    indexed_documents: list[dict[str, Any]] = field(default_factory=list)

    async def index_documents(self, docs: list[dict[str, Any]]) -> dict[str, Any]:
        self.indexed_documents.extend(docs)
        return {
            "collection_name": "default",
            "document_count": len(self.indexed_documents),
            "chunk_count": len(self.indexed_documents),
            "embedding_model": "mock-embed",
            "chunk_size": 512,
            "chunk_overlap": 64,
            "vector_store_backend": "chroma",
            "persist_dir": None,
            "chroma_collection_count": len(self.indexed_documents),
        }

    async def get_collection_stats(self) -> dict[str, Any]:
        return {
            "collection_name": "default",
            "document_count": len(self.indexed_documents),
            "chunk_count": len(self.indexed_documents),
            "embedding_model": "mock-embed",
            "chunk_size": 512,
            "chunk_overlap": 64,
            "vector_store_backend": "chroma",
            "persist_dir": None,
            "chroma_collection_count": len(self.indexed_documents),
        }


class StubMCPTool:
    def __init__(self, name: str) -> None:
        self._name = name

    def model_dump(self, mode: str = "json") -> dict[str, Any]:
        return {
            "name": self._name,
            "description": "stub tool",
            "inputSchema": {"type": "object", "properties": {}},
        }


class StubMCPAdapter:
    async def list_tools(self) -> list[StubMCPTool]:
        return [StubMCPTool("search_documents"), StubMCPTool("calculate")]


def _build_client() -> TestClient:
    app = create_app()
    app.state.runtime = AppRuntime(
        retriever=StubRetriever(),
        graph=StubGraph(),
        mcp_adapter=StubMCPAdapter(),
    )
    return TestClient(app)


def test_agent_query_endpoint_returns_expected_payload() -> None:
    client = _build_client()
    response = client.post("/agent/query", json={"query": "What is LangGraph?", "thread_id": "thread-1"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == "stub answer"
    assert payload["iteration_count"] == 1
    assert payload["sources"][0]["id"] == "doc-1"


def test_agent_resume_modify_requires_modified_answer() -> None:
    client = _build_client()
    client.post("/agent/query", json={"query": "seed", "thread_id": "thread-2"})

    response = client.post(
        "/agent/resume",
        json={"thread_id": "thread-2", "human_decision": "modify"},
    )

    assert response.status_code == 400
    assert "modified_answer" in response.json()["detail"]


def test_agent_trace_endpoint_returns_state_snapshot() -> None:
    client = _build_client()
    client.post("/agent/query", json={"query": "trace me", "thread_id": "thread-3"})

    response = client.get("/agent/trace/thread-3")

    assert response.status_code == 200
    payload = response.json()
    assert payload["thread_id"] == "thread-3"
    assert payload["values"]["final_answer"] == "stub answer"


def test_retrieval_endpoints_index_and_stats() -> None:
    client = _build_client()
    index_response = client.post(
        "/retrieval/index",
        json={
            "documents": [
                {
                    "id": "doc-1",
                    "content": "LangGraph supports cycles",
                    "metadata": {"source": "notes.md"},
                }
            ]
        },
    )

    stats_response = client.get("/retrieval/stats")

    assert index_response.status_code == 200
    assert stats_response.status_code == 200
    assert stats_response.json()["document_count"] == 1


def test_mcp_tools_list_endpoint_returns_tool_schemas() -> None:
    client = _build_client()
    response = client.post("/mcp/tools/list")

    assert response.status_code == 200
    payload = response.json()
    assert len(payload["tools"]) == 2
    assert payload["tools"][0]["name"] == "search_documents"