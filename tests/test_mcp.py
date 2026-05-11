from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.mcp.client import MCPServerConfig, MCPToolAdapter
from app.mcp.server import MCPServerDependencies, create_mcp_server


class StubRetriever:
    async def retrieve(self, query: str, top_k: int | None = None) -> dict[str, Any]:
        return {
            "query": query,
            "context": f"retrieved context for {query}",
            "sources": [{"source": "stub-doc", "score": 1.0}],
            "top_k": top_k or 3,
        }


def _project_root() -> str:
    return str(Path(__file__).resolve().parents[1])


def test_mcp_server_registers_required_tools_with_schema() -> None:
    async def _run() -> None:
        server = create_mcp_server(
            MCPServerDependencies(
                retriever=StubRetriever(),
                time_provider=lambda: datetime(2026, 1, 1, tzinfo=UTC),
            )
        )
        tools = await server.list_tools()
        names = {tool.name for tool in tools}
        assert names == {"search_documents", "calculate", "get_current_time"}

        tool_map = {tool.name: tool for tool in tools}
        assert "query" in tool_map["search_documents"].inputSchema["properties"]
        assert "top_k" in tool_map["search_documents"].inputSchema["properties"]
        assert "expression" in tool_map["calculate"].inputSchema["properties"]

    asyncio.run(_run())


def test_mcp_server_tool_calls_return_structured_results() -> None:
    async def _run() -> None:
        server = create_mcp_server(
            MCPServerDependencies(
                retriever=StubRetriever(),
                time_provider=lambda: datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
            )
        )

        _, search_result = await server.call_tool("search_documents", {"query": "LangGraph", "top_k": 2})
        _, calc_result = await server.call_tool("calculate", {"expression": "3*(4+1)"})
        _, time_result = await server.call_tool("get_current_time", {})

        assert search_result["top_k"] == 2
        assert "LangGraph" in search_result["context"]
        assert calc_result["result"] == 15.0
        assert time_result["timezone"] == "UTC"
        assert time_result["current_time"].startswith("2026-01-01T12:00:00")

    asyncio.run(_run())


def test_mcp_tool_adapter_lists_and_invokes_stdio_tools() -> None:
    async def _run() -> None:
        adapter = MCPToolAdapter(
            MCPServerConfig(
                command="C:/Users/Mani/anaconda3/python.exe",
                args=["-m", "app.mcp.server"],
                cwd=_project_root(),
                read_timeout_seconds=20.0,
            )
        )
        tools = await adapter.list_tools()
        names = {tool.name for tool in tools}
        assert {"search_documents", "calculate", "get_current_time"}.issubset(names)

        result = await adapter.call_tool("calculate", {"expression": "21/3"})
        assert result["result"] == 7.0

    asyncio.run(_run())


def test_mcp_tool_adapter_converts_tools_to_langchain_structured_tools() -> None:
    async def _run() -> None:
        adapter = MCPToolAdapter(
            MCPServerConfig(
                command="C:/Users/Mani/anaconda3/python.exe",
                args=["-m", "app.mcp.server"],
                cwd=_project_root(),
                read_timeout_seconds=20.0,
            )
        )
        tools = await adapter.get_langchain_tools()
        tool_map = {tool.name: tool for tool in tools}
        assert "calculate" in tool_map

        result = await tool_map["calculate"].ainvoke({"expression": "10+5"})
        assert result["result"] == 15.0

    asyncio.run(_run())