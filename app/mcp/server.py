from __future__ import annotations

import ast
import os
import operator
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Annotated, Any, Protocol

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from app.retrieval import LlamaIndexRetriever


@dataclass(slots=True)
class _LazyRetriever:
    _instance: "LlamaIndexRetriever | None" = None

    async def retrieve(self, query: str, top_k: int | None = None) -> dict[str, Any]:
        if self._instance is None:
            from app.retrieval import LlamaIndexRetriever

            self._instance = LlamaIndexRetriever()
        return await self._instance.retrieve(query, top_k=top_k)


class RetrievalToolPort(Protocol):
    async def retrieve(self, query: str, top_k: int | None = None) -> dict[str, Any]: ...


class SearchDocumentsResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str
    context: str
    sources: list[dict[str, Any]]
    top_k: int


class CalculationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expression: str
    result: float


class CurrentTimeResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    current_time: str
    timezone: str


@dataclass(slots=True)
class MCPServerDependencies:
    retriever: RetrievalToolPort
    time_provider: Callable[[], datetime]


class _ExpressionEvaluator(ast.NodeVisitor):
    _binary_ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.Mod: operator.mod,
        ast.FloorDiv: operator.floordiv,
    }
    _unary_ops = {
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
    }

    def visit_Expression(self, node: ast.Expression) -> float:
        return self.visit(node.body)

    def visit_BinOp(self, node: ast.BinOp) -> float:
        operator_type = type(node.op)
        if operator_type not in self._binary_ops:
            raise ValueError(f"Unsupported operator: {operator_type.__name__}")
        return float(self._binary_ops[operator_type](self.visit(node.left), self.visit(node.right)))

    def visit_UnaryOp(self, node: ast.UnaryOp) -> float:
        operator_type = type(node.op)
        if operator_type not in self._unary_ops:
            raise ValueError(f"Unsupported unary operator: {operator_type.__name__}")
        return float(self._unary_ops[operator_type](self.visit(node.operand)))

    def visit_Constant(self, node: ast.Constant) -> float:
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError("Only numeric literals are allowed")

    def generic_visit(self, node: ast.AST) -> float:
        raise ValueError(f"Unsupported expression node: {type(node).__name__}")


def _evaluate_expression(expression: str) -> float:
    parsed = ast.parse(expression, mode="eval")
    evaluator = _ExpressionEvaluator()
    return evaluator.visit(parsed)


def _default_dependencies() -> MCPServerDependencies:
    return MCPServerDependencies(
        retriever=_LazyRetriever(),
        time_provider=lambda: datetime.now(UTC),
    )


def create_mcp_server(dependencies: MCPServerDependencies | None = None) -> FastMCP:
    resolved_dependencies = dependencies or _default_dependencies()
    # MCP standardizes discovery, contracts, and invocation semantics so teams
    # can add tools without each agent stack inventing its own registration and
    # payload conventions. That matters once multiple services contribute tools.
    server = FastMCP(
        name="agentic-ai-platform-mcp",
        instructions="Expose retrieval and utility tools to agent runtimes over MCP.",
        log_level="INFO",
        host=os.getenv("MCP_HOST", "127.0.0.1"),
        port=int(os.getenv("MCP_PORT", "8000")),
    )

    @server.tool(
        name="search_documents",
        description="Search indexed documents through the LlamaIndex retrieval layer.",
        structured_output=True,
    )
    async def search_documents(
        query: Annotated[str, Field(description="Natural-language query to run against indexed documents.")],
        top_k: Annotated[int, Field(description="Maximum number of source chunks to return.", ge=1, le=20)] = 3,
    ) -> SearchDocumentsResult:
        result = await resolved_dependencies.retriever.retrieve(query, top_k=top_k)
        return SearchDocumentsResult.model_validate(result)

    @server.tool(
        name="calculate",
        description="Safely evaluate a basic arithmetic expression without executing arbitrary code.",
        structured_output=True,
    )
    async def calculate(
        expression: Annotated[
            str,
            Field(description="Arithmetic expression using numbers, parentheses, and + - * / // % ** operators."),
        ]
    ) -> CalculationResult:
        return CalculationResult(expression=expression, result=_evaluate_expression(expression))

    @server.tool(
        name="get_current_time",
        description="Return the current UTC timestamp for time-aware workflows and debugging.",
        structured_output=True,
    )
    async def get_current_time() -> CurrentTimeResult:
        current_time = resolved_dependencies.time_provider().astimezone(UTC)
        return CurrentTimeResult(current_time=current_time.isoformat(), timezone="UTC")

    return server


mcp_server = create_mcp_server()


def main() -> None:
    mcp_server.run("stdio")


if __name__ == "__main__":
    main()