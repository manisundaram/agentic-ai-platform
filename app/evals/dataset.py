from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class EvalTestCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    category: Literal["factual_retrieval", "multi_hop_reasoning", "edge_case"]
    query: str = Field(min_length=1)
    expected_answer: str = Field(min_length=1)
    expected_sources: list[str] = Field(default_factory=list)
    requires_tool_use: bool = False


def build_eval_dataset() -> list[EvalTestCase]:
    factual = [
        EvalTestCase(
            id="fact-1",
            category="factual_retrieval",
            query="What does LangGraph add over a simple prompt chain?",
            expected_answer="LangGraph adds typed state, explicit edges, checkpointing, and controlled cycles.",
            expected_sources=["architecture/langgraph.md"],
            requires_tool_use=True,
        ),
        EvalTestCase(
            id="fact-2",
            category="factual_retrieval",
            query="Which vector database backend is configured by default?",
            expected_answer="The default vector backend is Chroma.",
            expected_sources=["config/retrieval.md"],
            requires_tool_use=True,
        ),
        EvalTestCase(
            id="fact-3",
            category="factual_retrieval",
            query="Why do teams adopt MCP instead of custom tool wiring?",
            expected_answer="MCP standardizes tool discovery, schema contracts, and invocation semantics.",
            expected_sources=["architecture/mcp.md"],
            requires_tool_use=True,
        ),
        EvalTestCase(
            id="fact-4",
            category="factual_retrieval",
            query="What is the purpose of LangSmith tracing in this platform?",
            expected_answer="LangSmith tracing captures execution spans and run context for debugging and observability.",
            expected_sources=["ops/langsmith.md"],
            requires_tool_use=False,
        ),
        EvalTestCase(
            id="fact-5",
            category="factual_retrieval",
            query="How does the project support no-credential local testing?",
            expected_answer="It supports mock providers for LLM and embeddings so tests can run without API keys.",
            expected_sources=["ops/mock-providers.md"],
            requires_tool_use=False,
        ),
    ]

    multi_hop = [
        EvalTestCase(
            id="reason-1",
            category="multi_hop_reasoning",
            query="Connect LangGraph checkpointing with human review requirements in one explanation.",
            expected_answer="Checkpointing persists graph state so a human can interrupt, review, and safely resume execution.",
            expected_sources=["architecture/langgraph.md", "architecture/hitl.md"],
            requires_tool_use=True,
        ),
        EvalTestCase(
            id="reason-2",
            category="multi_hop_reasoning",
            query="Compare why LlamaIndex is used with Chroma instead of direct Chroma-only code for this service.",
            expected_answer="LlamaIndex accelerates loaders/chunking/retrieval composition while Chroma provides the vector persistence layer.",
            expected_sources=["architecture/retrieval.md", "architecture/vectorstore.md"],
            requires_tool_use=True,
        ),
        EvalTestCase(
            id="reason-3",
            category="multi_hop_reasoning",
            query="Explain how MCP tools and LangGraph nodes interact during an agent run.",
            expected_answer="LangGraph orchestrates state transitions while MCP tool calls provide standardized external actions inside node execution.",
            expected_sources=["architecture/langgraph.md", "architecture/mcp.md"],
            requires_tool_use=True,
        ),
        EvalTestCase(
            id="reason-4",
            category="multi_hop_reasoning",
            query="Why is evaluation required alongside tracing for production readiness?",
            expected_answer="Tracing explains what happened in a run, while evaluation quantifies quality and regression risk over time.",
            expected_sources=["ops/langsmith.md", "ops/evals.md"],
            requires_tool_use=False,
        ),
        EvalTestCase(
            id="reason-5",
            category="multi_hop_reasoning",
            query="Summarize the tradeoff between deterministic mock providers and real provider validation.",
            expected_answer="Mock providers improve repeatability and speed, but real-provider tests are still needed to validate production behavior.",
            expected_sources=["ops/mock-providers.md", "ops/validation.md"],
            requires_tool_use=False,
        ),
    ]

    edge = [
        EvalTestCase(
            id="edge-1",
            category="edge_case",
            query="[EMPTY_INPUT]",
            expected_answer="The request should be rejected for empty input.",
            expected_sources=[],
            requires_tool_use=False,
        ),
        EvalTestCase(
            id="edge-2",
            category="edge_case",
            query="Tell me about the policy.",
            expected_answer="The answer should acknowledge ambiguity and request clarification.",
            expected_sources=[],
            requires_tool_use=False,
        ),
        EvalTestCase(
            id="edge-3",
            category="edge_case",
            query="Simulate retrieval failure handling behavior.",
            expected_answer="The system should degrade gracefully with a safe fallback response.",
            expected_sources=[],
            requires_tool_use=True,
        ),
        EvalTestCase(
            id="edge-4",
            category="edge_case",
            query="Provide legal compliance advice for a contract dispute.",
            expected_answer="The run should flag human review for high-risk legal guidance.",
            expected_sources=[],
            requires_tool_use=False,
        ),
        EvalTestCase(
            id="edge-5",
            category="edge_case",
            query="Return answer when retrieval context is empty.",
            expected_answer="The system should avoid hallucination and indicate insufficient context.",
            expected_sources=[],
            requires_tool_use=True,
        ),
    ]

    return factual + multi_hop + edge