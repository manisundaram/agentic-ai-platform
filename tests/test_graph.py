from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from app.graph import get_graph, graph
from app.graph.nodes import GraphNodeDependencies, human_review_check, should_retry


@dataclass(slots=True)
class StubRetriever:
    calls: list[str] = field(default_factory=list)

    async def retrieve(self, query: str) -> dict[str, Any]:
        self.calls.append(query)
        return {
            "context": f"retrieved context for {query}",
            "sources": [{"id": "doc-1"}],
            "top_k": 3,
        }


@dataclass(slots=True)
class StubGenerator:
    calls: int = 0

    async def generate(self, *, query: str, context: str, messages: list[dict[str, Any]]) -> dict[str, Any]:
        self.calls += 1
        return {
            "answer": f"answer-{self.calls} using {context}",
            "tool_calls": [{"tool": "generator", "call": self.calls}],
        }


@dataclass(slots=True)
class SequenceCritic:
    responses: list[dict[str, Any]]
    calls: int = 0

    async def critique(self, *, query: str, context: str, answer: str) -> dict[str, Any]:
        response = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        return response


def _run_graph(dependencies: GraphNodeDependencies, state: dict[str, Any], thread_id: str) -> dict[str, Any]:
    compiled_graph = get_graph(dependencies)
    return asyncio.run(compiled_graph.ainvoke(state, config={"configurable": {"thread_id": thread_id}}))


def test_should_retry_returns_retry_when_issues_remain_with_budget() -> None:
    state = {
        "critique_has_issues": True,
        "iteration_count": 1,
        "max_iterations": 3,
    }

    assert should_retry(state) == "retry"


def test_human_review_check_routes_to_interrupt_when_requested() -> None:
    assert human_review_check({"requires_human_review": True}) == "human_review"
    assert human_review_check({"requires_human_review": False}) == "done"


def test_graph_completes_without_retry_when_critique_passes() -> None:
    dependencies = GraphNodeDependencies(
        retriever=StubRetriever(),
        generator=StubGenerator(),
        critic=SequenceCritic(
            responses=[
                {
                    "has_issues": False,
                    "feedback": "grounded answer",
                    "eval_scores": {"groundedness": 0.98},
                    "requires_human_review": False,
                }
            ]
        ),
    )
    result = _run_graph(
        dependencies,
        {
            "messages": [],
            "query": "What is LangGraph?",
            "context": "",
            "tool_calls": [],
            "iteration_count": 0,
            "max_iterations": 2,
            "final_answer": "",
            "requires_human_review": False,
            "eval_scores": {},
        },
        thread_id="graph-pass",
    )

    assert result["context"] == "retrieved context for What is LangGraph?"
    assert result["final_answer"].startswith("answer-1")
    assert result["iteration_count"] == 1
    assert result["critique_has_issues"] is False
    assert result["requires_human_review"] is False
    assert len(result["tool_calls"]) == 2


def test_graph_retries_before_finishing_when_critique_flags_issues() -> None:
    retriever = StubRetriever()
    generator = StubGenerator()
    critic = SequenceCritic(
        responses=[
            {
                "has_issues": True,
                "feedback": "missing source grounding",
                "eval_scores": {"groundedness": 0.42},
                "requires_human_review": False,
            },
            {
                "has_issues": False,
                "feedback": "second draft is grounded",
                "eval_scores": {"groundedness": 0.95},
                "requires_human_review": False,
            },
        ]
    )
    result = _run_graph(
        GraphNodeDependencies(retriever=retriever, generator=generator, critic=critic),
        {
            "messages": [],
            "query": "Explain retry logic",
            "context": "",
            "tool_calls": [],
            "iteration_count": 0,
            "max_iterations": 3,
            "final_answer": "",
            "requires_human_review": False,
            "eval_scores": {},
        },
        thread_id="graph-retry",
    )

    assert retriever.calls == ["Explain retry logic", "Explain retry logic"]
    assert generator.calls == 2
    assert critic.calls == 2
    assert result["iteration_count"] == 2
    assert result["final_answer"].startswith("answer-2")
    assert result["critique_feedback"] == "second draft is grounded"


def test_graph_interrupts_before_human_review_node() -> None:
    result = _run_graph(
        GraphNodeDependencies(
            retriever=StubRetriever(),
            generator=StubGenerator(),
            critic=SequenceCritic(
                responses=[
                    {
                        "has_issues": False,
                        "feedback": "needs compliance review",
                        "eval_scores": {"groundedness": 0.91},
                        "requires_human_review": True,
                    }
                ]
            ),
        ),
        {
            "messages": [],
            "query": "Return regulated advice",
            "context": "",
            "tool_calls": [],
            "iteration_count": 0,
            "max_iterations": 1,
            "final_answer": "",
            "requires_human_review": False,
            "eval_scores": {},
        },
        thread_id="graph-hitl",
    )

    assert result["requires_human_review"] is True
    assert result["critique_feedback"] == "needs compliance review"


def test_default_graph_export_is_compiled() -> None:
    assert graph is not None