from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from .state import AgentState


class RetrievalPort(Protocol):
    async def retrieve(self, query: str) -> dict[str, Any]: ...


class GenerationPort(Protocol):
    async def generate(self, *, query: str, context: str, messages: list[dict[str, Any]]) -> dict[str, Any]: ...


class CritiquePort(Protocol):
    async def critique(self, *, query: str, context: str, answer: str) -> dict[str, Any]: ...


@dataclass(slots=True)
class GraphNodeDependencies:
    retriever: RetrievalPort
    generator: GenerationPort
    critic: CritiquePort


def _normalize_messages(messages: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    return list(messages or [])


def _normalize_tool_calls(tool_calls: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    return list(tool_calls or [])


async def retrieve_context(state: AgentState, dependencies: GraphNodeDependencies) -> AgentState:
    retrieval_result = await dependencies.retriever.retrieve(state["query"])
    tool_calls = _normalize_tool_calls(state.get("tool_calls"))
    retrieval_tool_call = {
        "tool": "llamaindex_retrieval",
        "query": state["query"],
        "top_k": retrieval_result.get("top_k"),
        "sources": retrieval_result.get("sources", []),
    }
    tool_calls.append(retrieval_tool_call)

    return {
        **state,
        "context": retrieval_result.get("context", ""),
        "tool_calls": tool_calls,
    }


async def generate_answer(state: AgentState, dependencies: GraphNodeDependencies) -> AgentState:
    generation_result = await dependencies.generator.generate(
        query=state["query"],
        context=state.get("context", ""),
        messages=_normalize_messages(state.get("messages")),
    )

    messages = _normalize_messages(state.get("messages"))
    assistant_message = {
        "role": "assistant",
        "content": generation_result.get("answer", ""),
    }
    messages.append(assistant_message)

    combined_tool_calls = _normalize_tool_calls(state.get("tool_calls"))
    combined_tool_calls.extend(generation_result.get("tool_calls", []))

    return {
        **state,
        "messages": messages,
        "tool_calls": combined_tool_calls,
        "final_answer": generation_result.get("answer", ""),
    }


async def critique_answer(state: AgentState, dependencies: GraphNodeDependencies) -> AgentState:
    critique_result = await dependencies.critic.critique(
        query=state["query"],
        context=state.get("context", ""),
        answer=state.get("final_answer", ""),
    )
    iteration_count = int(state.get("iteration_count", 0)) + 1

    messages = _normalize_messages(state.get("messages"))
    messages.append(
        {
            "role": "critic",
            "content": critique_result.get("feedback", ""),
        }
    )

    return {
        **state,
        "messages": messages,
        "iteration_count": iteration_count,
        "requires_human_review": bool(critique_result.get("requires_human_review", False)),
        "eval_scores": dict(critique_result.get("eval_scores", {})),
        "critique_has_issues": bool(critique_result.get("has_issues", False)),
        "critique_feedback": critique_result.get("feedback", ""),
    }


def should_retry(state: AgentState) -> str:
    has_issues = bool(state.get("critique_has_issues", False))
    iteration_count = int(state.get("iteration_count", 0))
    max_iterations = int(state.get("max_iterations", 1))
    if has_issues and iteration_count < max_iterations:
        return "retry"
    return "done"


def human_review_check(state: AgentState) -> str:
    if bool(state.get("requires_human_review", False)):
        return "human_review"
    return "done"
