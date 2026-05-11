from __future__ import annotations

from functools import partial
from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from .nodes import GraphNodeDependencies, critique_answer, generate_answer, human_review_check, retrieve_context, should_retry
from .state import AgentState


class _UnavailableRetriever:
    async def retrieve(self, query: str) -> dict[str, Any]:
        raise RuntimeError("No retriever dependency was configured for the default graph.")


class _UnavailableGenerator:
    async def generate(self, *, query: str, context: str, messages: list[dict[str, Any]]) -> dict[str, Any]:
        raise RuntimeError("No generator dependency was configured for the default graph.")


class _UnavailableCritic:
    async def critique(self, *, query: str, context: str, answer: str) -> dict[str, Any]:
        raise RuntimeError("No critic dependency was configured for the default graph.")


def _default_dependencies() -> GraphNodeDependencies:
    return GraphNodeDependencies(
        retriever=_UnavailableRetriever(),
        generator=_UnavailableGenerator(),
        critic=_UnavailableCritic(),
    )


def build_graph(dependencies: GraphNodeDependencies):
    workflow = StateGraph(AgentState)

    workflow.add_node("retrieve_context", partial(retrieve_context, dependencies=dependencies))
    workflow.add_node("generate_answer", partial(generate_answer, dependencies=dependencies))
    workflow.add_node("critique_answer", partial(critique_answer, dependencies=dependencies))
    workflow.add_node("human_review", lambda state: state)

    # Retrieval runs first because the downstream answer node is only reliable
    # when it receives grounded context instead of improvising from the prompt.
    workflow.add_edge(START, "retrieve_context")

    # Generation is a separate step so the graph can revisit it after critique
    # without repeating retrieval work unless the retry path explicitly chooses to.
    workflow.add_edge("retrieve_context", "generate_answer")

    # Critique sits after generation because the self-correction loop depends on
    # evaluating the produced answer, not the raw retrieval output.
    workflow.add_edge("generate_answer", "critique_answer")

    workflow.add_conditional_edges(
        "critique_answer",
        should_retry,
        {
            "retry": "retrieve_context",
            "done": "human_review",
        },
    )
    workflow.add_conditional_edges(
        "human_review",
        human_review_check,
        {
            "human_review": END,
            "done": END,
        },
    )

    checkpointer = MemorySaver()
    return workflow.compile(checkpointer=checkpointer, interrupt_before=["human_review"])


graph = build_graph(_default_dependencies())


def get_graph(dependencies: GraphNodeDependencies | None = None):
    return build_graph(dependencies or _default_dependencies())