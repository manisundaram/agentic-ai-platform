from __future__ import annotations

from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    """Typed graph state keeps every node aligned on the same contract.

    LangGraph will happily pass arbitrary dictionaries between nodes, which is
    convenient for prototypes but brittle in a production workflow. A TypedDict
    makes state shape explicit, catches drift earlier in tests and type-checking,
    and reduces the chance that a retry loop or human-in-the-loop boundary drops
    fields the next node depends on.
    """

    messages: list[dict[str, Any]]
    query: str
    context: str
    tool_calls: list[dict[str, Any]]
    iteration_count: int
    max_iterations: int
    final_answer: str
    requires_human_review: bool
    eval_scores: dict[str, float]
    critique_has_issues: bool
    critique_feedback: str