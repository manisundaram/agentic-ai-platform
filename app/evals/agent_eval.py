from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import mean
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .dataset import EvalTestCase


class AgentEvaluatorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_iterations: int = Field(default=3, ge=1, le=10)
    efficiency_target: int = Field(default=2, ge=1, le=10)


@dataclass(slots=True)
class AgentCaseResult:
    id: str
    query: str
    task_completed: bool
    tool_call_accurate: bool
    cycle_efficient: bool
    hallucinated: bool
    iteration_count: int


class AgentEvaluator:
    def __init__(self, graph: Any, config: AgentEvaluatorConfig | None = None) -> None:
        self._graph = graph
        self.config = config or AgentEvaluatorConfig()

    async def evaluate(self, cases: list[EvalTestCase]) -> dict[str, Any]:
        agent_cases = [case for case in cases if case.category in {"multi_hop_reasoning", "edge_case"}]
        case_results: list[AgentCaseResult] = []

        for case in agent_cases:
            state = await self._graph.ainvoke(
                {
                    "messages": [],
                    "query": case.query,
                    "context": "",
                    "tool_calls": [],
                    "iteration_count": 0,
                    "max_iterations": self.config.max_iterations,
                    "final_answer": "",
                    "requires_human_review": False,
                    "eval_scores": {},
                },
                config={"configurable": {"thread_id": f"eval-{case.id}"}},
            )

            final_answer = str(state.get("final_answer", ""))
            context = str(state.get("context", ""))
            tool_calls = list(state.get("tool_calls", []))
            iteration_count = int(state.get("iteration_count", 0))
            task_completed = self._task_completed(case, final_answer)
            tool_call_accurate = self._tool_call_accurate(case.requires_tool_use, tool_calls)
            cycle_efficient = iteration_count <= self.config.efficiency_target
            hallucinated = self._hallucinated(final_answer, context)

            case_results.append(
                AgentCaseResult(
                    id=case.id,
                    query=case.query,
                    task_completed=task_completed,
                    tool_call_accurate=tool_call_accurate,
                    cycle_efficient=cycle_efficient,
                    hallucinated=hallucinated,
                    iteration_count=iteration_count,
                )
            )

        total = len(case_results) or 1
        task_completion_rate = sum(1 for result in case_results if result.task_completed) / total
        tool_call_accuracy = sum(1 for result in case_results if result.tool_call_accurate) / total
        cycle_efficiency = sum(1 for result in case_results if result.cycle_efficient) / total
        hallucination_rate = sum(1 for result in case_results if result.hallucinated) / total

        return {
            "metrics": {
                "task_completion_rate": task_completion_rate,
                "tool_call_accuracy": tool_call_accuracy,
                "cycle_efficiency": cycle_efficiency,
                "hallucination_rate": hallucination_rate,
            },
            "case_results": [asdict(result) for result in case_results],
        }

    def _task_completed(self, case: EvalTestCase, answer: str) -> bool:
        expected_tokens = self._token_set(case.expected_answer)
        answer_tokens = self._token_set(answer)
        if not expected_tokens:
            return bool(answer.strip())
        overlap = len(expected_tokens & answer_tokens)
        return overlap / len(expected_tokens) >= 0.2

    def _tool_call_accurate(self, requires_tool_use: bool, tool_calls: list[dict[str, Any]]) -> bool:
        made_call = bool(tool_calls)
        if requires_tool_use:
            return made_call
        return True

    def _hallucinated(self, answer: str, context: str) -> bool:
        answer_tokens = self._token_set(answer)
        if not answer_tokens:
            return True
        context_tokens = self._token_set(context)
        grounded = len(answer_tokens & context_tokens)
        return (grounded / len(answer_tokens)) < 0.2

    def _token_set(self, text: str) -> set[str]:
        normalized = [token.strip(".,!?;:\"'()[]{}") for token in text.lower().split()]
        return {token for token in normalized if token}