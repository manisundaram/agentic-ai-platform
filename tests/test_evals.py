from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from app.evals import AgentEvaluator, RAGEvaluator, build_eval_dataset, run_full_eval


class StubRetriever:
    async def retrieve(self, query: str, top_k: int | None = None) -> dict[str, Any]:
        return {
            "query": query,
            "context": "LangGraph adds checkpointing and structured orchestration with retrieval grounding.",
            "sources": [
                {
                    "chunk_id": "architecture-langgraph-md",
                    "text": "LangGraph adds checkpointing and structured orchestration.",
                    "metadata": {"source": "architecture/langgraph.md"},
                    "score": 0.9,
                }
            ],
            "top_k": top_k or 3,
        }


class StubGraph:
    async def ainvoke(self, payload: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
        return {
            **payload,
            "context": "retrieved context mentions checkpointing orchestration and grounding",
            "final_answer": "checkpointing orchestration grounding",
            "iteration_count": 1,
            "tool_calls": [{"tool": "llamaindex_retrieval"}],
        }


def test_eval_dataset_has_required_case_distribution() -> None:
    dataset = build_eval_dataset()
    factual = [case for case in dataset if case.category == "factual_retrieval"]
    multi_hop = [case for case in dataset if case.category == "multi_hop_reasoning"]
    edge = [case for case in dataset if case.category == "edge_case"]

    assert len(dataset) == 15
    assert len(factual) == 5
    assert len(multi_hop) == 5
    assert len(edge) == 5


def test_rag_evaluator_returns_metric_scores_with_real_numbers() -> None:
    dataset = build_eval_dataset()
    report = asyncio.run(RAGEvaluator(StubRetriever()).evaluate(dataset))

    assert set(report["metrics"].keys()) == {
        "faithfulness",
        "context_precision",
        "context_recall",
        "answer_relevancy",
    }
    for value in report["metrics"].values():
        assert isinstance(value, float)
        assert 0.0 <= value <= 1.0


def test_agent_evaluator_returns_expected_metric_shape() -> None:
    dataset = build_eval_dataset()
    report = asyncio.run(AgentEvaluator(StubGraph()).evaluate(dataset))

    assert set(report["metrics"].keys()) == {
        "task_completion_rate",
        "tool_call_accuracy",
        "cycle_efficiency",
        "hallucination_rate",
    }
    assert report["case_results"]


def test_run_full_eval_writes_latest_json(tmp_path) -> None:
    output_file = tmp_path / "evals" / "results" / "latest.json"
    summary = asyncio.run(
        run_full_eval(
            retriever=StubRetriever(),
            graph=StubGraph(),
            output_path=output_file,
        )
    )

    assert output_file.exists()
    persisted = json.loads(output_file.read_text(encoding="utf-8"))
    assert persisted["counts"]["total_cases"] == 15
    assert "rag" in persisted
    assert "agent" in persisted
    assert isinstance(summary["overall_score"], float)