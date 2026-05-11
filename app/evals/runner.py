from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any

from .agent_eval import AgentEvaluator
from .dataset import build_eval_dataset
from .rag_eval import RAGEvaluator


async def run_full_eval(
    *,
    retriever: Any,
    graph: Any,
    output_path: str | Path = "evals/results/latest.json",
) -> dict[str, Any]:
    dataset = build_eval_dataset()
    rag_report = await RAGEvaluator(retriever).evaluate(dataset)
    agent_report = await AgentEvaluator(graph).evaluate(dataset)

    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "counts": {
            "total_cases": len(dataset),
            "factual_retrieval": sum(1 for case in dataset if case.category == "factual_retrieval"),
            "multi_hop_reasoning": sum(1 for case in dataset if case.category == "multi_hop_reasoning"),
            "edge_case": sum(1 for case in dataset if case.category == "edge_case"),
        },
        "rag": rag_report,
        "agent": agent_report,
    }

    combined_scores = [
        rag_report.get("overall_score", 0.0),
        mean(agent_report.get("metrics", {}).values()) if agent_report.get("metrics") else 0.0,
    ]
    summary["overall_score"] = mean(combined_scores)

    target_path = Path(output_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary