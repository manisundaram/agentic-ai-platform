from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from statistics import mean
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .dataset import EvalTestCase


class RAGEvaluatorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    top_k: int = Field(default=3, ge=1, le=20)
    failure_threshold: float = Field(default=0.6, ge=0.0, le=1.0)


@dataclass(slots=True)
class RAGCaseResult:
    id: str
    query: str
    faithfulness: float
    context_precision: float
    context_recall: float
    answer_relevancy: float
    overall: float
    failing: bool


class RAGEvaluator:
    def __init__(self, retriever: Any, config: RAGEvaluatorConfig | None = None) -> None:
        self._retriever = retriever
        self.config = config or RAGEvaluatorConfig()

    async def evaluate(self, cases: list[EvalTestCase]) -> dict[str, Any]:
        retrieval_cases = [case for case in cases if case.category == "factual_retrieval"]
        results: list[RAGCaseResult] = []

        for case in retrieval_cases:
            retrieved = await self._retriever.retrieve(case.query, top_k=self.config.top_k)
            answer_text = str(retrieved.get("context", "")).strip()
            contexts = [str(source.get("text", "")) for source in retrieved.get("sources", [])]

            faithfulness = self._faithfulness(answer_text, contexts)
            context_precision = self._context_precision(case.expected_sources, retrieved.get("sources", []))
            context_recall = self._context_recall(case.expected_sources, retrieved.get("sources", []))
            answer_relevancy = self._answer_relevancy(case.query, answer_text)
            overall = mean([faithfulness, context_precision, context_recall, answer_relevancy])
            results.append(
                RAGCaseResult(
                    id=case.id,
                    query=case.query,
                    faithfulness=faithfulness,
                    context_precision=context_precision,
                    context_recall=context_recall,
                    answer_relevancy=answer_relevancy,
                    overall=overall,
                    failing=overall < self.config.failure_threshold,
                )
            )

        metric_groups = {
            "faithfulness": [result.faithfulness for result in results],
            "context_precision": [result.context_precision for result in results],
            "context_recall": [result.context_recall for result in results],
            "answer_relevancy": [result.answer_relevancy for result in results],
        }
        metric_scores = {name: (mean(values) if values else 0.0) for name, values in metric_groups.items()}
        overall_score = mean(metric_scores.values()) if metric_scores else 0.0
        failing_cases = [asdict(result) for result in results if result.failing]

        return {
            "metrics": metric_scores,
            "overall_score": overall_score,
            "failing_cases": failing_cases,
            "case_results": [asdict(result) for result in results],
        }

    def _faithfulness(self, answer: str, contexts: list[str]) -> float:
        if not answer:
            return 0.0
        context_tokens = self._token_set(" ".join(contexts))
        answer_tokens = self._token_set(answer)
        if not answer_tokens:
            return 0.0
        grounded = len(answer_tokens & context_tokens)
        return grounded / len(answer_tokens)

    def _context_precision(self, expected_sources: list[str], retrieved_sources: list[dict[str, Any]]) -> float:
        if not retrieved_sources:
            return 0.0
        expected_tokens = self._token_set(" ".join(expected_sources))
        matched = 0
        for source in retrieved_sources:
            source_tokens = self._token_set(" ".join([str(source.get("chunk_id", "")), str(source.get("metadata", ""))]))
            if expected_tokens and source_tokens and (expected_tokens & source_tokens):
                matched += 1
        return matched / len(retrieved_sources)

    def _context_recall(self, expected_sources: list[str], retrieved_sources: list[dict[str, Any]]) -> float:
        if not expected_sources:
            return 1.0
        expected_tokens = self._token_set(" ".join(expected_sources))
        if not expected_tokens:
            return 0.0
        retrieved_tokens = self._token_set(
            " ".join([str(source.get("chunk_id", "")) + " " + str(source.get("metadata", "")) for source in retrieved_sources])
        )
        return len(expected_tokens & retrieved_tokens) / len(expected_tokens)

    def _answer_relevancy(self, query: str, answer: str) -> float:
        query_tokens = self._token_set(query)
        answer_tokens = self._token_set(answer)
        if not query_tokens or not answer_tokens:
            return 0.0
        return len(query_tokens & answer_tokens) / len(query_tokens)

    def _token_set(self, text: str) -> set[str]:
        normalized = [token.strip(".,!?;:\"'()[]{}") for token in text.lower().split()]
        return {token for token in normalized if token}