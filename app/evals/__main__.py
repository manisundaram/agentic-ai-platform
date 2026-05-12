"""Eval runner entry point.

Usage:
    python -m app.evals

Loads knowledge documents from data/knowledge/, indexes them into a fresh retriever with real
embeddings (if credentials available), runs the full eval suite (RAG + agent), and writes
results to evals/results/latest.json.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

# Ensure repo root is on sys.path when running as __main__
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dotenv import load_dotenv

load_dotenv(_REPO_ROOT / ".env", override=False)

from ai_service_kit.providers import LLMProviderFactory, MockLLMProvider, ProviderFactory
from app.graph.builder import build_graph
from app.graph.nodes import GraphNodeDependencies
from app.retrieval import LlamaIndexRetriever, RetrievalConfig

from .runner import run_full_eval

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

_KNOWLEDGE_DIR = _REPO_ROOT / "data" / "knowledge"
_RESULTS_PATH = _REPO_ROOT / "evals" / "results" / "latest.json"


def _build_embedding_provider():
    """Return a real embedding provider if credentials are available, else mock."""
    try:
        from app.config import get_settings
        settings = get_settings()
        provider_name = settings.resolved_provider("embedding")
        config = settings.embedding_provider_config()
        provider = ProviderFactory().create_provider(provider_name, config)
        logger.info("Eval using real embedding provider=%s", provider_name)
        return provider
    except Exception as exc:
        logger.warning("Real embedding provider unavailable (%s), falling back to mock", exc)
        from ai_service_kit.providers import MockEmbeddingProvider
        return MockEmbeddingProvider({"model": "mock-embed", "dimension": 128, "seed": 7})


def _build_llm_provider():
    """Return a real LLM provider if credentials are available, else mock."""
    try:
        from app.config import get_settings
        settings = get_settings()
        provider_name = settings.resolved_provider("llm")
        config = settings.llm_provider_config()
        provider = LLMProviderFactory().create_provider(provider_name, config)
        logger.info("Eval using real LLM provider=%s", provider_name)
        return provider
    except Exception as exc:
        logger.warning("Real LLM provider unavailable (%s), falling back to mock", exc)
        return MockLLMProvider({"model": "mock-llm", "seed": 17})


class _EvalGenerator:
    """Minimal generator used during evals — wraps the LLM provider."""

    def __init__(self, llm_provider) -> None:
        self._llm = llm_provider

    @staticmethod
    def _normalize_roles(messages):
        allowed = {"system", "assistant", "user", "function", "tool", "developer"}
        return [{**m, "role": m.get("role", "user") if m.get("role") in allowed else "system"} for m in messages]

    async def generate(self, *, query: str, context: str, messages: list) -> dict:
        msgs = self._normalize_roles(list(messages))
        msgs.append({"role": "user", "content": f"Query: {query}\n\nContext:\n{context}"})
        result = await self._llm.generate(msgs, model=None)
        return {"answer": result.content, "tool_calls": [{"tool": "llm"}]}


class _EvalCritic:
    async def critique(self, *, query: str, context: str, answer: str) -> dict:
        has_issues = not bool(context.strip()) or not bool(answer.strip())
        risky = ("vpn", "mdm", "enroll", "admin", "password reset", "disable", "credential")
        requires_human_review = any(r in query.lower() for r in ("legal", "finance", "medical", *risky))
        return {
            "has_issues": has_issues,
            "feedback": "Needs grounding" if has_issues else "Answer appears grounded",
            "eval_scores": {"groundedness": 0.35 if has_issues else 0.9},
            "requires_human_review": requires_human_review,
        }


async def main() -> None:
    logger.info("Initialising retriever with knowledge directory: %s", _KNOWLEDGE_DIR)
    if not _KNOWLEDGE_DIR.exists():
        logger.error("Knowledge directory not found: %s", _KNOWLEDGE_DIR)
        sys.exit(1)

    embedding_provider = _build_embedding_provider()
    retriever = LlamaIndexRetriever(
        RetrievalConfig(collection_name="eval-knowledge", default_top_k=3),
        embedding_provider=embedding_provider,
    )

    logger.info("Indexing knowledge documents...")
    doc_files = list(_KNOWLEDGE_DIR.rglob("*.md"))
    if not doc_files:
        logger.error("No .md files found under %s", _KNOWLEDGE_DIR)
        sys.exit(1)

    docs = []
    for path in sorted(doc_files):
        relative = path.relative_to(_KNOWLEDGE_DIR).as_posix()
        docs.append({
            "id": relative,
            "content": path.read_text(encoding="utf-8"),
            "metadata": {"file_path": relative, "source": relative},
        })
    logger.info("Found %d knowledge files", len(docs))
    stats = await retriever.index_documents(docs)
    logger.info("Index complete: %s documents, %s chunks", stats["document_count"], stats["chunk_count"])

    if stats["chunk_count"] == 0:
        logger.error("No chunks indexed — check that data/knowledge/ contains .md files")
        sys.exit(1)

    llm_provider = _build_llm_provider()
    graph = build_graph(
        GraphNodeDependencies(
            retriever=retriever,
            generator=_EvalGenerator(llm_provider),
            critic=_EvalCritic(),
        )
    )

    logger.info("Running eval suite...")
    summary = await run_full_eval(
        retriever=retriever,
        graph=graph,
        output_path=_RESULTS_PATH,
    )

    rag = summary.get("rag", {}).get("metrics", {})
    agent = summary.get("agent", {}).get("metrics", {})
    logger.info(
        "RAG scores — faithfulness=%.2f context_precision=%.2f context_recall=%.2f answer_relevancy=%.2f",
        rag.get("faithfulness", 0),
        rag.get("context_precision", 0),
        rag.get("context_recall", 0),
        rag.get("answer_relevancy", 0),
    )
    logger.info(
        "Agent scores — task_completion=%.2f tool_accuracy=%.2f cycle_efficiency=%.2f hallucination=%.2f",
        agent.get("task_completion_rate", 0),
        agent.get("tool_call_accuracy", 0),
        agent.get("cycle_efficiency", 0),
        agent.get("hallucination_rate", 0),
    )
    logger.info("Results written to %s", _RESULTS_PATH)


if __name__ == "__main__":
    asyncio.run(main())
