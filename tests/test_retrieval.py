from __future__ import annotations

import asyncio
from uuid import uuid4

from ai_service_kit.providers import MockEmbeddingProvider

from app.retrieval import HybridRetriever, LlamaIndexRetriever, RetrievalConfig


def _build_retriever(collection_name: str) -> LlamaIndexRetriever:
    return LlamaIndexRetriever(
        config=RetrievalConfig(
            collection_name=collection_name,
            chunk_size=128,
            chunk_overlap=16,
            default_top_k=2,
        ),
        embedding_provider=MockEmbeddingProvider({"model": "mock-embed", "dimension": 64, "seed": 11}),
    )


def test_indexer_indexes_manual_documents_and_returns_collection_stats() -> None:
    retriever = _build_retriever(f"manual-{uuid4().hex}")

    stats = asyncio.run(
        retriever.index_documents(
            [
                {
                    "id": "doc-1",
                    "content": "LangGraph provides stateful orchestration for multi-step agents.",
                    "metadata": {"source": "architecture.md"},
                },
                {
                    "id": "doc-2",
                    "content": "LlamaIndex sits between raw vector stores and retrieval workflows.",
                    "metadata": {"source": "retrieval.md"},
                },
            ]
        )
    )
    result = asyncio.run(retriever.retrieve("Which framework provides stateful orchestration?", top_k=2))

    assert stats["document_count"] == 2
    assert stats["chunk_count"] >= 2
    assert stats["chroma_collection_count"] == stats["chunk_count"]
    assert result["top_k"] == 2
    assert result["sources"]
    assert "LangGraph" in result["context"]
    assert result["sources"][0]["metadata"]["source"] in {"architecture.md", "retrieval.md"}


def test_indexer_supports_simple_directory_reader_input(tmp_path) -> None:
    (tmp_path / "alpha.txt").write_text("MCP standardizes tool discovery and invocation contracts.", encoding="utf-8")
    (tmp_path / "beta.txt").write_text("ChromaDB stores vectors while LlamaIndex orchestrates retrieval.", encoding="utf-8")
    retriever = _build_retriever(f"directory-{uuid4().hex}")

    stats = asyncio.run(retriever.index_documents(tmp_path))
    result = asyncio.run(retriever.retrieve("What standardizes tool discovery?", top_k=2))

    assert stats["document_count"] == 2
    assert stats["chunk_count"] >= 2
    assert result["sources"]
    returned_text = [result.get("context", "")]
    returned_text.extend(source.get("content", "") for source in result["sources"])
    returned_text.extend(source.get("text", "") for source in result["sources"])
    assert any("tool discovery" in text.lower() for text in returned_text)


def test_hybrid_retriever_boosts_exact_keyword_matches() -> None:
    retriever = _build_retriever(f"hybrid-{uuid4().hex}")
    asyncio.run(
        retriever.index_documents(
            [
                {
                    "id": "keyword-doc",
                    "content": "MCPX-42 is the internal codename for the tool discovery protocol migration.",
                    "metadata": {"source": "migration.md"},
                },
                {
                    "id": "general-doc",
                    "content": "Protocols can help multiple teams integrate tools consistently across a platform.",
                    "metadata": {"source": "platform.md"},
                },
            ]
        )
    )
    hybrid = HybridRetriever(retriever)

    result = asyncio.run(hybrid.retrieve("MCPX-42 migration", top_k=1))

    assert result["sources"]
    assert result["sources"][0]["metadata"]["source"] == "migration.md"
    assert result["sources"][0]["keyword_score"] > 0.0