from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import chromadb
from ai_service_kit.providers import BaseEmbeddingProvider, MockEmbeddingProvider
from llama_index.core import Document, SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator


logger = logging.getLogger(__name__)


class RetrievalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    collection_name: str = Field(default="default", min_length=1)
    persist_dir: str | None = None
    chunk_size: int = Field(default=512, ge=32)
    chunk_overlap: int = Field(default=64, ge=0)
    default_top_k: int = Field(default=3, ge=1)

    @model_validator(mode="after")
    def validate_chunking(self) -> "RetrievalConfig":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self


@dataclass(slots=True)
class IndexedChunkRecord:
    chunk_id: str
    text: str
    metadata: dict[str, Any]


@dataclass(slots=True)
class SourceDocumentRecord:
    document_id: str
    text: str
    metadata: dict[str, Any]


class AIServiceKitEmbeddingAdapter(BaseEmbedding):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    model_name: str = Field(default="ai-service-kit")
    _provider: BaseEmbeddingProvider = PrivateAttr()
    _provider_model: str | None = PrivateAttr(default=None)

    def __init__(self, *, provider: BaseEmbeddingProvider, provider_model: str | None = None) -> None:
        super().__init__(model_name=provider_model or provider.get_provider_name())
        self._provider = provider
        self._provider_model = provider_model

    def _get_query_embedding(self, query: str) -> list[float]:
        return asyncio.run(self._embed_one(query))

    def _get_text_embedding(self, text: str) -> list[float]:
        return asyncio.run(self._embed_one(text))

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return await self._embed_one(query)

    async def _embed_one(self, value: str) -> list[float]:
        result = await self._provider.embed([value], model=self._provider_model)
        return list(result.embeddings[0])


class LlamaIndexRetriever:
    # LlamaIndex is the right abstraction when you want document loading,
    # chunking, retrieval composition, and vector-store portability in one place.
    # Raw ChromaDB calls remain better when a team needs store-specific tuning,
    # bespoke ingestion pipelines, or direct control over every query primitive.
    def __init__(
        self,
        config: RetrievalConfig | None = None,
        *,
        embedding_provider: BaseEmbeddingProvider | None = None,
        embed_model: BaseEmbedding | None = None,
        chroma_client: chromadb.ClientAPI | None = None,
    ) -> None:
        self.config = config or RetrievalConfig()
        self._embedding_provider = embedding_provider or MockEmbeddingProvider(
            {"model": "mock-embed", "dimension": 128, "seed": 7}
        )
        self._embed_model = embed_model or AIServiceKitEmbeddingAdapter(provider=self._embedding_provider)
        self._chroma_client = chroma_client or self._build_chroma_client()
        self._collection = self._chroma_client.get_or_create_collection(name=self.config.collection_name)
        self._vector_store = ChromaVectorStore(chroma_collection=self._collection)
        self._storage_context = StorageContext.from_defaults(vector_store=self._vector_store)
        self._source_documents: list[SourceDocumentRecord] = []
        self._indexed_chunks: list[IndexedChunkRecord] = []
        self._index: VectorStoreIndex | None = None

    async def index_documents(self, docs: Sequence[Document | dict[str, Any] | str] | str | Path) -> dict[str, Any]:
        logger.info("index_documents started")
        documents = await asyncio.to_thread(self._load_documents, docs)
        logger.info("index_documents loaded_documents=%s", len(documents))
        for position, document in enumerate(documents, start=1):
            doc_id = document.doc_id or f"document-{len(self._source_documents) + position}"
            self._source_documents.append(
                SourceDocumentRecord(
                    document_id=doc_id,
                    text=document.text,
                    metadata=dict(document.metadata or {}),
                )
            )

        await asyncio.to_thread(self._rebuild_index)
        logger.info("index_documents completed document_count=%s chunk_count=%s", len(self._source_documents), len(self._indexed_chunks))
        return await self.get_collection_stats()

    async def retrieve(self, query: str, top_k: int | None = None) -> dict[str, Any]:
        if self._index is None:
            logger.info("retrieve called with empty index query=%s", query)
            return {"query": query, "context": "", "sources": [], "top_k": top_k or self.config.default_top_k}

        resolved_top_k = top_k or self.config.default_top_k
        logger.info("retrieve started query=%s top_k=%s", query, resolved_top_k)
        retriever = self._index.as_retriever(similarity_top_k=resolved_top_k)
        nodes = await asyncio.to_thread(retriever.retrieve, query)
        sources: list[dict[str, Any]] = []
        context_parts: list[str] = []
        for node_with_score in nodes:
            node = node_with_score.node
            text = node.get_content()
            context_parts.append(text)
            sources.append(
                {
                    "chunk_id": node.node_id,
                    "text": text,
                    "metadata": dict(node.metadata or {}),
                    "score": node_with_score.score,
                }
            )

        logger.info("retrieve completed query=%s sources=%s", query, len(sources))
        return {
            "query": query,
            "context": "\n\n".join(context_parts),
            "sources": sources,
            "top_k": resolved_top_k,
        }

    async def get_collection_stats(self) -> dict[str, Any]:
        return {
            "collection_name": self.config.collection_name,
            "document_count": len(self._source_documents),
            "chunk_count": len(self._indexed_chunks),
            "embedding_model": getattr(self._embed_model, "model_name", self._embedding_provider.get_provider_name()),
            "chunk_size": self.config.chunk_size,
            "chunk_overlap": self.config.chunk_overlap,
            "vector_store_backend": "chroma",
            "persist_dir": self.config.persist_dir,
            "chroma_collection_count": self._collection.count(),
        }

    def get_indexed_records(self) -> tuple[IndexedChunkRecord, ...]:
        return tuple(self._indexed_chunks)

    def _build_chroma_client(self) -> chromadb.ClientAPI:
        if self.config.persist_dir:
            return chromadb.PersistentClient(path=self.config.persist_dir)
        return chromadb.EphemeralClient()

    def _load_documents(self, docs: Sequence[Document | dict[str, Any] | str] | str | Path) -> list[Document]:
        if isinstance(docs, (str, Path)):
            return list(SimpleDirectoryReader(input_dir=str(docs)).load_data())

        loaded_documents: list[Document] = []
        for position, item in enumerate(docs, start=1):
            if isinstance(item, Document):
                loaded_documents.append(item)
                continue
            if isinstance(item, str):
                loaded_documents.append(Document(text=item, doc_id=f"manual-{position}"))
                continue
            if isinstance(item, dict):
                content = str(item.get("content", ""))
                metadata = dict(item.get("metadata", {}))
                document_id = item.get("id") or f"manual-{position}"
                loaded_documents.append(Document(text=content, metadata=metadata, doc_id=document_id))
                continue
            raise TypeError(f"Unsupported document type: {type(item)!r}")
        return loaded_documents

    def _rebuild_index(self) -> None:
        logger.info("rebuild_index started collection=%s", self.config.collection_name)
        try:
            self._chroma_client.delete_collection(self.config.collection_name)
        except Exception:
            pass
        self._collection = self._chroma_client.get_or_create_collection(name=self.config.collection_name)
        self._vector_store = ChromaVectorStore(chroma_collection=self._collection)
        self._storage_context = StorageContext.from_defaults(vector_store=self._vector_store)

        splitter = SentenceSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )
        documents = [
            Document(text=record.text, metadata=record.metadata, doc_id=record.document_id)
            for record in self._source_documents
        ]
        self._index = VectorStoreIndex.from_documents(
            documents,
            storage_context=self._storage_context,
            transformations=[splitter],
            embed_model=self._embed_model,
            show_progress=False,
        )
        nodes = splitter.get_nodes_from_documents(documents)
        self._indexed_chunks = [
            IndexedChunkRecord(
                chunk_id=node.node_id,
                text=node.get_content(),
                metadata=dict(node.metadata or {}),
            )
            for node in nodes
        ]
        logger.info("rebuild_index completed chunk_count=%s", len(self._indexed_chunks))
