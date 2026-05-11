from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .indexer import IndexedChunkRecord, LlamaIndexRetriever


class HybridSearchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    alpha: float = Field(default=0.7, ge=0.0, le=1.0)
    keyword_candidate_multiplier: int = Field(default=3, ge=1)
    bm25_k1: float = Field(default=1.5, gt=0.0)
    bm25_b: float = Field(default=0.75, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_weighting(self) -> "HybridSearchConfig":
        return self


@dataclass(slots=True)
class HybridSearchHit:
    chunk_id: str
    text: str
    metadata: dict[str, Any]
    vector_score: float
    keyword_score: float
    fused_score: float


class HybridRetriever:
    # Hybrid search outperforms pure vector search when exact identifiers,
    # product names, acronyms, or compliance terms matter because BM25 keeps
    # keyword precision while vector search preserves semantic recall.
    def __init__(self, indexer: LlamaIndexRetriever, config: HybridSearchConfig | None = None) -> None:
        self._indexer = indexer
        self.config = config or HybridSearchConfig()

    async def retrieve(self, query: str, top_k: int = 5) -> dict[str, Any]:
        vector_result = await self._indexer.retrieve(query, top_k=top_k)
        vector_scores = {
            source["chunk_id"]: float(source.get("score") or 0.0)
            for source in vector_result["sources"]
        }
        corpus = self._indexer.get_indexed_records()
        keyword_scores = self._bm25_scores(query, corpus)

        candidate_ids = set(vector_scores)
        candidate_ids.update(
            chunk_id
            for chunk_id, _ in sorted(keyword_scores.items(), key=lambda item: item[1], reverse=True)[: top_k * self.config.keyword_candidate_multiplier]
        )

        if not candidate_ids:
            return {"query": query, "context": "", "sources": [], "top_k": top_k}

        max_vector = max(vector_scores.values(), default=1.0) or 1.0
        max_keyword = max(keyword_scores.values(), default=1.0) or 1.0
        records_by_id = {record.chunk_id: record for record in corpus}
        hits: list[HybridSearchHit] = []
        for chunk_id in candidate_ids:
            record = records_by_id.get(chunk_id)
            if record is None:
                continue
            normalized_vector = vector_scores.get(chunk_id, 0.0) / max_vector
            normalized_keyword = keyword_scores.get(chunk_id, 0.0) / max_keyword
            fused_score = (self.config.alpha * normalized_vector) + ((1.0 - self.config.alpha) * normalized_keyword)
            hits.append(
                HybridSearchHit(
                    chunk_id=chunk_id,
                    text=record.text,
                    metadata=record.metadata,
                    vector_score=vector_scores.get(chunk_id, 0.0),
                    keyword_score=keyword_scores.get(chunk_id, 0.0),
                    fused_score=fused_score,
                )
            )

        ranked_hits = sorted(hits, key=lambda item: item.fused_score, reverse=True)[:top_k]
        return {
            "query": query,
            "context": "\n\n".join(hit.text for hit in ranked_hits),
            "sources": [asdict(hit) for hit in ranked_hits],
            "top_k": top_k,
        }

    def _bm25_scores(self, query: str, corpus: tuple[IndexedChunkRecord, ...]) -> dict[str, float]:
        if not corpus:
            return {}

        query_terms = self._tokenize(query)
        if not query_terms:
            return {}

        tokenized_docs = {record.chunk_id: self._tokenize(record.text) for record in corpus}
        avg_doc_len = sum(len(tokens) for tokens in tokenized_docs.values()) / len(tokenized_docs)
        document_frequency: dict[str, int] = {}
        for tokens in tokenized_docs.values():
            for token in set(tokens):
                document_frequency[token] = document_frequency.get(token, 0) + 1

        scores: dict[str, float] = {}
        document_count = len(corpus)
        for record in corpus:
            tokens = tokenized_docs[record.chunk_id]
            doc_len = max(len(tokens), 1)
            term_counts: dict[str, int] = {}
            for token in tokens:
                term_counts[token] = term_counts.get(token, 0) + 1

            total = 0.0
            for term in query_terms:
                frequency = term_counts.get(term, 0)
                if frequency == 0:
                    continue
                df = document_frequency.get(term, 0)
                idf = math.log(1.0 + ((document_count - df + 0.5) / (df + 0.5)))
                numerator = frequency * (self.config.bm25_k1 + 1.0)
                denominator = frequency + self.config.bm25_k1 * (
                    1.0 - self.config.bm25_b + self.config.bm25_b * (doc_len / avg_doc_len)
                )
                total += idf * (numerator / denominator)

            if total > 0.0:
                scores[record.chunk_id] = total
        return scores

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"[A-Za-z0-9_]+", text.lower())