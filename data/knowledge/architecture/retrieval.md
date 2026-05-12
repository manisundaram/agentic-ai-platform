# Retrieval Architecture

This platform uses LlamaIndex as the retrieval abstraction layer over ChromaDB as the vector store backend.

## Why LlamaIndex Over Direct ChromaDB Calls

LlamaIndex accelerates:
- **Document loading**: `SimpleDirectoryReader` handles multiple file formats without custom parsers.
- **Chunking**: `SentenceSplitter` divides documents into semantically coherent chunks with configurable size and overlap.
- **Retrieval composition**: Retriever interfaces support switching between vector search, keyword search, and hybrid modes without changing application code.
- **Vector store portability**: The `VectorStoreIndex` works with multiple backends. Switching from ChromaDB to Pinecone or Weaviate requires changing only the storage context, not the retrieval logic.

Direct ChromaDB calls remain preferable when a team needs store-specific tuning, bespoke ingestion pipelines, or direct control over every query primitive.

## Retrieval Pipeline

1. Documents are loaded from a directory or list of strings.
2. `SentenceSplitter` splits each document into chunks (default: 512 tokens, 64 overlap).
3. Chunks are embedded using the configured embedding provider (OpenAI, Gemini, or Mock).
4. Embeddings are stored in ChromaDB.
5. At query time, the query is embedded and ChromaDB performs ANN search.
6. The top-k chunks are returned with text, metadata, and similarity scores.

## Metadata

Each retrieved chunk includes:
- `chunk_id`: LlamaIndex node UUID
- `text`: The chunk text
- `metadata`: File path, source document name, and any custom metadata
- `score`: Similarity score from ChromaDB
