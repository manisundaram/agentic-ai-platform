# Retrieval Configuration

The retrieval subsystem is configured via `RetrievalConfig` in `app/retrieval/indexer.py`.

## Default Vector Backend

The default vector backend is Chroma. ChromaDB is used as the vector store for all document embedding storage and similarity search. The collection name defaults to `"default"` and can be overridden via the `default_collection_name` setting.

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `collection_name` | `"default"` | ChromaDB collection name |
| `persist_dir` | `None` | Path for persistent storage; `None` = ephemeral |
| `chunk_size` | `512` | Token size per chunk |
| `chunk_overlap` | `64` | Token overlap between adjacent chunks |
| `default_top_k` | `3` | Number of chunks returned per query |

## Indexing Documents

Documents can be indexed via:
- `POST /retrieval/index` with a JSON body containing a `documents` array.
- Direct call to `retriever.index_documents(path)` with a directory path.

## Embedding Provider

The embedding provider is configured by the `EMBEDDING_PROVIDER` environment variable. Supported values: `openai`, `gemini`, `mock`. The mock provider uses a seeded random vector for testing without API keys.
