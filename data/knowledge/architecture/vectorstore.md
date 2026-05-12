# Vector Store: ChromaDB

ChromaDB is the default vector database backend for this platform. It provides ANN (approximate nearest neighbor) search over document embeddings.

## Default Configuration

The default vector backend is Chroma. The collection name, persistence directory, chunk size, and chunk overlap are all configurable via `RetrievalConfig`:

```python
class RetrievalConfig(BaseModel):
    collection_name: str = "default"
    persist_dir: str | None = None   # None = in-memory EphemeralClient
    chunk_size: int = 512
    chunk_overlap: int = 64
    default_top_k: int = 3
```

When `persist_dir` is set, a `PersistentClient` is used and the collection survives restarts. Without it, an `EphemeralClient` is used for ephemeral in-process storage.

## Why ChromaDB

- Embeds well with LlamaIndex's `ChromaVectorStore` adapter.
- Supports both ephemeral (test/dev) and persistent (production) modes.
- No external service required for local development.
- Native Python client with simple collection API.

## ChromaDB vs Other Stores

ChromaDB is ideal when simplicity and local development speed matter. For production at scale, teams often migrate to Pinecone (managed, high throughput) or Weaviate (hybrid BM25 + vector). LlamaIndex's storage context abstraction makes this migration straightforward.

## Collection Stats

`GET /retrieval/stats` returns current collection state:
```json
{
  "collection_name": "default",
  "document_count": 15,
  "chunk_count": 42,
  "vector_store_backend": "chroma",
  "chroma_collection_count": 42
}
```
