# Validation and Real-Provider Testing

Validation testing confirms that the platform behaves correctly under real provider conditions. It complements mock-based unit tests.

## Tradeoff Between Deterministic Mocks and Real Provider Validation

Mock providers improve repeatability and speed. They return deterministic outputs and require no API keys, making them ideal for CI and local development. However, mock providers cannot detect:
- Prompt formatting errors that cause model refusals.
- Token limit violations for large context windows.
- Rate limiting behavior under load.
- Model-specific output formatting differences.

Real-provider tests are still needed to validate production behavior. These tests typically run in a separate integration suite, triggered before deploys rather than on every commit, to manage API costs and latency.

## Smoke Tests

The quickest validation after setup is a smoke test against the running service:

```bash
# Check health
curl http://localhost:8000/health

# Submit a test query
curl -X POST http://localhost:8000/agent/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is LangGraph?", "thread_id": "smoke-1"}'

# Check retrieval stats
curl http://localhost:8000/retrieval/stats
```

## Validation Checklist

- [ ] LLM provider returns valid completions for the configured model.
- [ ] Embedding provider returns vectors of the expected dimension.
- [ ] ChromaDB collection is non-empty after indexing.
- [ ] Retrieval returns relevant chunks for known queries.
- [ ] HITL interrupt fires for queries containing risky keywords.
- [ ] LangSmith trace URL is returned in query responses.
