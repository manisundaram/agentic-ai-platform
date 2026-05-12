# Mock Providers for Local Testing

This platform supports mock LLM and embedding providers that enable full end-to-end testing without API keys or external service calls.

## How No-Credential Local Testing Works

The `ai-service-kit` library provides `MockLLMProvider` and `MockEmbeddingProvider`. These can be configured via environment variables to replace real providers at startup:

```bash
LLM_PROVIDER=mock
EMBEDDING_PROVIDER=mock
```

`MockLLMProvider` returns a deterministic fixed-text response for any prompt. `MockEmbeddingProvider` returns a fixed-dimension random vector seeded by a configurable integer, ensuring reproducible retrieval behavior across test runs.

## Benefits of Mock Providers

- **No API keys required**: Developers can run the full platform locally without OpenAI or Gemini credentials.
- **Deterministic**: Same inputs always produce the same outputs, making tests repeatable.
- **Fast**: No network calls, so iteration cycles are instant.
- **Cost-free**: No token charges during development and CI runs.

## Tradeoff: Mock vs Real Provider Validation

Mock providers improve repeatability and speed. However, they do not validate that prompts are well-formed for a specific model, that the model produces useful output, or that rate limits and context windows are respected. Real-provider tests are still required to validate production behavior — typically run in a separate integration test suite or as part of pre-deploy checks.

## Configuration

```bash
# Use mock providers for all local testing
LLM_PROVIDER=mock
MOCK_LLM_MODEL=mock-llm
MOCK_LLM_SEED=17
EMBEDDING_PROVIDER=mock
MOCK_EMBED_MODEL=mock-embed
MOCK_EMBED_DIMENSION=128
MOCK_EMBED_SEED=7
```
