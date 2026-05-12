# LangSmith Tracing

LangSmith is the observability platform for LangChain and LangGraph workloads. It captures execution spans, run context, model inputs and outputs, and evaluation results.

## Purpose of LangSmith Tracing in This Platform

LangSmith tracing captures:

- Every node execution in the LangGraph run as a named span.
- LLM inputs (prompt messages) and outputs (completions) with token counts.
- Tool calls and their arguments.
- End-to-end latency per node and per run.
- The full thread_id and run metadata for correlation.

This makes LangSmith the primary debugging tool when a query produces an unexpected answer. Engineers can inspect exactly which context chunks were retrieved, what prompt was sent to the LLM, and which conditional edge was taken.

## Integration

Tracing is enabled via `langsmith.run_helpers.tracing_context`:

```python
with tracing_context(project_name="agentic-ai-platform", enabled=True):
    state = await graph.ainvoke(initial_state, config=config)
```

The `LANGSMITH_API_KEY` environment variable authenticates the LangSmith client.

## Trace URL

After each query, the response includes a `langsmith_trace_url` field pointing to the run in the LangSmith dashboard. This enables one-click navigation from a demo response to the full execution trace.

## Tracing vs Evaluation

Tracing explains what happened in a specific run. Evaluation quantifies quality across many runs using metrics like faithfulness, context precision, and task completion rate. Both are needed for production readiness: tracing for debugging individual failures, evaluation for tracking regression risk over time.
