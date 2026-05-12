# Evaluation Framework

Evals quantify the quality of retrieval and agent behavior using structured test cases. They complement LangSmith tracing: tracing explains individual run behavior, while evals measure aggregate quality across a dataset.

## Why Evaluation Is Required Alongside Tracing for Production Readiness

Tracing captures what happened in a specific run — which nodes executed, what the LLM received, what it returned. It is diagnostic.

Evaluation quantifies quality over a dataset of test cases — measuring metrics like faithfulness, context recall, task completion rate, and hallucination rate. It is preventive: it catches regressions before they reach production. Together, tracing and evaluation provide both incident investigation capability and regression detection coverage.

## Eval Metrics

**RAG metrics** (measured per factual_retrieval case):

- `faithfulness`: Fraction of answer tokens that appear in the retrieved context. Measures groundedness.
- `context_precision`: Fraction of retrieved sources that match expected sources. Measures retrieval accuracy.
- `context_recall`: Fraction of expected source tokens covered by retrieved sources. Measures retrieval completeness.
- `answer_relevancy`: Token overlap between query and answer. Measures topicality.

**Agent metrics** (measured per multi_hop_reasoning and edge_case):

- `task_completion_rate`: Fraction of cases where the answer has sufficient token overlap with the expected answer.
- `tool_call_accuracy`: Fraction of cases where tool usage matches the `requires_tool_use` flag.
- `cycle_efficiency`: Fraction of cases where iteration count stays within the efficiency target.
- `hallucination_rate`: Fraction of cases where the answer has low token overlap with the retrieved context.

## Running Evals

```bash
python -m app.evals
```

Results are written to `evals/results/latest.json`. Rerun after indexing new documents to see updated scores.
