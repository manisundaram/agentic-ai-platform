# LangGraph

LangGraph is a graph-based orchestration framework built on top of LangChain. It adds typed state, explicit node edges, checkpointing, and controlled cycles to multi-step agent workflows.

## What LangGraph Adds Over a Simple Prompt Chain

A simple prompt chain executes steps sequentially with no shared state, no cycles, and no ability to pause mid-run. LangGraph adds:

- **Typed state**: Every node reads and writes to a shared typed state object, making inter-node communication explicit and inspectable.
- **Explicit edges**: Control flow is modeled as edges between nodes. Conditional edges let the graph branch or loop based on state values.
- **Checkpointing**: The `MemorySaver` checkpointer persists graph state after every node, enabling resumable execution. A run can be interrupted, stored, and resumed from exactly where it paused.
- **Controlled cycles**: Unlike a DAG, LangGraph supports cycles (self-correction loops) with a configurable iteration limit to prevent infinite loops.
- **HITL interrupt**: The `interrupt_before` parameter on a node causes the graph to pause before executing that node, yielding control to a human reviewer.

## Node Architecture

This platform uses four nodes:

1. `retrieve_context`: Calls LlamaIndex to retrieve relevant document chunks from ChromaDB.
2. `generate_answer`: Calls the LLM provider with retrieved context to generate a grounded answer.
3. `critique_answer`: Checks the answer for groundedness and flags risky actions requiring human review.
4. `human_review`: An interrupt node that pauses execution for human approval before continuing.

## Conditional Edges

After `critique_answer`, a conditional edge decides:

- If `has_issues=True` and `iteration_count < max_iterations`: loop back to `retrieve_context` for another attempt.
- If `requires_human_review=True`: route to `human_review` interrupt node.
- Otherwise: end the run.

## State Schema

```python
class AgentState(TypedDict):
    messages: list[dict]
    query: str
    context: str
    tool_calls: list[dict]
    iteration_count: int
    max_iterations: int
    final_answer: str
    requires_human_review: bool
    eval_scores: dict
```
