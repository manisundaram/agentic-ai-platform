# Graph State Design

The LangGraph state object is the shared memory for a single agent run. Every node reads from and writes to this state.

## State Schema

```python
class AgentState(TypedDict):
    messages: list[dict]       # Conversation history including system, user, assistant, critic roles
    query: str                 # The original user query, immutable after initialization
    context: str               # Retrieved document context, updated by retrieve_context node
    tool_calls: list[dict]     # Log of all tool invocations including sources and models used
    iteration_count: int       # Number of generate+critique cycles completed
    max_iterations: int        # Upper bound on cycles, prevents infinite loops
    final_answer: str          # The most recent answer from generate_answer
    requires_human_review: bool # Set by critique_answer when risky action detected
    eval_scores: dict          # Groundedness and completeness scores from critique
```

## Thread Identity

Every run is associated with a `thread_id`. The LangGraph checkpointer uses the thread_id as the persistence key. Multiple concurrent users each have their own thread, and their states do not interfere.

## Message Role Normalization

LangGraph state allows arbitrary role labels (including `"critic"` for internal critique messages). Before sending messages to an LLM provider, the `_normalize_message_roles` method maps non-standard roles to `"system"` to comply with OpenAI's role enumeration (`system`, `assistant`, `user`, `function`, `tool`, `developer`).

## Checkpoint Resume

When a run is paused at the `human_review` interrupt, the state is preserved. On resume via `POST /agent/resume`, the human's decision is merged into the state via `Command(update={...})`, and the graph continues from the interrupted node.
