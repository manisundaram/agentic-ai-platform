# HITL — Human-In-The-Loop

Human-In-The-Loop (HITL) is a design pattern where an agent pauses execution and waits for a human decision before continuing with a risky or irreversible action.

## How HITL Works in LangGraph

LangGraph's `interrupt_before` parameter on the `human_review` node causes the graph executor to stop before entering that node. The graph state is persisted via the `MemorySaver` checkpointer so no work is lost.

The run is paused in a way that:

1. The client receives a response with `requires_human_review=True`.
2. The thread's state remains stored in the checkpointer under its `thread_id`.
3. A human can inspect the pending action and call `POST /agent/resume` with a decision of `approve`, `reject`, or `modify`.
4. The graph resumes from the checkpoint, applies the human's decision to the state, and continues to completion.

## Checkpointing Enables Safe HITL

Without checkpointing, an interrupt would lose all in-progress state. LangGraph's `MemorySaver` checkpointer ensures:

- State is preserved across the pause boundary.
- The graph can resume from exactly the interrupted node.
- Multiple concurrent threads can each be paused independently.

## Triggering Conditions

The `critique_answer` node sets `requires_human_review=True` when it detects risky keywords in the query or answer, including: `vpn`, `mdm`, `enroll`, `admin`, `password reset`, `disable`, `credential`, `access set up`, `production`.

## Resume Decisions

- `approve`: The pending action proceeds; the agent continues with its original answer.
- `reject`: The action is blocked; the agent returns a rejection message.
- `modify`: The human provides a modified answer that replaces the agent's output.
