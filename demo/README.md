# Demo Guide: IT Support AI Agent

This folder contains the interactive demo UI for the project. It is optimized for recruiter demos and interview walkthroughs.

## Demo Objective

Show, in a few minutes, that this project is not just a chatbot UI:

- Stateful orchestration with LangGraph
- Human-In-The-Loop (HITL) approval for risky actions
- Traceability with LangSmith
- Retrieval + tool integration (LlamaIndex + MCP)

## Quick Start

1. Start the backend from repository root.

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

2. Open the demo UI.

- Route: `http://localhost:8000/demo`
- Or open `demo/index.html` directly (API base still points to `http://localhost:8000`)

3. Confirm status indicators:

- Backend shows `Connected`
- Log stream changes from `Disconnected` to `Recording`

## Demo Assets

Place files in `demo/assets/`:

- `demo-recording.mp4`
- `screenshot-1.png`
- `screenshot-2.png`

The UI auto-detects these files and renders them when present.

## Recommended 3-Minute Script

1. Introduce project in one sentence.
   This is an IT Support AI Agent using LangGraph for multi-turn reasoning, with human approval checkpoints and LangSmith tracing.

2. Run a realistic IT request.
   Use Scenario 2 (new employee onboarding with VPN + MDM), then click Submit.

3. Show orchestration and visibility.

- Trace panel updates node flow (`retrieve_context -> generate_answer -> critique_answer`)
- Backend Logs stream events live from SSE
- Conversation card shows answer, sources, and LangSmith trace link

4. Show safety gate.
   If marked risky, the HITL panel appears with Approve, Reject, Modify.

5. Resume from HITL.
   Apply a decision and show continuation from the same `thread_id`.

## What Each Panel Demonstrates

- IT Support AI Agent panel:
  Input, final answer, source citations, and run-level latency
- Agent Trace panel:
  Stateful workflow progression, node execution, iteration count
- Backend Logs panel:
  Operational observability and real-time event stream
- HITL panel:
  Human control for sensitive actions before completion

## Click-To-Enlarge Media

- Screenshots are clickable to open in a larger preview
- Demo video is also clickable for enlarged playback
- Close options: Close button, click outside, or `Esc`

## Troubleshooting

1. Backend status is Disconnected.

- Verify backend is running on `http://localhost:8000`
- Check `GET /ping`

2. Log stream does not connect.

- Verify `GET /demo/logs` returns an SSE stream
- Restart backend if hot reload left stale state

3. Query feels slow.

- UI uses `stream=true` so progress is visible
- Latency may still depend on provider response times and max iterations

4. No screenshots/video visible.

- Confirm exact filenames in `demo/assets/`
- Hard refresh browser cache

## Related Documentation

- Root project overview: [../README.md](../README.md)
- Architecture details: [../docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md)
- Installation and setup: [../docs/INSTALLATION.md](../docs/INSTALLATION.md)
- Framework decision story: [../docs/FRAMEWORK_COMPARISON.md](../docs/FRAMEWORK_COMPARISON.md)
