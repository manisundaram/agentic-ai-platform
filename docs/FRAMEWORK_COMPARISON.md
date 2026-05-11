# Framework Comparison

## 1. Custom ReAct (agents-api) vs LangGraph (this repo)

### What LangGraph adds
- Typed state contracts that make multi-node workflows safer under refactors.
- Native cycles and conditional routing without handwritten loop orchestration.
- Human-in-the-loop interruption and resume as first-class runtime behavior.
- Checkpoint persistence for long-running or review-gated executions.
- LangSmith-friendly execution traces with graph-level observability.

### What LangGraph hides
- Raw message-passing mechanics between each reasoning step.
- Manual control over loop termination edge cases and low-level retry policies.

### When to choose custom ReAct
- You are optimizing for learning, transparency, and explicit reasoning internals.
- You need maximal low-level control and can accept higher maintenance.
- You want minimal framework dependency surface for a small scoped service.

### When to choose LangGraph
- You are building team-owned production systems with evolving workflows.
- You need stateful conversations, branching, retries, and HITL controls.
- You want reliability and debuggability to be baked into orchestration.

## 2. LlamaIndex vs raw ChromaDB (semantic-search-api)

### What LlamaIndex adds
- Document loaders and ingestion pipelines for heterogeneous source formats.
- Configurable chunking strategies and retrieval composition primitives.
- Hybrid retrieval and reranking patterns available without custom rebuilds.
- Large connector ecosystem that accelerates integration work.

### What LlamaIndex hides
- Fine-grained vector-store operations and direct index lifecycle control.
- Store-specific query tuning details that can matter for peak optimization.

### When to choose LlamaIndex
- You need rapid iteration on document-heavy RAG systems.
- You expect ingestion, chunking, retrieval strategy, and connectors to evolve.
- You value faster delivery over full control of storage-level internals.

### When to choose raw ChromaDB
- You require precise control over indexing/query behavior and performance.
- You already have custom retrieval pipelines that do not need framework layers.
- You are optimizing for minimal abstraction in a narrowly scoped service.

## 3. MCP vs bespoke tools (agents-api tool layer)

### What MCP standardizes
- Tool discovery semantics across independently developed tool providers.
- Shared schema contracts for tool inputs and outputs.
- Consistent invocation protocol regardless of tool host implementation.

### What MCP solves
- Tool sprawl and integration inconsistency when multiple teams add capabilities.
- Repeated one-off adapters per agent runtime and per service boundary.
- Fragile hand-rolled contracts that drift over time.

### When MCP wins
- You are building a platform that serves many product teams and agent runtimes.
- You need governance and compatibility guarantees for tool interfaces.
- You want composability and interoperability across organizational boundaries.

### When bespoke tools win
- A single team owns both orchestrator and tools end to end.
- You need highly specialized performance tuning beyond protocol abstractions.
- You can tolerate tight coupling for speed in a narrow deployment context.

## 4. CrewAI vs LangGraph

### CrewAI profile
- Role-based collaboration model that is quick to assemble and demo.
- Faster to ship initial multi-agent behavior for straightforward workflows.
- Less flexible once branching logic, durable state, and interrupts get complex.

### LangGraph profile
- Stateful graph orchestration with explicit control-flow semantics.
- Better fit for production reliability, replayability, and HITL operations.
- Higher implementation complexity but stronger long-term operational posture.

### Decision framework: timeline vs reliability
- Choose CrewAI when timeline pressure dominates and workflow complexity is moderate.
- Choose LangGraph when reliability, auditability, and controlled branching dominate.
- Re-evaluate as soon as workflow count, compliance risk, or team count increases.