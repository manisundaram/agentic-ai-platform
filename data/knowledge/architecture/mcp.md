# MCP — Model Context Protocol

MCP (Model Context Protocol) is a standardized protocol for tool discovery, schema contracts, and invocation semantics between agents and external tools or services.

## Why Teams Adopt MCP Instead of Custom Tool Wiring

Custom tool wiring requires each agent implementation to define its own schema format, discovery mechanism, and invocation contract. This creates tight coupling between the agent and each tool. When the tool interface changes, every agent that uses it must be updated.

MCP standardizes:
- **Tool discovery**: Agents call a standard `list_tools` endpoint to discover available tools and their schemas at runtime.
- **Schema contracts**: Every tool is described by a JSON schema that defines its input parameters and output shape. Agents use this schema to construct valid calls without hardcoded assumptions.
- **Invocation semantics**: Tool calls follow a consistent request/response format regardless of which tool is being called.

This decoupling means tools can be updated, replaced, or added without changing the agent orchestration code.

## MCP in This Platform

The MCP server in `app/mcp/server.py` exposes tools including:
- `search_documents`: Queries the LlamaIndex retriever for relevant context.
- `calculate`: Evaluates arithmetic expressions.
- `get_current_time`: Returns the current UTC time.

The `MCPToolAdapter` in `app/mcp/client.py` connects to the MCP server as a subprocess using stdio transport. The agent can discover all available tools via `GET /mcp/tools/list` without knowing their implementations.

## MCP vs Bespoke Tool Wiring

| Aspect | Bespoke Tool Wiring | MCP |
|--------|---------------------|-----|
| Discovery | Hardcoded in agent | Dynamic via list_tools |
| Schema | Custom per tool | Standardized JSON schema |
| Coupling | Tight | Loose |
| Cross-agent reuse | Difficult | Built-in |
