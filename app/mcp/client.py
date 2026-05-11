from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

from langchain_core.tools import StructuredTool
from mcp import ClientSession, StdioServerParameters, Tool
from mcp.client.stdio import stdio_client
from pydantic import BaseModel, ConfigDict, Field, create_model


class MCPServerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    command: str = Field(min_length=1)
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    cwd: str | None = None
    read_timeout_seconds: float = Field(default=30.0, gt=0.0)


@dataclass(slots=True)
class _ToolDefinition:
    name: str
    description: str
    input_schema: dict[str, Any]


class MCPToolAdapter:
    def __init__(self, config: MCPServerConfig) -> None:
        self._config = config

    @asynccontextmanager
    async def _session(self):
        server_params = StdioServerParameters(
            command=self._config.command,
            args=self._config.args,
            env=self._config.env or None,
            cwd=self._config.cwd,
        )
        async with stdio_client(server_params) as (read_stream, write_stream):
            session = ClientSession(read_stream, write_stream)
            async with session:
                await session.initialize()
                yield session

    async def list_tools(self) -> list[Tool]:
        async with self._session() as session:
            result = await session.list_tools()
            return list(result.tools)

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        async with self._session() as session:
            result = await session.call_tool(
                name,
                arguments or {},
                read_timeout_seconds=timedelta(seconds=self._config.read_timeout_seconds),
            )
        structured = getattr(result, "structuredContent", None)
        if structured is not None:
            return dict(structured)

        content_blocks: list[dict[str, Any]] = []
        for block in getattr(result, "content", []):
            if hasattr(block, "model_dump"):
                content_blocks.append(block.model_dump())
            else:
                content_blocks.append({"text": str(block)})
        return {"content": content_blocks, "is_error": getattr(result, "isError", False)}

    async def get_langchain_tools(self) -> list[StructuredTool]:
        definitions = [self._to_definition(tool) for tool in await self.list_tools()]
        tools: list[StructuredTool] = []
        for definition in definitions:
            args_schema = self._json_schema_to_model(definition)

            async def _invoke(_definition: _ToolDefinition = definition, **kwargs: Any) -> dict[str, Any]:
                return await self.call_tool(_definition.name, kwargs)

            tools.append(
                StructuredTool.from_function(
                    coroutine=_invoke,
                    name=definition.name,
                    description=definition.description,
                    args_schema=args_schema,
                    infer_schema=False,
                )
            )
        return tools

    def _to_definition(self, tool: Tool) -> _ToolDefinition:
        return _ToolDefinition(
            name=tool.name,
            description=tool.description or tool.title or tool.name,
            input_schema=dict(tool.inputSchema or {}),
        )

    def _json_schema_to_model(self, definition: _ToolDefinition) -> type[BaseModel]:
        schema = definition.input_schema
        properties = schema.get("properties", {})
        required = set(schema.get("required", []))
        field_definitions: dict[str, tuple[Any, Field]] = {}
        for name, property_schema in properties.items():
            annotation = self._schema_type_to_annotation(property_schema)
            default = ... if name in required else None
            field_definitions[name] = (
                annotation,
                Field(default=default, description=property_schema.get("description")),
            )

        model_name = "".join(part.capitalize() for part in definition.name.split("_")) + "Args"
        return create_model(model_name, **field_definitions)

    def _schema_type_to_annotation(self, schema: dict[str, Any]) -> Any:
        schema_type = schema.get("type")
        if schema_type == "string":
            return str
        if schema_type == "integer":
            return int
        if schema_type == "number":
            return float
        if schema_type == "boolean":
            return bool
        if schema_type == "array":
            item_type = self._schema_type_to_annotation(schema.get("items", {"type": "string"}))
            return list[item_type]
        if schema_type == "object":
            return dict[str, Any]
        return Any