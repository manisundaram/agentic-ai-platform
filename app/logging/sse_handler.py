from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

LOG_QUEUE: asyncio.Queue[str] = asyncio.Queue()


@dataclass(slots=True)
class SSELogEvent:
    time: str
    level: str
    source: str
    message: str

    def to_json(self) -> str:
        return json.dumps(
            {
                "time": self.time,
                "level": self.level,
                "source": self.source,
                "message": self.message,
            },
            ensure_ascii=True,
        )


class SSELogHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            event = SSELogEvent(
                time=datetime.now(UTC).strftime("%H:%M:%S"),
                level=record.levelname,
                source=self._resolve_source(record.name),
                message=self.format(record),
            )
            try:
                LOG_QUEUE.put_nowait(event.to_json())
            except asyncio.QueueFull:
                pass
        except Exception:
            self.handleError(record)

    def _resolve_source(self, logger_name: str) -> str:
        normalized = logger_name.lower()
        if normalized.startswith("langgraph"):
            return "langgraph"
        if normalized.startswith("llama_index"):
            return "llamaindex"
        if normalized.startswith("app.graph"):
            return "graph"
        if normalized.startswith("app.mcp"):
            return "mcp"
        if normalized.startswith("app.retrieval"):
            return "retrieval"
        if normalized.startswith("uvicorn"):
            return "server"
        if normalized.startswith("app.logging"):
            return "logging"
        return "app"


class SSELogFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if getattr(record, "_sse_event", False):
            return False
        return True


def install_sse_log_handler(level: int = logging.INFO) -> SSELogHandler:
    root_logger = logging.getLogger()
    existing_handler = next((handler for handler in root_logger.handlers if isinstance(handler, SSELogHandler)), None)
    if existing_handler is not None:
        return existing_handler

    handler = SSELogHandler()
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(name)s %(message)s"))
    root_logger.addHandler(handler)
    if root_logger.level == logging.NOTSET or root_logger.level > level:
        root_logger.setLevel(level)
    return handler


def clear_sse_queue() -> None:
    while not LOG_QUEUE.empty():
        try:
            LOG_QUEUE.get_nowait()
        except asyncio.QueueEmpty:
            break


def format_demo_log(source: str, level: str, message: str) -> str:
    return SSELogEvent(
        time=datetime.now(UTC).strftime("%H:%M:%S"),
        level=level,
        source=source,
        message=message,
    ).to_json()


def enqueue_demo_log(*, source: str, level: str, message: str) -> None:
    try:
        LOG_QUEUE.put_nowait(format_demo_log(source, level, message))
    except asyncio.QueueFull:
        pass
