"""Application package exports the ASGI app lazily to avoid import side effects."""

from typing import Any

__all__ = ["app"]


def __getattr__(name: str) -> Any:
	if name == "app":
		from .main import app

		return app
	raise AttributeError(name)
