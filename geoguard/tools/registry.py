from __future__ import annotations

from collections import defaultdict
from typing import Callable

from pydantic_ai.toolsets import FunctionToolset

from geoguard.schemas import EventType


class ToolRegistry:
    """Singleton registry of tools tagged by event type.

    Always returns the same instance — `ToolRegistry()` is idempotent.
    Use `ToolRegistry.clear()` in tests to clear state for isolation.
    """

    _instance: ToolRegistry | None = None

    def __new__(cls) -> ToolRegistry:
        if cls._instance is None:
            instance = super().__new__(cls)
            instance._tools = defaultdict(list)
            cls._instance = instance
        return cls._instance

    @classmethod
    def clear(cls) -> None:
        """Empty the singleton's tools in place — for test isolation.

        Does not swap the instance, so any module that imported the
        module-level `registry` binding stays correctly wired after a clear.
        """
        if cls._instance is not None:
            cls._instance._tools.clear()

    def register(self, *event_types: EventType):
        """Decorator: register a tool callable under one or more event types.

        Supports multi-event in one call AND stacking — both forms work:

            @registry(EventType.FLOOD, EventType.OTHER)
            async def t1(...): ...

            @registry(EventType.FLOOD)
            @registry(EventType.OTHER)
            async def t2(...): ...

        Class-based tools: instantiate first, register the instance.
        """

        def deco(fn):
            for et in event_types:
                self._tools[et] = [
                    t for t in self._tools[et] if t.__name__ != fn.__name__
                ]
                self._tools[et].append(fn)
            return fn

        return deco

    def __call__(self, *event_types: EventType):
        return self.register(*event_types)

    def get_candidates(self, event_type: EventType) -> list[Callable]:
        """Candidates for this event type + always-on (OTHER) tools.

        Tools registered under multiple event types appear once. Falls back
        to ALL registered tools when nothing matches.
        """
        matched = list(self._tools.get(event_type, []))
        if event_type is not EventType.OTHER:
            matched.extend(self._tools.get(EventType.OTHER, []))
        if not matched:
            matched = [fn for fns in self._tools.values() for fn in fns]
        return list({fn.__name__: fn for fn in matched}.values())

    def build_toolset(
        self, tools: list[Callable], id: str = "selected"
    ) -> FunctionToolset:
        return FunctionToolset(tools=tools, id=id)


registry = ToolRegistry()
