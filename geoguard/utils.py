"""Shared utilities for the geoguard package."""

import functools
from collections.abc import Awaitable, Callable
from datetime import date, timedelta
from typing import Any

import httpx


def date_range(start: str | date, end: str | date) -> list[date]:
    """Inclusive list of dates between start and end.

    Accepts ISO strings (YYYY-MM-DD) or `date` objects for either bound.
    Returns an empty list if `end` is before `start`.
    """
    s = date.fromisoformat(start) if isinstance(start, str) else start
    e = date.fromisoformat(end) if isinstance(end, str) else end
    return [s + timedelta(days=i) for i in range((e - s).days + 1)]


def graceful_http(fn: Callable[..., Awaitable[dict]]) -> Callable[..., Awaitable[dict]]:
    """Catch httpx errors and return a structured failure dict instead of raising.

    Keeps a tool's network failures from killing the entire verification:
    the agent receives a dict with an `error` key and can reason about
    the failure (typically: skip and proceed, mark INCONCLUSIVE).

    Decorator order with @registry matters — apply @graceful_http first
    (inner), then @registry (outer):

        @registry(EventType.FLOOD)
        @graceful_http
        async def my_tool(...) -> dict: ...
    """

    @functools.wraps(fn)
    async def wrapper(*args: Any, **kwargs: Any) -> dict:
        try:
            return await fn(*args, **kwargs)
        except httpx.HTTPError as e:
            return {
                "tool": fn.__name__,
                "error": f"{type(e).__name__}: {e}",
            }

    return wrapper
