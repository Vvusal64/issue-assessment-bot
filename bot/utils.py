"""Shared helpers: logging, retry decorator, ADF flattening."""

from __future__ import annotations

import functools
import logging
import random
import time
from typing import Any, Callable, Iterable, Optional, Tuple, Type, TypeVar

import requests


T = TypeVar("T")


def configure_logging(level: str = "INFO") -> logging.Logger:
    """Configure root logging once and return the bot logger."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    return logging.getLogger("bot")


# ----------------------------------------------------------------------
# Retry
# ----------------------------------------------------------------------

class RetryableError(Exception):
    """Marker for errors the retry helper should treat as transient."""


def _is_retryable_response(resp: requests.Response) -> bool:
    if resp.status_code == 429:
        return True
    if 500 <= resp.status_code < 600:
        return True
    return False


def retry(
    *,
    attempts: int = 4,
    base_delay: float = 1.0,
    max_delay: float = 8.0,
    exceptions: Tuple[Type[BaseException], ...] = (
        requests.RequestException, RetryableError,
    ),
    logger: Optional[logging.Logger] = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Exponential-backoff retry decorator with jitter.

    Retries on declared exceptions. If a ``requests.Response`` is returned
    and has a retryable status code (5xx/429), it is treated as a failure
    on the first ``attempts - 1`` tries. The final attempt returns the
    response unchanged so the caller can handle it.
    """
    log = logger or logging.getLogger("bot.retry")

    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exc: Optional[BaseException] = None
            for attempt in range(1, attempts + 1):
                try:
                    result = fn(*args, **kwargs)
                except exceptions as exc:  # transient
                    last_exc = exc
                    if attempt == attempts:
                        raise
                    delay = min(max_delay, base_delay * 2 ** (attempt - 1))
                    delay += random.uniform(0, 0.25)
                    log.warning(
                        "%s failed (attempt %d/%d): %s — retrying in %.1fs",
                        fn.__name__, attempt, attempts, exc, delay,
                    )
                    time.sleep(delay)
                    continue

                # Success path: also retry on retryable HTTP responses.
                if isinstance(result, requests.Response) \
                        and _is_retryable_response(result) \
                        and attempt < attempts:
                    delay = min(max_delay, base_delay * 2 ** (attempt - 1))
                    delay += random.uniform(0, 0.25)
                    log.warning(
                        "%s got HTTP %d (attempt %d/%d) — retrying in %.1fs",
                        fn.__name__, result.status_code, attempt,
                        attempts, delay,
                    )
                    time.sleep(delay)
                    continue
                return result
            # Should be unreachable, but satisfies type checkers.
            assert last_exc is not None
            raise last_exc
        return wrapper
    return decorator


# ----------------------------------------------------------------------
# Atlassian Document Format (ADF) → plain text
# ----------------------------------------------------------------------

def adf_to_text(node: Any) -> str:
    """Best-effort flatten of an ADF document into searchable plain text.

    Jira Cloud returns descriptions and comment bodies as ADF (a JSON
    tree). Extraction works on text, so we walk the tree and pull out
    every ``text`` node, prefixing block-level boundaries with newlines.
    Unknown node types fall through harmlessly.
    """
    if node is None:
        return ""
    if isinstance(node, str):
        return node
    if isinstance(node, list):
        return "\n".join(filter(None, (adf_to_text(n) for n in node)))
    if not isinstance(node, dict):
        return ""

    node_type = node.get("type")
    text_value = node.get("text")
    children = node.get("content")

    inner = adf_to_text(children) if children else ""
    if text_value:
        inner = f"{text_value}\n{inner}".strip() if inner else text_value

    block_types = {
        "paragraph", "heading", "bulletList", "orderedList",
        "listItem", "blockquote", "codeBlock", "rule", "panel",
        "table", "tableRow", "tableCell", "tableHeader", "doc",
    }
    if node_type in block_types and inner:
        return f"\n{inner}\n"
    return inner


def truncate(text: str, limit: int) -> str:
    """Trim text to a character budget, preserving the head and tail."""
    if len(text) <= limit:
        return text
    head = limit // 2
    tail = limit - head - 20
    return f"{text[:head]}\n…[{len(text) - limit} chars omitted]…\n{text[-tail:]}"


def chunked(seq: Iterable[Any], size: int):
    """Yield successive chunks of size ``size`` from ``seq``."""
    bucket = []
    for item in seq:
        bucket.append(item)
        if len(bucket) == size:
            yield bucket
            bucket = []
    if bucket:
        yield bucket
