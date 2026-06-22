#!/usr/bin/env python3
"""Shared LLM plumbing for the transcript-preparation scripts.

Centralises the OpenAI-compatible client (HPI endpoint, gpt-oss-120b by default), a
JSON-returning chat helper, a cheap token estimate for context budgeting, and a
thread-pool map for processing independent sessions in parallel.
"""

from __future__ import annotations

import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

DEFAULT_MODEL = "gpt-oss-120b"
# gpt-oss is a reasoning model: max_tokens caps reasoning + answer, so a small value
# silently truncates to empty content. We self-host without token limits, so default to
# generous headroom; override per call/CLI if needed.
DEFAULT_MAX_TOKENS = 32000
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def load_env() -> None:
    """Load the project .env if python-dotenv is available (no-op otherwise)."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    # Search upward from this file so it works regardless of cwd.
    here = Path(__file__).resolve()
    for parent in (here.parent, *here.parents):
        env = parent / ".env"
        if env.is_file():
            load_dotenv(env)
            return
    load_dotenv()  # fall back to dotenv's own discovery


def have_key() -> bool:
    load_env()
    return bool(os.getenv("OPENAI_API_KEY"))


def make_client(base_url: str | None = None):
    """Construct the OpenAI client from env; raise a clear error if the key is unset."""
    load_env()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set (add it to .env)")
    from openai import OpenAI
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"),
                  base_url=base_url or os.getenv("OPENAI_API_BASE"))


def chat_json(client, model: str, prompt: str, *, max_tokens: int = DEFAULT_MAX_TOKENS) -> dict:
    """temperature=0 chat call; return the first JSON object found, or {} on any failure.

    The default budget is generous because gpt-oss is a reasoning model: reasoning tokens
    count against max_tokens, and too small a budget yields an empty (None) answer. Warns if
    the response was truncated (``finish_reason == 'length'``) or empty so it never bites
    silently — raise ``max_tokens`` if you see that.
    """
    try:
        choice = client.chat.completions.create(
            model=model, temperature=0, max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        ).choices[0]
    except Exception as exc:
        print(f"  LLM error: {exc!r}", file=sys.stderr)
        return {}
    raw = choice.message.content
    if choice.finish_reason == "length" or not raw:
        print(f"  LLM response truncated/empty (finish_reason={choice.finish_reason}, "
              f"max_tokens={max_tokens}) — raise --max-tokens", file=sys.stderr)
    m = _JSON_RE.search(raw or "")
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}


def estimate_tokens(text: str) -> int:
    """Cheap token estimate (~4 chars/token) for context budgeting."""
    return len(text) // 4


def run_parallel(items: list, fn, concurrency: int) -> list:
    """Map ``fn`` over ``items`` with a thread pool (OpenAI calls are I/O-bound).

    Results are returned in input order. Exceptions are captured and returned as the
    item's result so one failure does not abort the batch.
    """
    if concurrency <= 1 or len(items) <= 1:
        return [_safe(fn, it) for it in items]
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        return list(pool.map(lambda it: _safe(fn, it), items))


def _safe(fn, item):
    try:
        return fn(item)
    except Exception as exc:  # surfaced by the caller's reporting
        return exc
