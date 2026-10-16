"""Optional LangSmith tracing. A no-op unless LANGSMITH_TRACING=true and the package is installed."""
from __future__ import annotations

import os


def traceable(name: str | None = None, run_type: str = "chain"):
    try:
        if os.getenv("LANGSMITH_TRACING", "").lower() == "true":
            from langsmith import traceable as _t
            return _t(name=name, run_type=run_type)
    except Exception:  # package missing or misconfigured: never break the app for tracing
        pass

    def deco(fn):
        return fn
    return deco


def wrap_anthropic(client):
    """Wrap the Anthropic client so each LLM call shows up as a LangSmith child run."""
    try:
        if os.getenv("LANGSMITH_TRACING", "").lower() == "true":
            from langsmith.wrappers import wrap_anthropic as _w
            return _w(client)
    except Exception:
        pass
    return client


def wrap_openai(client):
    """Same for OpenAI-compatible clients (used for Groq)."""
    try:
        if os.getenv("LANGSMITH_TRACING", "").lower() == "true":
            from langsmith.wrappers import wrap_openai as _w
            return _w(client)
    except Exception:
        pass
    return client
