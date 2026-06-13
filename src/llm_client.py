"""OpenAI LLM wrapper used by judge + reformulator.

Design goals (mirrors the BrowseComp-Plus llm wiring):
    - Same client instance reused across tools (constructed once from config).
    - Synchronous public API but thread-pool fan-out for parallel judge calls
      within a round. Async is overkill for our concurrency budget (<= 10
      calls/round) and complicates the disk cache.
    - On-disk cache via `diskcache` keyed by (model, role, temp, prompt, schema)
      so reruns are free and identical.
    - Retries with exponential backoff on rate-limit / transient errors.
    - Returns structured Pydantic objects when a response_format is requested
      (reformulator); plain text otherwise (judge).
    - Usage accounting per call so the agent can total cost per query.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Type

try:  # pydantic is in requirements but keep the import safe for static checks
    from pydantic import BaseModel
except Exception:  # pragma: no cover
    BaseModel = object  # type: ignore[assignment,misc]

try:
    import openai
except Exception:  # pragma: no cover - allow importing the module for tests
    openai = None  # type: ignore[assignment]

try:
    import diskcache
except Exception:  # pragma: no cover
    diskcache = None  # type: ignore[assignment]


# GPT-4.1 published per-1M-token prices, USD. Update if OpenAI revises.
_PRICES_PER_1M = {
    "gpt-4.1": {"input": 2.00, "output": 8.00},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
}


def _estimate_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """USD estimate; returns 0.0 if the model is not in the price table."""
    base = model.split("/", 1)[-1].split(":", 1)[0]
    for k, p in _PRICES_PER_1M.items():
        if base.startswith(k):
            return (input_tokens * p["input"] + output_tokens * p["output"]) / 1_000_000
    return 0.0


def _stable_key(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


@dataclass
class LLMResult:
    text: str
    parsed: Any | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    cache_hit: bool = False

    @property
    def usd(self) -> float:
        return _estimate_usd(self.model, self.input_tokens, self.output_tokens)


class LLMClient:
    """Thin wrapper around `openai.OpenAI` with disk cache + retries."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        cache_dir: str | Path | None = ".cache/llm",
        max_retries: int = 3,
        extra_body: dict[str, Any] | None = None,
    ):
        if openai is None:
            raise RuntimeError("openai package not installed; pip install openai>=1.50")
        self.client = openai.OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            base_url=base_url,
        )
        self.max_retries = max_retries
        # extra_body is forwarded verbatim to every API call.
        # Use it to pass provider-specific parameters, e.g. to disable
        # thinking mode on OpenRouter Qwen3 models:
        #   extra_body: {"thinking": {"type": "disabled"}}
        self._extra_body: dict[str, Any] = extra_body or {}
        if cache_dir is None or diskcache is None:
            self._cache = None
        else:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            self._cache = diskcache.Cache(str(cache_dir))

    # ---------------------------------------------------------------- helpers

    def _cache_get(self, key: str) -> dict | None:
        if self._cache is None:
            return None
        try:
            return self._cache.get(key)
        except Exception:
            return None

    def _cache_set(self, key: str, value: dict) -> None:
        if self._cache is None:
            return
        try:
            self._cache.set(key, value)
        except Exception:
            pass

    def _with_retries(self, fn, *args, **kwargs):
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                return fn(*args, **kwargs)
            except Exception as e:  # noqa: BLE001 - retry blanket then re-raise
                last_exc = e
                # Crude backoff: 0.5 * 2**attempt seconds + small jitter
                delay = 0.5 * (2 ** attempt) + 0.1 * attempt
                time.sleep(delay)
        assert last_exc is not None
        raise last_exc

    # --------------------------------------------------------- public surface

    def chat_text(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 1024,
        cache_namespace: str = "chat",
    ) -> LLMResult:
        """Plain-text Chat Completions call (used by the judge)."""
        cache_key = _stable_key(
            {
                "ns": cache_namespace,
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        cached = self._cache_get(cache_key)
        if cached is not None:
            return LLMResult(**cached, cache_hit=True)

        def _call():
            return self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body=self._extra_body or None,
            )

        resp = self._with_retries(_call)
        text = resp.choices[0].message.content or ""
        usage = getattr(resp, "usage", None)
        result_payload = {
            "text": text,
            "parsed": None,
            "input_tokens": getattr(usage, "prompt_tokens", 0) or 0,
            "output_tokens": getattr(usage, "completion_tokens", 0) or 0,
            "model": model,
        }
        self._cache_set(cache_key, result_payload)
        return LLMResult(**result_payload, cache_hit=False)

    def chat_parsed(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        response_format: Type[Any],
        temperature: float = 0.3,
        max_tokens: int = 1024,
        cache_namespace: str = "parsed",
    ) -> LLMResult:
        """Structured-output call (used by the reformulator).

        Uses `chat.completions.parse` when available so the JSON shape is
        guaranteed by the SDK; falls back to JSON-mode parse-via-pydantic on
        older clients.
        """
        schema_name = getattr(response_format, "__name__", str(response_format))
        cache_key = _stable_key(
            {
                "ns": cache_namespace,
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "schema": schema_name,
            }
        )
        cached = self._cache_get(cache_key)
        if cached is not None:
            parsed = (
                response_format.model_validate_json(cached["parsed_json"])
                if cached.get("parsed_json")
                else None
            )
            return LLMResult(
                text=cached["text"],
                parsed=parsed,
                input_tokens=cached["input_tokens"],
                output_tokens=cached["output_tokens"],
                model=cached["model"],
                cache_hit=True,
            )

        def _call_parse():
            return self.client.beta.chat.completions.parse(
                model=model,
                messages=messages,
                response_format=response_format,
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body=self._extra_body or None,
            )

        def _call_json_fallback():
            return self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
                extra_body=self._extra_body or None,
            )

        try:
            resp = self._with_retries(_call_parse)
            msg = resp.choices[0].message
            parsed_obj = getattr(msg, "parsed", None)
            text = msg.content or (
                json.dumps(parsed_obj.model_dump()) if parsed_obj is not None else ""
            )
        except Exception:
            # Older SDK or provider does not support `.parse`; degrade to
            # json-mode and reconstruct.
            resp = self._with_retries(_call_json_fallback)
            text = resp.choices[0].message.content or ""
            parsed_obj = response_format.model_validate_json(text)

        usage = getattr(resp, "usage", None)
        result_payload = {
            "text": text,
            "parsed_json": parsed_obj.model_dump_json() if parsed_obj is not None else None,
            "input_tokens": getattr(usage, "prompt_tokens", 0) or 0,
            "output_tokens": getattr(usage, "completion_tokens", 0) or 0,
            "model": model,
        }
        self._cache_set(cache_key, result_payload)
        return LLMResult(
            text=text,
            parsed=parsed_obj,
            input_tokens=result_payload["input_tokens"],
            output_tokens=result_payload["output_tokens"],
            model=model,
            cache_hit=False,
        )
