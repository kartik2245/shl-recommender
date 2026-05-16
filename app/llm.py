"""Groq client wrapper with timeout + JSON mode + retries."""
from __future__ import annotations

import json
import os
from typing import Any

from groq import Groq
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# llama-3.3-70b-versatile: best Groq quality/speed tradeoff at time of writing.
# llama-3.1-8b-instant: fallback if the 70b is rate-limited.
DEFAULT_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
FALLBACK_MODEL = os.getenv("GROQ_FALLBACK_MODEL", "llama-3.1-8b-instant")

# Per-call timeout. The /chat endpoint has 30s total budget; we may call
# the LLM up to 2-3 times per turn (classify + answer). 8s leaves headroom.
PER_CALL_TIMEOUT = float(os.getenv("LLM_TIMEOUT_S", "8.0"))


class LLMClient:
    def __init__(self, api_key: str | None = None):
        api_key = api_key or os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Export it or put it in .env. "
                "Get a free key at https://console.groq.com/keys."
            )
        self._client = Groq(api_key=api_key, timeout=PER_CALL_TIMEOUT)

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=2),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        json_mode: bool = False,
        temperature: float = 0.2,
        max_tokens: int = 512,
        model: str | None = None,
    ) -> str:
        """Single chat completion. Returns the assistant content string."""
        kwargs: dict[str, Any] = {
            "model": model or DEFAULT_MODEL,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            resp = self._client.chat.completions.create(**kwargs)
        except Exception:
            # Try the smaller model once -- 70b can be rate-limited on free tier.
            if model is None:
                kwargs["model"] = FALLBACK_MODEL
                resp = self._client.chat.completions.create(**kwargs)
            else:
                raise
        return resp.choices[0].message.content or ""

    def chat_json(self, messages: list[dict[str, str]], **kw) -> dict:
        """Chat with JSON mode, parsed. Raises on invalid JSON."""
        raw = self.chat(messages, json_mode=True, **kw)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # Last-resort recovery: pull the first {...} block.
            start = raw.find("{")
            end = raw.rfind("}")
            if start != -1 and end != -1 and end > start:
                return json.loads(raw[start : end + 1])
            raise
