"""Anthropic LLM client — Anthropic API gateway or direct Anthropic.

Auto-selects backend:
  Anthropic API (internal):
    Fetches USSO token via `usso -ussh genai-api -print`
    Uses https://genai-api.example-gateway.com/ as base URL
    Sends token as Authorization: Bearer (auth_token parameter)
    Compatible models: claude-haiku-4-5, claude-sonnet-4-6, claude-opus-4-6, etc.

  Direct Anthropic:
    Set ANTHROPIC_API_KEY to fall back to direct access.

Usage:
    from evolvetool_bench.harness.llm_client import LLMClient
    client = LLMClient(model="claude-haiku-4-5")
    response = await client.complete("user prompt here")
    # response.text, response.input_tokens, response.output_tokens
"""
from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any

import httpx
from anthropic import AsyncAnthropic, APIError, RateLimitError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

_UBER_BASE_URL = os.environ.get("ANTHROPIC_BASE_URL", "https://genai-api.example-gateway.com/")
_TOKEN_CACHE: dict[str, tuple[str, float]] = {}   # key -> (token, expires_ts)
_TOKEN_TTL = 60 * 60  # refresh if < 1h remaining, valid ~20h


def _fetch_usso_token() -> str | None:
    """Fetch USSO token for Anthropic API, with simple in-process caching."""
    now = time.time()
    cached = _TOKEN_CACHE.get("genai-api")
    if cached and cached[1] > now + 60:
        return cached[0]
    # Check env first
    env_token = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    if env_token:
        _TOKEN_CACHE["genai-api"] = (env_token, now + _TOKEN_TTL)
        return env_token
    # Fetch via usso CLI
    try:
        out = subprocess.check_output(
            ["usso", "-ussh", "genai-api", "-print"],
            text=True, timeout=30, stderr=subprocess.DEVNULL,
        )
        token = out.strip().split("\n")[-1].strip()
        if token and len(token) > 20:
            _TOKEN_CACHE["genai-api"] = (token, now + _TOKEN_TTL)
            return token
    except Exception:
        pass
    return None


def _build_uber_client() -> AsyncAnthropic:
    token = _fetch_usso_token()
    if not token:
        raise RuntimeError("Cannot fetch Anthropic API USSO token. Is usso CLI available?")
    return AsyncAnthropic(
        base_url=_UBER_BASE_URL,
        auth_token=token,
        api_key="uber-internal",
        max_retries=0,
        http_client=httpx.AsyncClient(
            base_url=_UBER_BASE_URL,
            timeout=httpx.Timeout(120.0),
        ),
    )


def _use_uber() -> bool:
    if os.environ.get("ANTHROPIC_DIRECT", "").lower() in ("1", "true"):
        return False
    if os.environ.get("ANTHROPIC_API_KEY"):
        return False
    return True


@dataclass
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int
    raw_response: Any = field(default=None, repr=False)


@dataclass
class LLMClient:
    """Simple async Anthropic client with Anthropic API auto-detection and retry."""

    model: str = "claude-haiku-4-5"
    max_retries: int = 3
    temperature: float = 0.0
    max_tokens: int = 4096

    def __post_init__(self) -> None:
        self._client: AsyncAnthropic | None = None
        self._is_uber = False

    def _ensure_client(self) -> AsyncAnthropic:
        if self._client is None:
            if _use_uber():
                try:
                    self._client = _build_uber_client()
                    self._is_uber = True
                except Exception as e:
                    print(f"[llm_client] Anthropic API unavailable ({e}), trying direct Anthropic")
                    self._client = AsyncAnthropic(max_retries=0)
                    self._is_uber = False
            else:
                self._client = AsyncAnthropic(max_retries=0)
                self._is_uber = False
        return self._client

    async def complete(
        self,
        prompt: str,
        *,
        system: str = "You are a helpful assistant.",
        tools: list[dict] | None = None,
    ) -> LLMResponse:
        """Single-turn completion, returning the assistant's text."""
        return await self._call_with_retry(
            messages=[{"role": "user", "content": prompt}],
            system=system,
            tools=tools,
        )

    async def chat(
        self,
        messages: list[dict],
        *,
        system: str = "You are a helpful assistant.",
        tools: list[dict] | None = None,
    ) -> LLMResponse:
        """Multi-turn completion."""
        return await self._call_with_retry(messages=messages, system=system, tools=tools)

    async def _call_with_retry(
        self,
        messages: list[dict],
        *,
        system: str,
        tools: list[dict] | None,
    ) -> LLMResponse:
        client = self._ensure_client()

        @retry(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=2, min=2, max=20),
            retry=retry_if_exception_type((APIError, RateLimitError)),
            reraise=True,
        )
        async def _do() -> LLMResponse:
            # Refresh token if using Uber and cache may be stale
            nonlocal client
            if self._is_uber:
                fresh = _fetch_usso_token()
                if fresh:
                    client = _build_uber_client()
                    self._client = client

            kwargs: dict[str, Any] = {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "system": system,
                "messages": messages,
            }
            if tools:
                kwargs["tools"] = tools

            resp = await client.messages.create(**kwargs)
            text = "".join(
                b.text for b in resp.content if getattr(b, "type", None) == "text"
            )
            return LLMResponse(
                text=text,
                input_tokens=resp.usage.input_tokens,
                output_tokens=resp.usage.output_tokens,
                raw_response=resp,
            )

        return await _do()
