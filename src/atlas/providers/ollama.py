"""Ollama provider — talks to a local Ollama server.

Wraps ``POST /api/chat`` and normalizes Ollama's response shape into
``ChatResponse``. Ollama-specific quirks handled here:

- Tools are sent as ``{"type": "function", "function": {...}}`` items.
- Tool calls arrive under ``message.tool_calls[].function``.
- Token usage comes back as ``prompt_eval_count`` / ``eval_count``.
- Sampling options are nested under ``options`` (temperature, num_predict…).

Error handling (production rules):
- Connection refused / DNS → ``ProviderUnavailableError`` (server not running).
- Timeout → ``ProviderTimeoutError``.
- HTTP 4xx/5xx + bad JSON → ``ProviderError``.
Never let raw ``requests`` exceptions leak into the runner.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import requests

from atlas.exceptions import ProviderError, ProviderTimeoutError, ProviderUnavailableError
from atlas.models import ModelConfig, ToolDefinition
from atlas.providers.base import BaseProvider, ChatResponse

logger = logging.getLogger(__name__)


class OllamaProvider(BaseProvider):
    """Ollama backend. Assumes ``ollama serve`` is running locally."""

    name = "ollama"

    def __init__(self, base_url: str = "http://localhost:11434", timeout: int = 120) -> None:
        """
        Args:
            base_url: Ollama server URL (trailing slash stripped).
            timeout: Default request timeout in seconds.
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        logger.debug("OllamaProvider base_url=%s", self.base_url)

    def list_models(self) -> list:
        """GET /api/tags → list of model entries."""
        logger.debug("list_models GET %s/api/tags", self.base_url)
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=self.timeout)
            response.raise_for_status()
        except requests.exceptions.Timeout as e:
            raise ProviderTimeoutError(f"Ollama list_models timed out after {self.timeout}s: {e}") from e
        except requests.ConnectionError as e:
            raise ProviderUnavailableError(
                f"Cannot reach Ollama at {self.base_url}. Is `ollama serve` running? ({e})"
            ) from e
        except requests.HTTPError as e:
            raise ProviderError(f"Ollama list_models HTTP error: {e}") from e
        models = response.json().get("models", [])
        logger.info("list_models: found %d models", len(models))
        return models

    def chat(
        self,
        model: ModelConfig,
        input: str,
        tools: list[ToolDefinition] | None = None,
    ) -> ChatResponse:
        """Send one chat request with tools, return normalized response.

        Args:
            model: Model ID + sampling config + timeout.
            input: User prompt.
            tools: Tools to expose (``None`` = no tools, plain chat).

        Raises:
            ProviderUnavailableError: Server unreachable.
            ProviderTimeoutError: Request exceeded ``model.timeout``.
            ProviderError: HTTP error or unparseable response.
        """
        payload: dict[str, Any] = {
            "model": model.id,
            "messages": [{"role": "user", "content": input}],
            "stream": False,
            "options": {
                "temperature": model.temperature,
                "num_predict": model.max_tokens,
                "top_p": model.top_p,
            },
        }
        if tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    },
                }
                for tool in tools
            ]
        logger.debug("chat model=%s tools=%s prompt=%r",
                     model.id, [t.name for t in tools] if tools else [], input[:80])

        t0 = time.monotonic()
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=model.timeout,
            )
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.Timeout as e:
            raise ProviderTimeoutError(
                f"Ollama chat with model={model.id} timed out after {model.timeout}s"
            ) from e
        except requests.ConnectionError as e:
            raise ProviderUnavailableError(
                f"Cannot reach Ollama at {self.base_url}. Is `ollama serve` running? ({e})"
            ) from e
        except requests.HTTPError as e:
            raise ProviderError(f"Ollama chat HTTP error for model={model.id}: {e}") from e
        except ValueError as e:  # response.json() failed
            raise ProviderError(f"Ollama returned invalid JSON for model={model.id}: {e}") from e
        latency_ms = (time.monotonic() - t0) * 1000

        normalized = self._normalize(data, latency_ms)
        logger.debug("chat model=%s latency=%.0fms tool_calls=%d text_len=%d",
                     model.id, latency_ms, len(normalized.tool_calls), len(normalized.text))
        return normalized

    @staticmethod
    def _coerce_arguments(args: Any) -> tuple[Any, str | None]:
        """Return ``(arguments, parse_error)`` without discarding bad input."""
        if args is None:
            return {}, None  # no arguments sent
        if isinstance(args, dict):
            return args, None
        if isinstance(args, str):
            try:
                decoded = json.loads(args)
            except ValueError as e:
                return args, f"arguments are not valid JSON: {e}"
            if isinstance(decoded, dict):
                return decoded, None
            return args, f"arguments JSON is {type(decoded).__name__}, expected an object"
        return args, f"arguments should be an object, got {type(args).__name__}"

    @staticmethod
    def _normalize(data: dict, latency_ms: float | None) -> ChatResponse:
        """Convert Ollama's raw JSON into ChatResponse.

        Ollama nests everything under ``message``; tool calls use
        ``message.tool_calls[].function.{name,arguments}``. Some models
        return arguments as a JSON string: if it decodes to an object we use
        that. Anything else malformed is passed through UNCHANGED with a
        ``parse_error`` — never rewritten into ``{}`` or ``{"value": …}`` —
        so validation and reports see what the model really sent. Only an
        absent/``null`` ``arguments`` means "no arguments" (``{}``).
        """
        message = data.get("message", {}) if isinstance(data, dict) else {}
        text = message.get("content", "") or ""
        raw_calls = message.get("tool_calls") or []

        tool_calls: list[dict] = []
        for call in raw_calls:
            func = call.get("function", {}) if isinstance(call, dict) else {}
            args = func.get("arguments")
            if args is None:
                args = call.get("arguments")
            args, parse_error = OllamaProvider._coerce_arguments(args)
            tool_calls.append({
                "name": func.get("name") or call.get("name"),
                "arguments": args,
                "raw": call,
                "parse_error": parse_error,
            })

        # Ollama reports token usage as prompt_eval_count / eval_count.
        prompt_tokens = data.get("prompt_eval_count") if isinstance(data, dict) else None
        completion_tokens = data.get("eval_count") if isinstance(data, dict) else None
        total_tokens = (
            prompt_tokens + completion_tokens
            if prompt_tokens is not None and completion_tokens is not None
            else None
        )

        return ChatResponse(
            raw=data,
            tool_calls=tool_calls,
            text=text,
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )
