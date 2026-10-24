"""OpenAI-compatible provider: OpenAI, OpenRouter, vLLM, llama.cpp, LM Studio."""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

import requests

from atlas.exceptions import ProviderError, ProviderTimeoutError, ProviderUnavailableError
from atlas.models import ModelConfig, ToolDefinition
from atlas.providers.base import BaseProvider, ChatResponse

logger = logging.getLogger(__name__)


class OpenAICompatibleProvider(BaseProvider):
    name = "openai-compatible"

    def __init__(self, base_url: str | None = None, api_key: str | None = None,
                 timeout: int = 120, extra_headers: dict[str, str] | None = None) -> None:
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENROUTER_API_KEY") or ""
        self.timeout = timeout
        self.extra_headers = extra_headers or {}

    def list_models(self) -> list:
        headers = self._headers()
        try:
            resp = requests.get(f"{self.base_url}/models", headers=headers, timeout=self.timeout)
            resp.raise_for_status()
            return resp.json().get("data", [])
        except requests.exceptions.Timeout as e:
            raise ProviderTimeoutError(f"list_models timed out: {e}") from e
        except requests.ConnectionError as e:
            raise ProviderUnavailableError(f"Cannot reach {self.base_url}: {e}") from e
        except requests.HTTPError as e:
            raise ProviderError(f"list_models HTTP error: {e}") from e

    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        h.update(self.extra_headers)
        return h

    def chat(self, model: ModelConfig, input: str, tools: list[ToolDefinition] | None = None,
             stream: bool = False) -> ChatResponse:
        messages = [{"role": "user", "content": input}]
        return self.chat_messages(model, messages, tools=tools)

    def chat_messages(self, model: ModelConfig, messages: list[dict],
                      tools: list[ToolDefinition] | None = None) -> ChatResponse:
        payload: dict[str, Any] = {
            "model": model.id,
            "messages": messages,
            "temperature": model.temperature,
            "max_tokens": model.max_tokens,
        }
        if tools:
            payload["tools"] = [
                {"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.parameters}}
                for t in tools
            ]
        t0 = time.monotonic()
        try:
            resp = requests.post(f"{self.base_url}/chat/completions", json=payload,
                                 headers=self._headers(), timeout=model.timeout)
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.Timeout as e:
            raise ProviderTimeoutError(f"chat timed out after {model.timeout}s") from e
        except requests.ConnectionError as e:
            raise ProviderUnavailableError(f"Cannot reach {self.base_url}: {e}") from e
        except requests.HTTPError as e:
            raise ProviderError(f"HTTP error for model={model.id}: {e}") from e
        except ValueError as e:
            raise ProviderError(f"invalid JSON: {e}") from e
        latency_ms = (time.monotonic() - t0) * 1000
        return self._normalize(data, latency_ms)

    @staticmethod
    def _coerce(args: Any) -> tuple[Any, str | None]:
        if args is None:
            return {}, None
        if isinstance(args, dict):
            return args, None
        if isinstance(args, str):
            try:
                decoded = json.loads(args)
            except ValueError as e:
                return args, f"arguments are not valid JSON: {e}"
            if isinstance(decoded, dict):
                return decoded, None
            return args, f"arguments JSON is {type(decoded).__name__}, expected object"
        return args, f"arguments should be object, got {type(args).__name__}"

    @staticmethod
    def _normalize(data: dict, latency_ms: float | None) -> ChatResponse:
        choices = data.get("choices", []) if isinstance(data, dict) else []
        msg = choices[0].get("message", {}) if choices else {}
        text = msg.get("content") or ""
        tool_calls: list[dict] = []
        for call in msg.get("tool_calls") or []:
            func = call.get("function", {}) if isinstance(call, dict) else {}
            args = func.get("arguments")
            args, err = OpenAICompatibleProvider._coerce(args)
            tool_calls.append({"name": func.get("name") or call.get("name"), "arguments": args, "raw": call, "parse_error": err})
        usage = data.get("usage", {}) if isinstance(data, dict) else {}
        return ChatResponse(raw=data, tool_calls=tool_calls, text=text, latency_ms=latency_ms,
                            prompt_tokens=usage.get("prompt_tokens"), completion_tokens=usage.get("completion_tokens"),
                            total_tokens=usage.get("total_tokens"))


class OpenAIProvider(OpenAICompatibleProvider):
    name = "openai"
    def __init__(self, api_key: str | None = None, timeout: int = 120) -> None:
        super().__init__(base_url="https://api.openai.com/v1", api_key=api_key or os.environ.get("OPENAI_API_KEY"), timeout=timeout)


class OpenRouterProvider(OpenAICompatibleProvider):
    name = "openrouter"
    def __init__(self, api_key: str | None = None, timeout: int = 120) -> None:
        super().__init__(base_url="https://openrouter.ai/api/v1", api_key=api_key or os.environ.get("OPENROUTER_API_KEY"), timeout=timeout,
                         extra_headers={"HTTP-Referer": "https://github.com/atlas", "X-Title": "Atlas"})


class LMStudioProvider(OpenAICompatibleProvider):
    name = "lmstudio"
    def __init__(self, base_url: str = "http://localhost:1234/v1", timeout: int = 120) -> None:
        super().__init__(base_url=base_url, timeout=timeout)


class VLLMProvider(OpenAICompatibleProvider):
    name = "vllm"
    def __init__(self, base_url: str = "http://localhost:8000/v1", timeout: int = 120) -> None:
        super().__init__(base_url=base_url, timeout=timeout)


class LlamaCppProvider(OpenAICompatibleProvider):
    name = "llama-cpp"
    def __init__(self, base_url: str = "http://localhost:8080/v1", timeout: int = 120) -> None:
        super().__init__(base_url=base_url, timeout=timeout)
