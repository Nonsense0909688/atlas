"""Google Gemini provider."""

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


class GeminiProvider(BaseProvider):
    name = "gemini"

    def __init__(self, api_key: str | None = None, base_url: str = "https://generativelanguage.googleapis.com/v1beta", timeout: int = 120) -> None:
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or ""
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def list_models(self) -> list:
        try:
            resp = requests.get(f"{self.base_url}/models", params={"key": self.api_key}, timeout=self.timeout)
            resp.raise_for_status()
            return resp.json().get("models", [])
        except Exception as e:
            raise ProviderError(str(e)) from e

    def chat(self, model: ModelConfig, input: str, tools: list[ToolDefinition] | None = None,
             stream: bool = False) -> ChatResponse:
        return self.chat_messages(model, [{"role": "user", "content": input}], tools=tools)

    def chat_messages(self, model: ModelConfig, messages: list[dict], tools: list[ToolDefinition] | None = None) -> ChatResponse:
        contents = []
        for m in messages:
            role = "model" if m.get("role") == "assistant" else "user"
            contents.append({"role": role, "parts": [{"text": m.get("content", "")}]})
        payload: dict[str, Any] = {"contents": contents}
        if tools:
            payload["tools"] = [{"functionDeclarations": [
                {"name": t.name, "description": t.description, "parameters": t.parameters} for t in tools
            ]}]
        t0 = time.monotonic()
        try:
            resp = requests.post(f"{self.base_url}/models/{model.id}:generateContent",
                                 params={"key": self.api_key}, json=payload, timeout=model.timeout)
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.Timeout as e:
            raise ProviderTimeoutError(str(e)) from e
        except requests.ConnectionError as e:
            raise ProviderUnavailableError(str(e)) from e
        except requests.HTTPError as e:
            raise ProviderError(str(e)) from e
        latency_ms = (time.monotonic() - t0) * 1000
        text = ""
        tool_calls = []
        candidates = data.get("candidates", []) if isinstance(data, dict) else []
        parts = candidates[0].get("content", {}).get("parts", []) if candidates else []
        for part in parts:
            if "text" in part:
                text += part["text"]
            if "functionCall" in part:
                fc = part["functionCall"]
                args = fc.get("args", {})
                tool_calls.append({"name": fc.get("name"), "arguments": args, "raw": part, "parse_error": None})
        usage = data.get("usageMetadata", {}) if isinstance(data, dict) else {}
        return ChatResponse(raw=data, tool_calls=tool_calls, text=text, latency_ms=latency_ms,
                            prompt_tokens=usage.get("promptTokenCount"), completion_tokens=usage.get("candidatesTokenCount"),
                            total_tokens=usage.get("totalTokenCount"))
