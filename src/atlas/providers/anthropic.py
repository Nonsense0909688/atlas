"""Anthropic provider."""

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


class AnthropicProvider(BaseProvider):
    name = "anthropic"

    def __init__(self, api_key: str | None = None, base_url: str = "https://api.anthropic.com", timeout: int = 120) -> None:
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY") or ""
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def list_models(self) -> list:
        # Anthropic has no list-models endpoint; return known models.
        return [{"id": "claude-3-5-sonnet-latest"}, {"id": "claude-3-opus-latest"}]

    def chat(self, model: ModelConfig, input: str, tools: list[ToolDefinition] | None = None,
             stream: bool = False) -> ChatResponse:
        return self.chat_messages(model, [{"role": "user", "content": input}], tools=tools)

    def chat_messages(self, model: ModelConfig, messages: list[dict], tools: list[ToolDefinition] | None = None) -> ChatResponse:
        # Anthropic expects messages with system separate; simplify: extract.
        system = None
        msgs = []
        for m in messages:
            if m.get("role") == "system":
                system = m.get("content")
            elif m.get("role") == "tool":
                msgs.append({"role": "user", "content": f"[tool result for {m.get('name','')}]: {m.get('content')}"})
            else:
                msgs.append({"role": m.get("role", "user"), "content": m.get("content", "")})
        payload: dict[str, Any] = {
            "model": model.id,
            "messages": msgs,
            "max_tokens": model.max_tokens,
            "temperature": model.temperature,
        }
        if system:
            payload["system"] = system
        if tools:
            payload["tools"] = [{"name": t.name, "description": t.description, "input_schema": t.parameters} for t in tools]
        headers = {"x-api-key": self.api_key, "anthropic-version": "2023-06-01", "Content-Type": "application/json"}
        t0 = time.monotonic()
        try:
            resp = requests.post(f"{self.base_url}/v1/messages", json=payload, headers=headers, timeout=model.timeout)
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
        content = data.get("content", []) if isinstance(data, dict) else []
        for block in content:
            if block.get("type") == "text":
                text += block.get("text", "")
            elif block.get("type") == "tool_use":
                args = block.get("input")
                if isinstance(args, dict):
                    tool_calls.append({"name": block.get("name"), "arguments": args, "raw": block, "parse_error": None})
                else:
                    tool_calls.append({"name": block.get("name"), "arguments": args, "raw": block, "parse_error": "invalid args"})
        usage = data.get("usage", {}) if isinstance(data, dict) else {}
        return ChatResponse(raw=data, tool_calls=tool_calls, text=text, latency_ms=latency_ms,
                            prompt_tokens=usage.get("input_tokens"), completion_tokens=usage.get("output_tokens"),
                            total_tokens=(usage.get("input_tokens", 0) or 0) + (usage.get("output_tokens", 0) or 0) if usage else None)
