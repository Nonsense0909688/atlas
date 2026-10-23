"""Provider interface."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from atlas.models import ModelConfig, TestCase

logger = logging.getLogger(__name__)


@dataclass
class ChatResponse:
    raw: Any = None
    tool_calls: list[dict] = field(default_factory=list)
    text: str = ""
    latency_ms: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


class BaseProvider(ABC):
    name: str = "base"

    @abstractmethod
    def chat(self, model: ModelConfig, input: str, tools: list | None = None, stream: bool = False) -> ChatResponse:
        ...

    def chat_messages(self, model: ModelConfig, messages: list[dict], tools: list | None = None) -> ChatResponse:
        """Send a conversation while preserving history for simple providers.

        Concrete adapters should override this method to use their native
        message format. The fallback serializes every message as a transcript;
        it is less capable than native role-aware history, but never silently
        drops earlier turns.
        """
        transcript = "\n\n".join(
            f"{str(message.get('role', 'user')).upper()}: "
            f"{message.get('content', '')}"
            for message in messages
        )
        return self.chat(model, transcript, tools=tools)

    def chat_case(self, model: ModelConfig, case: TestCase) -> ChatResponse:
        logger.debug("chat_case case=%s prompt=%r", case.id, case.prompt[:80])
        return self.chat(model, case.prompt, tools=case.tools)

    def list_models(self) -> list:
        raise NotImplementedError(f"{self.name} does not support list_models")
