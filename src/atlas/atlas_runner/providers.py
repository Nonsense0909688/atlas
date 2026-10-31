"""Provider construction and the deterministic dummy provider."""

from __future__ import annotations

from atlas.exceptions import ConfigError
from atlas.providers.base import BaseProvider, ChatResponse
from atlas.providers.ollama import OllamaProvider


KNOWN_PROVIDERS = (
    "ollama",
    "openai",
    "anthropic",
    "gemini",
    "openrouter",
    "lmstudio",
    "vllm",
    "llama-cpp",
    "openai-compatible",
    "dummy",
)


def _make_openai_compatible(
    name: str,
    api_base: str | None,
    api_key: str | None,
) -> BaseProvider:
    """Create an OpenAI-compatible adapter without importing it for other providers."""
    from atlas.providers.openai_compatible import (
        LlamaCppProvider,
        LMStudioProvider,
        OpenAICompatibleProvider,
        OpenAIProvider,
        OpenRouterProvider,
        VLLMProvider,
    )

    if name == "openai":
        return OpenAIProvider(api_key=api_key)
    if name == "openrouter":
        return OpenRouterProvider(api_key=api_key)
    if name == "lmstudio":
        return LMStudioProvider(
            base_url=api_base or "http://localhost:1234/v1"
        )
    if name == "vllm":
        return VLLMProvider(
            base_url=api_base or "http://localhost:8000/v1"
        )
    if name in ("llama-cpp", "llama_cpp"):
        return LlamaCppProvider(
            base_url=api_base or "http://localhost:8080/v1"
        )
    return OpenAICompatibleProvider(base_url=api_base, api_key=api_key)


def make_provider(
    name: str,
    api_base: str | None = None,
    api_key: str | None = None,
) -> BaseProvider:
    """Build a provider adapter from its short name.

    Imports for hosted providers are lazy, so local Ollama use does not need
    to load every provider SDK.

    Raises:
        ConfigError: The requested provider is not supported.
    """
    provider_name = (name or "ollama").lower()

    if provider_name == "ollama":
        return OllamaProvider(base_url=api_base) if api_base else OllamaProvider()
    if provider_name == "dummy":
        return DummyProvider()
    if provider_name in {
        "openai",
        "openai-compatible",
        "openrouter",
        "lmstudio",
        "vllm",
        "llama-cpp",
        "llama_cpp",
        "compatible",
    }:
        return _make_openai_compatible(provider_name, api_base, api_key)
    if provider_name == "anthropic":
        from atlas.providers.anthropic import AnthropicProvider

        return AnthropicProvider(
            api_key=api_key,
            base_url=api_base or "https://api.anthropic.com",
        )
    if provider_name in ("gemini", "google"):
        from atlas.providers.gemini import GeminiProvider

        return GeminiProvider(
            api_key=api_key,
            base_url=api_base or "https://generativelanguage.googleapis.com/v1beta",
        )

    known = ", ".join(KNOWN_PROVIDERS)
    raise ConfigError(f"unknown provider: {name} (known: {known})")


class DummyProvider(BaseProvider):
    """Deterministic provider for smoke tests and CLI diagnostics."""

    name = "dummy"

    def chat(
        self,
        model,
        input: str,
        tools=None,
        stream: bool = False,
    ) -> ChatResponse:
        return ChatResponse(
            raw={"dummy": True},
            tool_calls=[],
            text="",
            latency_ms=1.0,
        )

    def chat_messages(self, model, messages, tools=None) -> ChatResponse:
        return ChatResponse(
            raw={"dummy": True},
            tool_calls=[],
            text="",
            latency_ms=1.0,
        )
