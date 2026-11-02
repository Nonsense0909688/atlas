"""Shared CLI exit codes and display vocabulary."""

EXIT_OK = 0
EXIT_FAIL = 1
EXIT_CONFIG = 2
EXIT_INTERRUPT = 130
DIFFICULTIES = ["easy", "medium", "hard", "expert"]

PROVIDER_CHOICES = (
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
