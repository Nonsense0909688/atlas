# Providers

`--provider` selects the model backend (default `ollama`). All adapters
implement `BaseProvider.chat()` (`src/atlas/providers/base.py`); multi-turn
conversations go through `chat_messages()`.

| Provider | What it talks to | Auth |
|---|---|---|
| `ollama` | Local Ollama (`http://localhost:11434`) | none; `atlas models` lists installed models |
| `openai` | OpenAI chat completions | `OPENAI_API_KEY` or `--api-key` |
| `openrouter` | OpenRouter (OpenAI-compatible) | `OPENAI_API_KEY` / `--api-key`, `--api-base` |
| `lmstudio` | LM Studio (`http://localhost:1234/v1`) | none typically |
| `vllm` | vLLM (`http://localhost:8000/v1`) | none typically |
| `llama-cpp` | llama.cpp server (`http://localhost:8080/v1`) | none typically |
| `openai-compatible` | Any OpenAI-compatible endpoint | `--api-base` + `--api-key` |
| `anthropic` | Anthropic Messages API (native) | `ANTHROPIC_API_KEY` |
| `gemini` | Google Gemini (native) | `GEMINI_API_KEY` |
| `dummy` | Deterministic no-call provider (tests/smoke) | none |

```bash
atlas run -m qwen3.5:2b --provider ollama
atlas run -m gpt-4o-mini --provider openai
atlas run -m claude-sonnet-4-5 --provider anthropic
atlas run -m gemini-2.5-flash --provider gemini
atlas run -m local-model --provider lmstudio --api-base http://localhost:1234/v1
atlas run -m dummy --provider dummy            # no calls, harness smoke test
```

## Tool calling modes

- `--calling-mode native` (default): provider-native tool calls.
- `--calling-mode prompt`: tool schemas are injected as text and calls are
  parsed from JSON in the reply — useful for models without native
  function calling.

## Adding a provider

1. Implement `BaseProvider.chat()` returning `ChatResponse` (and
   `chat_messages()` / `list_models()` if supported).
2. Register it in `atlas/atlas_runner/providers.py` (`make_provider`) and
   add its name to `PROVIDER_CHOICES` in `atlas/atlas_cli/constants.py`.
