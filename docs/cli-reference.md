# CLI reference

`atlas` has one global flag (`--no-color`, plus `--version`) and 10
subcommands. Bare `atlas`, `atlas --help`, and `atlas help` all show the
grouped top-level overview (EVALUATE / BROWSE & COMPARE / EXTEND / SETUP,
with example invocations under each group — heaven-style). For one
command's grouped options, examples, and exit codes:

```bash
atlas help run          # same as: atlas run --help
atlas help batch
atlas --help run
```

## `atlas run` — benchmark one model

```
atlas run [-m MODEL] [--provider PROVIDER] [selection] [generation]
          [robustness] [-o OUTPUT] [--format text|json|html]
          [--dry-run] [-v | -q] [CI gates]
```

| Flag | Default | Meaning |
|---|---|---|
| `-m, --model` | `$ATLAS_MODEL` | Model ID, e.g. `qwen3.5:2b` |
| `--provider` | `ollama` | `ollama openai anthropic gemini openrouter lmstudio vllm llama-cpp openai-compatible dummy` |
| `--suite` | builtin (32) | Custom suite JSON, or `full` for all 230 cases |
| `--case` | — | Run one case ID only |
| `--category` | — | Run one category only |
| `--difficulty` | — | `easy medium hard expert` |
| `--split` | — | `train held_out all` |
| `--language` | — | Filter by language (`en es fr de hi pt it` in suite v2) |
| `--temp` | `0.2` | Sampling temperature (0–2) |
| `--max-tokens` | `2048` | Max completion tokens |
| `--retries` | `1` | Run each case N times (consistency, pass@k) |
| `--calling-mode` | `native` | `native` tool calls, or `prompt` (JSON-in-text) |
| `--closed-loop` | off | Force multi-turn mock-tool loop |
| `--api-base/--api-key` | env fallback | Provider endpoint / key |
| `--scale-tools N` | — | Pad each case to N tools |
| `--shuffle-tools` | — | Shuffle tool order |
| `--truncate-desc CHARS` | — | Truncate tool descriptions |
| `--paraphrase` | — | Paraphrase prompts |
| `--rename-params JSON` | — | Rename params, e.g. `{"city":"town"}` |
| `-o, --output` | stdout | Report file (format inferred from `.json`/`.html`) |
| `--dry-run` | — | Show selected cases, make no model calls |
| `-v, --verbose` | — | Per-call expected/actual/score detail |
| `-q, --quiet` | — | Failures only |
| `--fail-under SCORE` | — | Exit 1 if overall < SCORE |
| `--baseline REPORT.json` | — | Compare against a previous report |
| `--fail-on-regression` | — | Exit 1 if a case regressed pass→fail |

## `atlas batch` — many models, one leaderboard

See [Batch & leaderboards](batch.md).

```
atlas batch --models a,b,c | --models-file models.txt
            [--provider ...] [same selection/generation flags as run]
            [-o out.json|out.html|out.md] [--format text|json|html|markdown]
            [--dry-run] [--fail-under SCORE]
```

## `atlas list` / `atlas show` — browse cases

```
atlas list [--suite FILE] [--category C] [--difficulty D] [-s SEARCH] [--json]
atlas show CASE_ID [--suite FILE]     # full ID or unique prefix
```

![list demo](assets/demo-list.gif)

## `atlas models`

```
atlas models [--provider PROVIDER]    # list installed models + sizes
```

## `atlas diff` / `atlas history`

```
atlas diff REPORT1.json REPORT2.json [--fail-on-regression]
atlas history [--dir .] [--model ID] [-n LIMIT]
```

## `atlas import` — bring your own data/tools

```
atlas import bfcl|tau-bench|toolbench FILE.jsonl -o suite.json
atlas import openapi SPEC.yaml -o suite.json
atlas import mcp http://localhost:3000/mcp -o suite.json
```

## `atlas generate` — draft new cases for human review

```
atlas generate -m qwen3.5:2b -n 10 --domain calendar -o review_queue.json
```

## `atlas doctor`

```
atlas doctor [--provider ...] [--api-base ...] [--api-key ...]
```

One-screen health check: Python version, built-in case count, provider
reachability, installed models.

## Environment variables

| Variable | Meaning |
|---|---|
| `ATLAS_MODEL` | Default for `--model` |
| `NO_COLOR` | Disable coloured output (also `--no-color`) |
| `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY` | Provider keys |

Exit codes: `0` ok · `1` failure (incl. failed CI gates) · `2` config error ·
`130` interrupted.
