<div align="center">

# Atlas

**A CLI benchmark for measuring how accurately models produce structured tool calls.**

Runs models against a suite of test cases in a **mock sandbox** (no real tool
execution, no side effects), then scores tool selection, argument accuracy,
structure validity, precision/recall, hallucination rate, safety, and more.

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Version 0.4.1](https://img.shields.io/badge/version-0.4.1-orange.svg)](pyproject.toml)
[![Stars](https://img.shields.io/github/stars/Nonsense0909688/atlas?style=social)](https://github.com/Nonsense0909688/atlas/stargazers)
[![Forks](https://img.shields.io/github/forks/Nonsense0909688/atlas?style=social)](https://github.com/Nonsense0909688/atlas/network/members)

[Quickstart](#quickstart) · [Demo](#demo) · [Batch & Leaderboards](#batch--leaderboards) · [Docs](docs/README.md) · [Scoring](#scoring) · [Star History](#star-history)

</div>

## Contents

- [Features](#features)
- [Demo](#demo)
- [Quickstart](#quickstart)
- [Batch & Leaderboards](#batch--leaderboards)
- [Scoring](#scoring)
- [Built-in Cases (32) + Full Suite (230)](#built-in-cases)
- [Star History](#star-history)
- [Documentation](#documentation)
- [Architecture](#architecture)
- [Contributing](#contributing)
- [License](#license)

## Features

|                            |                                                                                                                                                                                          |
| -------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 🎯 **13 scoring axes**     | Tool selection, argument accuracy, structure, multi-tool planning, precision/recall, hallucination, abstention, consistency, safety, overcall, injection resistance, final answer, state |
| 🏆 **Batch leaderboards**  | `atlas batch` — many models, one suite, ranked text/JSON/HTML/Markdown output                                                                                                            |
| 🔁 **Closed-loop agents**  | Multi-turn mock-tool execution with error injection and recovery grading                                                                                                                 |
| 🛡️ **Safety & injection**  | Confirmation, PII, forbidden-tool, and prompt-injection checks as hard pass gates                                                                                                        |
| 📊 **Reliability stats**   | `pass@k` / `pass^k`, Wilson & bootstrap confidence intervals                                                                                                                             |
| 🎛️ **Robustness controls** | Tool-count scaling, shuffling, truncation, paraphrasing, param renaming                                                                                                                  |
| 🔌 **9 providers**         | Ollama, OpenAI, Anthropic, Gemini, OpenRouter, LM Studio, vLLM, llama.cpp, dummy                                                                                                         |
| 📥 **Importers**           | BFCL, tau-bench, ToolBench, OpenAPI specs, MCP servers                                                                                                                                   |
| ✅ **CI gates**            | `--fail-under`, baselines, and regression diffs with proper exit codes                                                                                                                   |

## Demo

### Single model

![Run demo](docs/assets/demo-run.gif)

```bash
atlas run --model qwen3.5:2b
```

### Batch leaderboard

![Batch demo](docs/assets/demo-batch.gif)

```bash
atlas batch --models qwen3.5:2b,llama3.1:8b -o batch.json
```

### Browse cases

![List demo](docs/assets/demo-list.gif)

```bash
atlas list --difficulty hard
```

> GIFs are scripted terminal sessions rendered by `scripts/make_demos.py`
> (regenerate with `python scripts/make_demos.py`). Scores shown are illustrative.

## Quickstart

```bash
git clone https://github.com/Nonsense0909688/atlas.git
cd atlas
pip install -e ".[dev]"

atlas doctor                                  # check Python + provider setup
atlas models                                  # list installed Ollama models
ollama pull qwen3.5:2b                        # if needed

atlas run --model qwen3.5:2b --dry-run        # preview the 32 cases, no calls
atlas run --model qwen3.5:2b                  # full benchmark
atlas run --model qwen3.5:2b --difficulty hard --verbose
atlas run --model qwen3.5:2b --retries 3 -o report.html
atlas diff report1.json report2.json
```

No local model? Smoke-test the harness with the deterministic provider:

```bash
atlas run --model dummy --provider dummy --dry-run
```

## Batch & Leaderboards

Compare models on the same suite in one command:

```bash
atlas batch --models qwen3.5:2b,llama3.1:8b,mistral:7b
atlas batch --models-file models.txt --suite full --format markdown -o LEADERBOARD.md
atlas batch --models a,b --dry-run
```

Example leaderboard output (illustrative scores):

| Rank | Model         | Provider | Overall | Passed | Tool sel. | Arg acc. | Halluc. |
| ---: | ------------- | -------- | ------: | -----: | --------: | -------: | ------: |
|    1 | `qwen3.5:2b`  | ollama   |    0.91 |  31/32 |      1.00 |     0.89 |    0.00 |
|    2 | `llama3.1:8b` | ollama   |    0.84 |  27/32 |      0.94 |     0.81 |    0.03 |

One bad model never sinks the batch — it shows up as an `ERROR` row and
the run continues. `--fail-under` gates on the best model. Full guide:
[docs/batch.md](docs/batch.md).

## Scoring

Each case is scored on multiple axes (0–1, `None` = n/a). **Overall** is
the mean of applicable primary axes.

| Axis                                                          | What it measures                                                         |
| ------------------------------------------------------------- | ------------------------------------------------------------------------ |
| `tool_selection`                                              | Did the model pick the right tool(s)?                                    |
| `argument_accuracy`                                           | Are arguments correct?                                                   |
| `structure_validity`                                          | Required args present and correctly typed?                               |
| `multi_tool_planning`                                         | Correct count, order, and duplication for multi-call cases               |
| `argument_precision` / `argument_recall`                      | Emitted-vs-expected argument overlap                                     |
| `hallucination_rate`                                          | Fraction of calls inventing tools/params (0 = clean)                     |
| `abstention_accuracy`                                         | Correctly avoided tools when none expected                               |
| `consistency`                                                 | Same valid calls across `--retries` runs                                 |
| `safety_score` / `overcall_score` / `injection_resistance`    | Confirmation/PII/forbidden tools; redundant calls; injected instructions |
| `final_answer_accuracy` / `state_accuracy` / `recovery_score` | Closed-loop final answer, mock state, and error recovery                 |

A case passes only when **all** gates hold: overall ≥ 0.80, tool selection
≥ 0.80, argument accuracy ≥ 0.80 (when applicable), structure = 1.00,
multi-tool ≥ 0.80 (when applicable), hallucination = 0, safety = 1.00,
injection resistance = 1.00. Failed cases include a human-readable
`error_analysis`. Details: [docs/scoring.md](docs/scoring.md).

## Built-in Cases

**32 builtin cases** — 9 easy, 12 medium, 7 hard, 4 expert — across 14
categories: `single_tool`, `complex_args`, `multi_tool`,
`tool_selection`, `abstention`, `required_args`, `optional_args`,
`type_safety`, `hallucination`, `parallel_calls`, `dependency_chain`,
`conversation`, `instruction_conflict`, `edge`.

**Full suite (230 cases):** `--suite full` adds 198 domain cases
(calendar, finance, SQL, file ops, e-commerce, code tools, multilingual,
safety/injection/recovery/state) with `train`/`held_out` splits.

```bash
atlas list
atlas list --difficulty hard
atlas show weather-single
atlas run --model qwen3.5:2b --suite full --split held_out
```

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Nonsense0909688/atlas&type=Date)](https://star-history.com/#Nonsense0909688/atlas&Date)

If Atlas helped you evaluate a model's tool calling, a ⭐ is appreciated!

## Documentation

Full docs live in [`docs/`](docs/README.md):

| Guide                                      | Covers                                              |
| ------------------------------------------ | --------------------------------------------------- |
| [Getting started](docs/getting-started.md) | Install, first run, outputs                         |
| [CLI reference](docs/cli-reference.md)     | Every subcommand and flag                           |
| [Batch & leaderboards](docs/batch.md)      | `atlas batch` in depth                              |
| [Providers](docs/providers.md)             | Ollama, OpenAI-compatible, Anthropic, Gemini, dummy |
| [Scoring](docs/scoring.md)                 | Axes, pass gates, reliability                       |
| [Custom suites](docs/custom-suites.md)     | Suite JSON, matchers, imports, generation           |
| [Robustness & safety](docs/robustness.md)  | Perturbations, injection, closed-loop               |
| [CI & regression](docs/ci.md)              | Gates, baselines, GitHub Actions                    |
| [Architecture](docs/architecture.md)       | `src/` layout and extension points                  |
| [FAQ](docs/faq.md)                         | Common questions                                    |

## Architecture

Standard Python `src/` layout (`src/atlas/`): `atlas_cli/` (commands +
terminal UI), `atlas_runner/` (execution, batch leaderboards, providers,
reports), `evaluator/` (scoring/safety/matchers), `providers/`,
`tools/`, `cases/`, `datasets/`. See
[docs/architecture.md](docs/architecture.md).

```bash
python -m pytest        # 100+ tests
```

## Contributing

Issues and PRs are welcome. Start with `atlas doctor` output, the failing
case ID (if any), and your provider/model. New providers implement
`BaseProvider.chat()`; new cases are `TestCase` entries or a custom JSON
suite — see [docs/architecture.md](docs/architecture.md).

## License

MIT
