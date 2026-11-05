# Getting started

## Requirements

- Python 3.10+
- A model provider (Ollama by default) — or use `--provider dummy` for smoke tests

## Install

```bash
git clone https://github.com/Nonsense0909688/atlas.git
cd atlas
pip install -e ".[dev]"
atlas --help
```

Set a default model to skip repeating `-m`:

```bash
export ATLAS_MODEL=qwen3.5:2b   # Windows: $env:ATLAS_MODEL = "qwen3.5:2b"
```

## Check your setup

```bash
atlas doctor                     # Python, built-in cases, provider reachability
atlas models                     # models installed on the provider
ollama pull qwen3.5:2b           # if Ollama says the model is missing
```

## First run

```bash
atlas run --model qwen3.5:2b --dry-run   # list the 32 cases, no model calls
atlas run --model qwen3.5:2b             # full benchmark, terminal report
atlas run --model qwen3.5:2b --case weather-single --verbose
```

![run demo](assets/demo-run.gif)

No local model handy? The deterministic `dummy` provider returns no tool
calls, so runs complete instantly (and mostly fail) — handy for checking
the harness itself:

```bash
atlas run --model dummy --provider dummy --dry-run
```

## Outputs

```bash
atlas run -m qwen3.5:2b -o report.json     # JSON (format inferred from extension)
atlas run -m qwen3.5:2b -o report.html     # HTML
atlas run -m qwen3.5:2b --format json | jq .summary.overall
atlas history                              # list saved reports in a directory
```

## Next steps

- Compare several models at once: [Batch & leaderboards](batch.md)
- Browse cases: `atlas list`, `atlas show weather-single`
- Full flag list: [CLI reference](cli-reference.md)
