# Architecture

Standard Python `src/` layout. All installable code lives in
`src/atlas/`; top-level `cli`/`runner` modules are compatibility shims.

```text
atlas/
├── src/
│   └── atlas/
│       ├── __init__.py
│       ├── cli.py / runner.py       # compatibility entry points
│       ├── models.py                # TestCase, ScoreBreakdown, EvalResult, …
│       ├── exceptions.py            # ConfigError, ProviderError
│       ├── logging_config.py
│       ├── atlas_cli/               # CLI commands + terminal UI
│       │   ├── app.py               # dispatcher, exit-code policy
│       │   ├── parser.py            # argparse construction
│       │   ├── run.py / batch.py    # `atlas run` / `atlas batch`
│       │   └── terminal.py          # colours, LiveReporter
│       ├── atlas_runner/            # execution engine
│       │   ├── execution.py         # run_case, execute_run
│       │   ├── batch.py             # model parsing, leaderboard, formats
│       │   ├── case_loading.py      # suite loading + filtering
│       │   ├── parsing.py           # response → tool calls
│       │   ├── providers.py         # make_provider, DummyProvider
│       │   └── reporting.py         # text / JSON / HTML reports
│       ├── evaluator/               # scoring, safety, validation, matchers,
│       │                            # reliability, scaling, perturbations
│       ├── providers/               # ollama, openai-compatible, anthropic, gemini
│       ├── tools/                   # schemas, sandbox, OpenAPI/MCP importers, generator
│       ├── cases/                   # builtin.py (32) + domains.py (198)
│       └── datasets/                # bfcl / tau-bench / toolbench adapters
├── tests/                           # pytest suite (100 tests)
├── scripts/make_demos.py            # renders docs/assets/*.gif
├── docs/                            # these docs
└── pyproject.toml
```

Where to start:

- **CLI behaviour** → `atlas_cli/app.py`, then the per-command module.
- **Execution** → `atlas_runner/execution.py`.
- **Batch** → `atlas_runner/batch.py` (pure logic) + `atlas_cli/batch.py`.
- **Scoring** → `evaluator/scoring.py` and `evaluator/safety.py`.
- **New provider** → implement `providers/base.py`, register in
  `atlas_runner/providers.py`.

Run the tests with `python -m pytest`.
