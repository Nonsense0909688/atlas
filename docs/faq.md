# FAQ

**Do tools actually run?**
No. Atlas runs in a mock sandbox — tool outputs come from each case's
`mock_outputs` (or the closed-loop `StatefulToolbox`). No side effects.

**Which model should I start with?**
Any Ollama model (`ollama pull qwen3.5:2b`). Use `--provider dummy` to
smoke-test the harness with no model at all.

**How do I compare three models?**
`atlas batch --models a,b,c` — see [Batch & leaderboards](batch.md).

**Why did a high-scoring case fail?**
Safety and injection gates are hard requirements (score must be 1.00),
and structure validity must be perfect. The report's `error_analysis`
explains each failure.

**How do I run only hard cases / one category?**
`--difficulty hard`, `--category multi_tool`, or `--case <id>`.

**What is suite v2 / `--suite full`?**
230 cases (32 builtin + 198 domain cases) with `train`/`held_out`
splits and multilingual filtering — see [Custom suites](custom-suites.md).

**How do I add a new provider / tool / case?**
See [Architecture](architecture.md) and the "Extending" notes in each
guide. Providers implement `BaseProvider.chat()`; cases are `TestCase`
entries or a custom JSON suite.

**The GIFs look stale — how do I regenerate them?**
`python scripts/make_demos.py` (needs Pillow). See `docs/README.md`.
