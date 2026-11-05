# CI & regression

## Gates

```bash
atlas run -m qwen3.5:2b --fail-under 0.85
atlas run -m qwen3.5:2b -o new.json --baseline old.json --fail-on-regression
atlas diff old.json new.json --fail-on-regression
atlas batch --models a,b --fail-under 0.80   # gates on the best model
```

Exit code `1` on a failed gate, so CI fails without extra scripting.

## GitHub Actions

`.github/workflows/atlas.yml` is a ready-made example: it runs the full
suite, uploads `report.json` as an artifact, and (on PRs) can diff
against a saved `baseline.json`. Adjust the `model` / `provider` /
`fail_under` inputs to your models, and set provider keys as secrets
(`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`).

## Keeping a leaderboard fresh

```bash
atlas batch --models-file models.txt --suite full \
  --format markdown -o leaderboard.md
```

Commit the regenerated table, or let a scheduled workflow open a PR
with it. See [Batch & leaderboards](batch.md).
