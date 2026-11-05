# Batch & leaderboards

`atlas batch` runs **one suite against many models** and prints a ranked
leaderboard, so you can compare models without re-running the suite by hand.

![batch demo](assets/demo-batch.gif)

## Quick start

```bash
atlas batch --models qwen3.5:2b,llama3.1:8b
atlas batch --models-file models.txt --suite full -o batch.json
atlas batch --models a,b --dry-run        # show selection, call nothing
```

`models.txt` format — one model ID per line, `#` comments and blanks ignored:

```text
# local models
qwen3.5:2b
llama3.1:8b
mistral:7b
```

`--models` and `--models-file` can be combined; duplicates are dropped,
order is preserved.

## Flags

All `atlas run` selection/generation flags apply (`--suite`, `--case`,
`--category`, `--difficulty`, `--split`, `--language`, `--temp`,
`--retries`, `--calling-mode`, `--closed-loop`, `--api-base/--api-key`),
plus:

| Flag | Meaning |
|---|---|
| `--models a,b` | Comma-separated model IDs |
| `--models-file FILE` | One model per line |
| `-o, --output FILE` | Write the leaderboard (`.json` / `.html` / `.md` inferred) |
| `--format text\|json\|html\|markdown` | Output format |
| `--dry-run` | Print models + selection, make no calls |
| `--fail-under SCORE` | Exit 1 if the *best* model's overall is below SCORE |

One provider per batch (`--provider`, default `ollama`). To compare across
providers, run one batch per provider — or point an OpenAI-compatible
provider at each endpoint in turn.

## Output

Terminal leaderboard (sorted by overall, then pass rate, then name;
errored models sort last):

```text
  Atlas batch — 2 model(s)
  ──────────────────────────────────────────────────────────────────

  #   model                    overall    passed    tool    args    halluc
  1   qwen3.5:2b               0.91 █████████░ 31/32     1.00    0.89    0.00
  2   llama3.1:8b              0.84 ████████░░ 27/32     0.94    0.81    0.03

  Best: qwen3.5:2b (overall 0.91)
```

Markdown (paste straight into a README or PR):

```bash
atlas batch --models a,b --format markdown -o LEADERBOARD.md
```

```markdown
| Rank | Model | Provider | Overall | Passed | Tool sel. | Arg acc. | Halluc. |
|-----:|-------|----------|--------:|-------:|----------:|---------:|--------:|
| 1 | `a` | ollama | 0.91 | 31/32 | 1.00 | 0.89 | 0.00 |
```

JSON contains the leaderboard plus each model's full summary (not the
per-case dumps — use `atlas run -o` for those). HTML is a self-contained
styled table.

## Failure behaviour

A model that errors (provider down, bad ID, …) is recorded as an `ERROR`
row and the batch continues with the next model — one bad model never
sinks the whole leaderboard. The exit code stays `0` unless
`--fail-under` gates on the best score.

## Example: a README leaderboard section

```bash
atlas batch --models qwen3.5:2b,llama3.1:8b,mistral:7b \
  --suite full --format markdown -o leaderboard.md
```

Then include `leaderboard.md`, or paste its table under a
`## Leaderboard` heading. Regenerate it in CI (see [CI & regression](ci.md))
to keep it fresh.
