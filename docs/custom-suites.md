# Custom suites

Create a JSON array of cases:

```json
[
  {
    "id": "my-case",
    "prompt": "What's the weather in Tokyo?",
    "tools": [
      {"name": "get_weather", "description": "...", "parameters": {...}}
    ],
    "expected_calls": [
      {"name": "get_weather", "arguments": {"city": "Tokyo"}}
    ],
    "mock_outputs": {"get_weather": {"temp_c": 22}},
    "category": "custom",
    "difficulty": "easy"
  }
]
```

```bash
atlas run --model qwen3.5:2b --suite custom.json
atlas list --suite custom.json
atlas show my-case --suite custom.json
```

## Matchers

Per-argument matchers in the case:

```json
"matchers": {"amount": {"type": "numeric", "tolerance": 0.01}}
```

Types: `exact`, `regex`, `numeric`, `date`, `set`, `any_of`, `llm_judge`,
`fuzzy`. `alternative_sequences` lets several call orders pass — the
scorer picks the best tool-overlap/argument match without mutating
the case.

## Suite v2 (`--suite full`, 230 cases)

32 builtin + 198 extended (`src/atlas/cases/domains.py`): calendar,
finance, SQL, file ops, e-commerce, code tools, multilingual
(es/fr/de/hi/pt/it), safety/injection/recovery/state/matcher coverage.
`SUITE_VERSION=2.0.0`; use `--split train|held_out` for held-out
evaluation and `--language` to filter.

## Importing existing data

```bash
atlas import bfcl|tau-bench|toolbench file.jsonl -o suite.json
atlas import openapi spec.yaml -o suite.json
atlas import mcp http://localhost:3000/mcp -o suite.json
```

## Generating cases for review

```bash
atlas generate -m qwen3.5:2b -n 10 --domain calendar -o review_queue.json
```

Generated cases land in a review queue — a human edits/approves them
before they join a suite.
