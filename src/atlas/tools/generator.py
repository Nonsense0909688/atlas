"""LLM-assisted case generator with human-review queue."""

from __future__ import annotations

import json
from pathlib import Path

from atlas.providers.base import BaseProvider


GENERATE_PROMPT = """You are generating an Atlas tool-call test case.
Domain: {domain}
Difficulty: {difficulty}

Return ONLY JSON: {{"id": "...", "prompt": "...", "tools": [{{"name": "...", "description": "...", "parameters": {{...}}}}], "expected_calls": [{{"name": "...", "arguments": {{...}}}}], "category": "{domain}", "difficulty": "{difficulty}"}}
"""


def generate_cases_llm(provider: BaseProvider, model, count: int = 5, domain: str = "calendar",
                       difficulty: str = "medium", seed: int = 0) -> list[dict]:
    """Ask LLM to generate case JSONs. Returns parsed dicts (empty on failure)."""
    cases: list[dict] = []
    for i in range(count):
        prompt = GENERATE_PROMPT.format(domain=domain, difficulty=difficulty)
        try:
            resp = provider.chat(model, prompt, tools=[])
            text = resp.text.strip()
            # strip code fences
            if "```" in text:
                text = text.split("```")[1].strip()
                if text.startswith("json"):
                    text = text[4:].strip()
            data = json.loads(text)
            # Ensure keys
            data.setdefault("id", f"gen-{domain}-{seed}-{i}")
            cases.append(data)
        except Exception:
            continue
    return cases


def save_review_queue(cases: list[dict], path: str = "review_queue.json") -> None:
    """Write generated cases for human review."""
    out = Path(path)
    existing = []
    if out.exists():
        try:
            existing = json.loads(out.read_text(encoding="utf-8"))
        except Exception:
            existing = []
    existing.extend(cases)
    out.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")
