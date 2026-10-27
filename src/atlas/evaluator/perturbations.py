"""Perturbations: paraphrase, description rewrite/truncate, shuffle, rename."""

from __future__ import annotations

import copy
import random
import re

SYNONYMS = {
    "what is": "tell me",
    "what's": "tell me",
    "get": "fetch",
    "show": "display",
    "find": "locate",
    "please": "",
    "can you": "could you",
    "how": "how",
    "weather in": "weather for",
}


def paraphrase_prompt(prompt: str, seed: int = 0) -> str:
    rnd = random.Random(seed)
    out = prompt
    keys = list(SYNONYMS.keys())
    rnd.shuffle(keys)
    for k in keys[:2]:
        out = re.sub(re.escape(k), SYNONYMS[k], out, flags=re.IGNORECASE)
    # Change punctuation/word order slightly
    if rnd.random() > 0.5 and "?" in out:
        out = out.replace("?", " please.")
    return out.strip()


def truncate_descriptions(case, max_chars: int = 60):
    new_case = copy.deepcopy(case)
    for t in new_case.tools:
        if len(t.description) > max_chars:
            t.description = t.description[:max_chars]
    return new_case


def rewrite_descriptions(case, seed: int = 0):
    # Simple rewrite: shorten + synonym swap
    new_case = copy.deepcopy(case)
    rnd = random.Random(seed)
    for t in new_case.tools:
        t.description = paraphrase_prompt(t.description, seed + len(t.description))
    return new_case


def shuffle_tools(case, seed: int = 0):
    new_case = copy.deepcopy(case)
    rnd = random.Random(seed)
    rnd.shuffle(new_case.tools)
    return new_case


def rename_params(case, mapping: dict[str, str]):
    """Rename param keys consistently in tools + expected_calls."""
    new_case = copy.deepcopy(case)
    for t in new_case.tools:
        props = t.parameters.get("properties", {})
        new_props = {}
        for k, v in props.items():
            nk = mapping.get(k, k)
            new_props[nk] = v
        t.parameters = dict(t.parameters)
        t.parameters["properties"] = new_props
        if "required" in t.parameters:
            t.parameters["required"] = [mapping.get(r, r) for r in t.parameters["required"]]
    for e in new_case.expected_calls:
        e.arguments = {mapping.get(k, k): v for k, v in e.arguments.items()}
    # alternative sequences
    new_alts = []
    for seq in new_case.alternative_sequences or []:
        new_seq = []
        for exp in seq:
            if isinstance(exp, dict):
                renamed_arguments = {
                    mapping.get(key, key): value
                    for key, value in exp.get("arguments", {}).items()
                }
                new_seq.append({
                    "name": exp.get("name"),
                    "arguments": renamed_arguments,
                })
            else:
                renamed_arguments = {
                    mapping.get(key, key): value
                    for key, value in exp.arguments.items()
                }
                new_seq.append(type(exp)(exp.name, renamed_arguments))
        new_alts.append(new_seq)
    new_case.alternative_sequences = new_alts
    return new_case
