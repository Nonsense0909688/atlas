"""Commands for browsing and inspecting benchmark cases."""

from __future__ import annotations

import argparse
import difflib
import json
import shutil

from atlas.atlas_cli.constants import DIFFICULTIES, EXIT_OK
from atlas.atlas_cli.terminal import (
    _difficulty, _note, _truncate, bold, cyan, dim,
)
from atlas.exceptions import ConfigError
from atlas.atlas_runner.case_loading import load_cases


def _list_cases(args: argparse.Namespace) -> int:
    """``atlas list``: grouped, filterable case browser."""
    all_cases = load_cases(args.suite)
    cases = all_cases
    if args.category:
        cases = [c for c in cases if c.category == args.category]
    if args.difficulty:
        cases = [c for c in cases if c.difficulty == args.difficulty]
    if args.search:
        q = args.search.lower()
        cases = [c for c in cases if q in c.id.lower() or q in c.prompt.lower()]

    if args.json:
        print(json.dumps([{"id": c.id, "category": c.category, "difficulty": c.difficulty,
                           "prompt": c.prompt} for c in cases], indent=2, ensure_ascii=False))
        return EXIT_OK
    if not cases:
        _note("No cases match those filters.")
        if args.category and args.category not in {c.category for c in all_cases}:
            _note("Categories: " + ", ".join(sorted({c.category for c in all_cases})))
        return EXIT_OK

    cols = shutil.get_terminal_size((100, 24)).columns
    idw = min(max(len(c.id) for c in cases), 40)
    room = max(20, cols - idw - 12)
    by_cat: dict[str, list] = {}
    for c in cases:
        by_cat.setdefault(c.category, []).append(c)

    print(f"{bold('Atlas')} — {len(cases)} case{'s' if len(cases) != 1 else ''}")
    for cat in sorted(by_cat):
        print(f"\n{bold(cat)} {dim('(' + str(len(by_cat[cat])) + ')')}")
        for c in by_cat[cat]:
            print(f"  {c.id:<{idw}}  {_difficulty(c.difficulty)}  {dim(_truncate(c.prompt, room))}")

    counts = {d: sum(1 for c in cases if c.difficulty == d) for d in DIFFICULTIES}
    print("\n" + dim("  ".join(f"{d}: {n}" for d, n in counts.items() if n)))
    print(dim("Details: atlas show <case-id>"))
    return EXIT_OK


def _find_case(cases: list, case_id: str):
    exact = [c for c in cases if c.id == case_id]
    if exact:
        return exact[0]
    partial = [c for c in cases if c.id.startswith(case_id)] or \
              [c for c in cases if case_id.lower() in c.id.lower()]
    if len(partial) == 1:
        return partial[0]
    if len(partial) > 1:
        raise ConfigError(f"'{case_id}' is ambiguous: "
                          + ", ".join(c.id for c in partial[:6])
                          + (" …" if len(partial) > 6 else ""))
    close = difflib.get_close_matches(case_id, [c.id for c in cases], n=3, cutoff=0.5)
    raise ConfigError(f"case not found: {case_id}"
                      + (f" (did you mean: {', '.join(close)}?)" if close else
                         " — see `atlas list`"))


def _show_case(args: argparse.Namespace) -> int:
    """``atlas show``: everything the model sees and what we expect back."""
    case = _find_case(load_cases(args.suite), args.case_id)
    print(bold(case.id))
    print(f"{dim('category  ')} {case.category}")
    print(f"{dim('difficulty')} {_difficulty(case.difficulty)}")
    print(f"\n{bold('Prompt')}\n  {case.prompt}")

    print(f"\n{bold('Tools offered')} {dim('(' + str(len(case.tools)) + ')')}")
    for t in case.tools:
        params = t.parameters.get("properties", {}) if isinstance(t.parameters, dict) else {}
        sig = ", ".join(params) if params else ""
        print(f"  {cyan(t.name)}({sig})")
        if t.description:
            print(f"    {dim(_truncate(t.description, 90))}")

    print(f"\n{bold('Expected calls')}")
    if not case.expected_calls:
        print(f"  {dim('none — the model should abstain')}")
    for e in case.expected_calls:
        print(f"  {cyan(e.name)}({json.dumps(e.arguments, ensure_ascii=False)})")

    rubric = getattr(case, "rubric", None)
    if rubric:
        print(f"\n{bold('Rubric')}")
        for item in rubric:
            print(f"  · {item}")
    return EXIT_OK
