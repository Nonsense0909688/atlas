"""Flexible per-argument matchers."""

from __future__ import annotations

import datetime
import difflib
import logging
import math
import re
from typing import Any

logger = logging.getLogger(__name__)


def _parse_date(s: str) -> str | None:
    s = s.strip()
    date_formats = (
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%d-%m-%Y",
        "%m/%d/%Y",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M",
        "%d %b %Y",
        "%b %d, %Y",
        "%Y-%m-%d %H:%M:%S",
    )
    for fmt in date_formats:
        try:
            return datetime.datetime.strptime(s, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    try:
        dt = datetime.datetime.fromisoformat(s.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d")
    except ValueError:
        return None


def match_value(expected: Any, actual: Any, spec: dict | None = None) -> float:
    """Score one value 0..1 using matcher spec."""
    if not spec:
        # default exact (case-insensitive for strings)
        if isinstance(expected, str) and isinstance(actual, str):
            return 1.0 if expected.strip().lower() == actual.strip().lower() else 0.0
        if isinstance(expected, bool) != isinstance(actual, bool):
            return 0.0
        return 1.0 if expected == actual else 0.0
    typ = spec.get("type", "exact")
    if typ == "exact":
        if isinstance(expected, str) and isinstance(actual, str):
            return 1.0 if expected.strip().lower() == actual.strip().lower() else 0.0
        return 1.0 if expected == actual else 0.0
    if typ == "regex":
        pattern = spec.get("pattern", "")
        if not isinstance(actual, str):
            return 0.0
        return 1.0 if re.search(pattern, actual) else 0.0
    if typ == "numeric":
        tol = spec.get("tolerance", 1e-6)
        try:
            e = float(expected)
            a = float(actual)
        except (TypeError, ValueError):
            return 0.0
        return 1.0 if abs(e - a) <= tol else 0.0
    if typ == "date":
        if not isinstance(actual, str):
            return 0.0
        e = _parse_date(str(expected))
        a = _parse_date(actual)
        return 1.0 if e and a and e == a else 0.0
    if typ == "set":
        if not isinstance(actual, list) or not isinstance(expected, list):
            return 0.0
        expected_values = {str(value).lower() for value in expected}
        actual_values = {str(value).lower() for value in actual}
        return 1.0 if expected_values == actual_values else 0.0
    if typ == "any_of":
        options = spec.get("values", [])
        for opt in options:
            if isinstance(opt, str) and isinstance(actual, str):
                if opt.strip().lower() == actual.strip().lower():
                    return 1.0
            elif opt == actual:
                return 1.0
        return 0.0
    if typ == "llm_judge":
        # Fallback heuristic: token overlap unless caller provides judge fn via spec['_judge']
        judge = spec.get("_judge")
        if callable(judge):
            try:
                return 1.0 if judge(expected, actual, spec) else 0.0
            except Exception:
                return 0.0
        # Heuristic: Jaccard of word sets
        if not isinstance(actual, str):
            return 0.0
        e_tokens = set(str(expected).lower().split())
        a_tokens = set(actual.lower().split())
        if not e_tokens and not a_tokens:
            return 1.0
        inter = len(e_tokens & a_tokens)
        union = len(e_tokens | a_tokens)
        return inter / union if union else 0.0
    if typ == "fuzzy":
        # Levenshtein-ish ratio
        if not isinstance(actual, str):
            return 0.0
        thresh = spec.get("threshold", 0.8)
        e, a = str(expected).lower(), actual.lower()
        # simple: difflib ratio
        ratio = difflib.SequenceMatcher(None, e, a).ratio()
        return 1.0 if ratio >= thresh else ratio
    return 0.0


def score_args_with_matchers(expected_args: dict, actual_args: dict, matchers: dict) -> float:
    """Score args dict using per-arg matchers. matchers keys: arg name or 'tool.arg'."""
    if not expected_args:
        return 1.0 if not actual_args else 0.8
    scores = []
    for key, exp_val in expected_args.items():
        spec = None
        # try dotted key first, then plain
        for k, v in matchers.items():
            if k == key or k.endswith(f".{key}"):
                # if dotted, only apply if tool matches? ignore tool part for now
                spec = v
                break
        actual_val = actual_args.get(key) if isinstance(actual_args, dict) else None
        if isinstance(actual_args, dict) and key not in actual_args:
            scores.append(0.0)
        else:
            scores.append(match_value(exp_val, actual_val, spec))
    return sum(scores) / len(scores) if scores else 1.0
