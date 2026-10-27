"""Argument-level scoring helpers.

These pure functions compare one expected-arguments dict against one
actual-arguments dict. They know nothing about tools, cases, or models —
just "how close is this dict to that dict?".

Comparison rules (applied recursively, so nested values follow the same
rules as top-level ones):
- Strings match case-insensitively + surrounding whitespace ignored
  (``"Tokyo"`` vs ``"tokyo"`` is a hit; ``"Tokyo"`` vs ``"Berlin"`` is not).
- Bools never match ints (``True`` vs ``1`` is a miss, despite Python).
- Dicts score by the mean score of their expected inner keys. Extra inner
  keys are ignored; a missing inner key scores 0 (even if the expected
  value is ``None``).
- Lists require exact element-wise equality (order matters, e.g.
  attendees), using the same string/bool rules for the elements.
- Extra unexpected TOP-LEVEL keys cap the arg score at 0.8 (hallucinated
  params hurt).

Two scores come out of this module:
- ``_arg_score``       — single 0..1 accuracy for one aligned call.
- ``_precision_recall``— (precision, recall) split for one aligned call.
Both use the same underlying ``_deep_score`` so they can't disagree about
what counts as "correct".
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Sentinel: distinguishes "key absent" from "key present with value None".
_MISSING = object()

# Cap applied when the model emitted a top-level key that wasn't expected.
_EXTRA_KEY_CAP = 0.8


def _flat_match(expected: Any, actual: Any) -> bool:
    """Scalar equality with case-insensitive string comparison.

    >>> _flat_match("Tokyo", "tokyo")
    True
    >>> _flat_match(5, "5")
    False
    >>> _flat_match(True, 1)
    False
    >>> _flat_match(5, 5.0)
    True
    """
    if isinstance(expected, str) and isinstance(actual, str):
        return expected.strip().lower() == actual.strip().lower()
    # bool is a subclass of int in Python, so True == 1. Don't allow that.
    if isinstance(expected, bool) != isinstance(actual, bool):
        return False
    return expected == actual


def _deep_score(expected: Any, actual: Any) -> float:
    """Score one expected value against one actual value, 0..1.

    - dict  → mean of ``_deep_score`` over the expected inner keys
    - list  → 1.0 only if same length and every element fully matches
    - other → 1.0 / 0.0 via ``_flat_match``

    ``actual`` may be ``_MISSING`` (key absent), which always scores 0.
    """
    if actual is _MISSING:
        return 0.0

    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return 0.0
        if not expected:
            return 1.0 if not actual else 0.0
        return sum(
            _deep_score(v, actual.get(k, _MISSING)) for k, v in expected.items()
        ) / len(expected)

    if isinstance(expected, list):
        if not isinstance(actual, list) or len(expected) != len(actual):
            return 0.0
        return 1.0 if all(_deep_score(e, a) == 1.0 for e, a in zip(expected, actual)) else 0.0

    return 1.0 if _flat_match(expected, actual) else 0.0


def _arg_score(expected_args: dict, actual_args: dict) -> float:
    """Mean correctness of the expected argument keys, 0..1.

    Extra top-level keys the model invented cap the score at 0.8 (the cap
    is applied on top of the normal partial-credit score, so nested/list
    credit is treated identically with or without extras).
    Empty-expected + empty-actual = 1.0; empty-expected + non-empty-actual
    = 0.8 (correct tool, noisy args).
    """
    has_extra = any(k not in expected_args for k in actual_args)

    if not expected_args:
        return 0.8 if actual_args else 1.0

    base = sum(
        _deep_score(v, actual_args.get(k, _MISSING)) for k, v in expected_args.items()
    ) / len(expected_args)

    if has_extra:
        logger.debug("extra top-level arg(s) penalized (expected=%s actual=%s)",
                     expected_args, actual_args)
        return min(base, _EXTRA_KEY_CAP)
    return base


def _arg_score_penalized(expected_args: dict, actual_args: dict, extra_key: str) -> float:
    """Back-compat shim: arg score with the extra-key cap applied.

    ``_arg_score`` now handles extra keys itself; this only exists so older
    callers/tests importing it keep working. ``extra_key`` is ignored.
    """
    capped = dict(actual_args)
    capped.setdefault(extra_key, None)
    return _arg_score(expected_args, capped)


def _flatten_args(args: dict, prefix: tuple = ()) -> dict[tuple, Any]:
    """Flatten nested dicts into ``{path_tuple: leaf_value}``.

    ``{"a": {"x": 1, "y": {"z": 2}}, "b": [1]}`` →
    ``{("a","x"): 1, ("a","y","z"): 2, ("b",): [1]}``.
    Lists and empty dicts are leaves (compared whole, like ``_deep_score``).
    Tuple paths (not dotted strings) so keys containing ``.`` can't collide.
    """
    out: dict[tuple, Any] = {}
    for key, val in args.items():
        path = prefix + (key,)
        if isinstance(val, dict) and val:
            out.update(_flatten_args(val, path))
        else:
            out[path] = val
    return out


def _precision_recall(expected_args: dict, actual_args: dict) -> tuple[float, float]:
    """Return (precision, recall) for one aligned expected/actual pair.

    Both sides are flattened to leaf paths (see ``_flatten_args``) and
    compared leaf-by-leaf, so nested arguments count per leaf instead of as
    one top-level key:

    - Precision = correct leaves / leaves the model emitted.
      Low precision ⇒ model padded args with junk/hallucinations
      (including extra keys inside nested objects).
    - Recall = correct leaves / leaves that were expected.
      Low recall ⇒ model omitted required info.

    (``_arg_score`` is deliberately unchanged: top-level mean with partial
    credit for nested dicts. Precision/recall are the finer-grained view.)

    Edge cases: both empty → (1.0, 1.0); nothing expected but model emitted
    something → (0.0, 1.0); nothing emitted but args expected → (1.0, 0.0).
    """
    if not expected_args and not actual_args:
        return 1.0, 1.0
    if not expected_args:
        return 0.0, 1.0
    if not actual_args:
        return 1.0, 0.0

    exp_leaves = _flatten_args(expected_args)
    act_leaves = _flatten_args(actual_args)
    correct = sum(
        1 for path, exp_val in exp_leaves.items()
        if _deep_score(exp_val, act_leaves.get(path, _MISSING)) == 1.0
    )
    return correct / len(act_leaves), correct / len(exp_leaves)
