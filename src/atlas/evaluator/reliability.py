"""Reliability metrics: pass@k, pass^k, Wilson CI, bootstrap CI."""

from __future__ import annotations

import math
import random
from typing import Any


def pass_at_k(per_case_success: list[int], n: int, k: int) -> float:
    """Unbiased pass@k: per case, probability at least one success in k draws from n."""
    if k > n:
        k = n
    vals = []
    for c in per_case_success:
        if c == 0:
            vals.append(0.0)
        else:
            # 1 - C(n-c, k)/C(n, k)
            try:
                vals.append(1 - math.comb(n - c, k) / math.comb(n, k))
            except ValueError:
                vals.append(1.0)
    return sum(vals) / len(vals) if vals else 0.0


def pass_pow_k(per_case_success: list[int], n: int, k: int) -> float:
    """pass^k: all k succeed."""
    vals = []
    for c in per_case_success:
        if c < k:
            vals.append(0.0)
        else:
            # C(c, k)/C(n, k)
            try:
                vals.append(math.comb(c, k) / math.comb(n, k))
            except ValueError:
                vals.append(1.0)
    return sum(vals) / len(vals) if vals else 0.0


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, (centre - margin) / denom), min(1.0, (centre + margin) / denom))


def bootstrap_ci(
    scores: list[float],
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Return (mean, low, high) via bootstrap."""
    if not scores:
        return (0.0, 0.0, 0.0)
    rnd = random.Random(seed)
    means = []
    for _ in range(n_bootstrap):
        sample = [rnd.choice(scores) for _ in scores]
        means.append(sum(sample) / len(sample))
    means.sort()
    mean = sum(scores) / len(scores)
    lo_idx = int((alpha / 2) * n_bootstrap)
    hi_idx = int((1 - alpha / 2) * n_bootstrap)
    lo = means[lo_idx] if means else mean
    hi = means[hi_idx] if means else mean
    return (mean, lo, hi)


def reliability_summary(results_by_case: dict[str, list[bool]], scores: list[float]) -> dict:
    """Build reliability dict from per-case pass/fail and overall scores."""
    per_case_success = [sum(lst) for lst in results_by_case.values()]
    n = max((len(v) for v in results_by_case.values()), default=0)
    out: dict[str, Any] = {}
    if n >= 1:
        for k in (1, 3, 5, 10):
            if k <= n:
                out[f"pass_at_{k}"] = pass_at_k(per_case_success, n, k)
                out[f"pass_pow_{k}"] = pass_pow_k(per_case_success, n, k)
    successes = sum(per_case_success)
    total = sum(len(v) for v in results_by_case.values())
    lo, hi = wilson_interval(successes, total)
    out["pass_rate"] = successes / total if total else 0.0
    out["pass_rate_ci"] = [lo, hi]
    mean, blo, bhi = bootstrap_ci(scores)
    out["overall"] = mean
    out["overall_ci"] = [blo, bhi]
    return out

