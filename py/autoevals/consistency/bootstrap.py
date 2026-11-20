"""Reproducible marginal case-bootstrap diagnostics shared with TypeScript."""

from typing import Any, Sequence

from .alignment import PanelCase
from .config import ConsistencyPlan
from .statistics import Statistic


class CaseRandom:
    """Specified xorshift32 stream; seed zero maps to a fixed nonzero state."""

    def __init__(self, seed: int, namespace: str) -> None:
        state = (2166136261 ^ seed) & 0xFFFFFFFF
        for value in namespace.encode("utf-8"):
            state = ((state ^ value) * 16777619) & 0xFFFFFFFF
        self.state = state or 0x9E3779B9

    def next_u32(self) -> int:
        """Return the next unsigned32-bit value without process-global state."""
        value = self.state
        value ^= (value << 13) & 0xFFFFFFFF
        value ^= value >> 17
        value ^= (value << 5) & 0xFFFFFFFF
        self.state = value & 0xFFFFFFFF
        return self.state

    def index(self, size: int) -> int:
        """Rejection-sample indices without a modulo bias."""
        ceiling = (2**32 // size) * size
        while True:
            value = self.next_u32()
            if value < ceiling:
                return value % size


def percentile(values: Sequence[float], probability: float) -> float:
    """Type7 quantile of a sorted nonempty finite sequence."""
    position = (len(values) - 1) * probability
    low = int(position)
    high = min(low + 1, len(values) - 1)
    weight = position - low
    return values[low] * (1 - weight) + values[high] * weight


def bootstrap_intervals(
    cases: Sequence[PanelCase], plan: ConsistencyPlan, group: str, observed: dict[str, Statistic]
) -> list[dict[str, Any]]:
    """Resample whole case occurrences; never deduplicate repeated sampled IDs.

    Undefined draws are counted but never dropped to construct a conditional
    confidence interval. Any such draw makes that interval unavailable. This is
    intentionally conservative for degenerate chance-corrected coefficients.
    """
    from .analysis import coefficients

    options = plan.bootstrap
    if not options.resamples:
        return []
    results = []
    eligible = []
    distributions: dict[str, list[float]] = {}
    for name in options.statistics:
        statistic = observed[name]
        reason = None
        if statistic.value is None:
            reason = "undefined_observed_statistic"
        elif statistic.support_cases < options.minimum_cases:
            reason = "insufficient_support"
        elif options.resamples * (1 - options.confidence) / 2 < 20:
            reason = "insufficient_tail_resamples"
        result = {
            "statistic": name,
            "confidence": options.confidence,
            "resamples": options.resamples,
            "attempted": 0,
            "valid": 0,
            "invalid": 0,
            "lower": None,
            "upper": None,
            "reason": reason,
            "seed": options.seed,
            "algorithm": "case-xorshift32-percentile-v1",
        }
        results.append(result)
        if reason is None:
            eligible.append(result)
            distributions[name] = []
    if not eligible:
        return results
    rng = CaseRandom(options.seed, group)
    for _ in range(options.resamples):
        sample = tuple(cases[rng.index(len(cases))] for _ in cases)
        values = coefficients(sample, plan)
        for result in eligible:
            result["attempted"] += 1
            value = values[result["statistic"]].value
            if value is None:
                result["invalid"] += 1
            else:
                result["valid"] += 1
                distributions[result["statistic"]].append(value)
    for result in eligible:
        distribution = sorted(distributions[result["statistic"]])
        if result["invalid"]:
            result["reason"] = "undefined_resamples"
        elif distribution[0] == distribution[-1]:
            result["reason"] = "degenerate_resamples"
        else:
            alpha = (1 - options.confidence) / 2
            result["lower"] = percentile(distribution, alpha)
            result["upper"] = percentile(distribution, 1 - alpha)
    return results
