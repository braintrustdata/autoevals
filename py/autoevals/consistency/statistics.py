"""Nominal agreement coefficients with their own support and chance models."""

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

from .alignment import PanelCase


def stable_sum(values: Iterable[float]) -> float:
    """Explicit Neumaier accumulation, identical to the TypeScript algorithm.

    Python's built-in float sum changed in 3.12. Specify the operations here so
    supported runtimes make the same decision at a configured gate boundary.
    """
    result, correction = 0.0, 0.0
    for value in values:
        next_value = result + value
        if abs(result) >= abs(value):
            correction += (result - next_value) + value
        else:
            correction += (value - next_value) + result
        result = next_value
    return result + correction


@dataclass(frozen=True)
class Statistic:
    """Undefined statistics retain a reason and never impersonate perfect agreement."""

    value: float | None
    support_cases: int
    support_ratings: int
    denominator_cases: int
    observed_agreement: float | None
    expected_agreement: float | None
    weighting: str
    reason: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def category_counts(case: PanelCase) -> Counter[str]:
    """Count only available categorical observations, never missing slots."""
    return Counter(category for category in case.categories if category is not None)


def pair_agreement(counts: Counter[str]) -> float | None:
    """Fraction of equal unordered pairs within one case."""
    total = sum(counts.values())
    if total < 2:
        return None
    return sum(count * (count - 1) for count in counts.values()) / (total * (total - 1))


def raw_agreement(cases: Sequence[PanelCase]) -> Statistic:
    """Equal-case mean of available-pair agreement; no chance correction."""
    eligible = [(category_counts(case), case) for case in cases]
    eligible = [(counts, case) for counts, case in eligible if sum(counts.values()) >= 2]
    if not eligible:
        return Statistic(None, 0, 0, len(cases), None, None, "equal_pairable_cases", "no_pairable_cases")
    agreement = stable_sum(pair_agreement(counts) or 0 for counts, _ in eligible) / len(eligible)
    return Statistic(
        agreement,
        len(eligible),
        sum(sum(counts.values()) for counts, _ in eligible),
        len(cases),
        agreement,
        None,
        "equal_pairable_cases",
    )


def fleiss_kappa(cases: Sequence[PanelCase], rater_count: int) -> Statistic:
    """Pooled-marginal Fleiss coefficient on complete fixed-size panels only.

    This describes repeated categorical coding. Interpreting it as reliability
    over randomly sampled raters requires that study design; named persistent
    judges additionally have pairwise Cohen statistics with separate marginals.
    """
    complete = [case for case in cases if all(category is not None for category in case.categories)]
    if not complete:
        return Statistic(None, 0, 0, len(cases), None, None, "complete_panels", "no_complete_panels")
    pooled: Counter[str] = Counter()
    agreements = []
    for case in complete:
        counts = category_counts(case)
        pooled.update(counts)
        agreements.append(pair_agreement(counts) or 0)
    count = len(complete) * rater_count
    observed = stable_sum(agreements) / len(complete)
    expected = stable_sum((value / count) * (value / count) for value in pooled.values())
    denominator = 1 - expected
    value = None if denominator <= 0 else (observed - expected) / denominator
    return Statistic(
        value,
        len(complete),
        count,
        len(cases),
        observed,
        expected,
        "complete_panels",
        "zero_expected_disagreement" if value is None else None,
    )


def nominal_alpha(cases: Sequence[PanelCase]) -> Statistic:
    """Krippendorff nominal alpha with pairable-observation weighting.

    A unit with m observations contributes n_c(n_k-delta_ck)/(m-1) to
    coincidence counts. Summing disagreement directly avoids a vocabulary²
    matrix. Singletons contribute neither observed nor expected coincidences.
    The finite-sample expected disagreement denominator is N(N-1), not N².
    """
    pooled: Counter[str] = Counter()
    disagreements = []
    support = 0
    for case in cases:
        counts = category_counts(case)
        count = sum(counts.values())
        if count < 2:
            continue
        support += 1
        pooled.update(counts)
        disagreements.append((count * count - sum(value * value for value in counts.values())) / (count - 1))
    total = sum(pooled.values())
    if not support:
        return Statistic(None, 0, 0, len(cases), None, None, "pairable_observations", "no_pairable_cases")
    observed_disagreement = stable_sum(disagreements) / total
    expected_disagreement = (total * total - sum(value * value for value in pooled.values())) / (total * (total - 1))
    value = None if expected_disagreement <= 0 else 1 - observed_disagreement / expected_disagreement
    return Statistic(
        value,
        support,
        total,
        len(cases),
        1 - observed_disagreement,
        1 - expected_disagreement,
        "pairable_observations",
        "zero_expected_disagreement" if value is None else None,
    )


def cohen_kappa(
    cases: Sequence[PanelCase], left: int, right: int
) -> tuple[Statistic, tuple[tuple[str, str, int], ...]]:
    """Cohen kappa and sparse left-row/right-column confusion cells.

    Both marginals use the same overlap cohort. A constant rater compared with
    a varying rater remains defined; only zero expected disagreement is singular.
    """
    matrix: Counter[tuple[str, str]] = Counter()
    left_counts: Counter[str] = Counter()
    right_counts: Counter[str] = Counter()
    for case in cases:
        a, b = case.categories[left], case.categories[right]
        if a is None or b is None:
            continue
        matrix[(a, b)] += 1
        left_counts[a] += 1
        right_counts[b] += 1
    total = sum(matrix.values())
    cells = tuple(
        (a, b, count)
        for (a, b), count in sorted(
            matrix.items(), key=lambda item: (item[0][0].encode("utf-8"), item[0][1].encode("utf-8"))
        )
    )
    if not total:
        return Statistic(None, 0, 0, len(cases), None, None, "pairwise_overlap", "no_overlap"), cells
    observed = sum(count for (a, b), count in matrix.items() if a == b) / total
    expected = sum(count * right_counts[category] for category, count in left_counts.items()) / (total * total)
    value = None if expected >= 1 else (observed - expected) / (1 - expected)
    return (
        Statistic(
            value,
            total,
            2 * total,
            len(cases),
            observed,
            expected,
            "pairwise_overlap",
            "zero_expected_disagreement" if value is None else None,
        ),
        cells,
    )
