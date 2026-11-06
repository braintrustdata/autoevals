"""Explicit categorical repeated-judge study contracts and trusted ceilings."""

import math
from dataclasses import dataclass, field
from typing import Any

STATISTICS = ("raw_agreement", "fleiss_kappa", "nominal_alpha", "consensus_accuracy")


def identifier(value: Any, name: str) -> str:
    """Validate portable Unicode identifiers without coercion or normalization."""
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a nonempty string")
    try:
        size = len(value.encode("utf-8"))
    except UnicodeError as exc:
        raise ValueError(f"{name} contains an invalid Unicode surrogate") from exc
    if size > 4096:
        raise ValueError(f"{name} exceeds the string limit")
    return value


def integer(value: Any, name: str, low: int, high: int) -> int:
    """Restrict counts to integers exactly representable in both runtimes."""
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} must be an integer between {low} and {high}")
    return value


def finite(value: Any, name: str, low: float, high: float) -> float:
    """Reject coercible strings and boolean scores, including Python bools."""
    if type(value) not in (int, float):
        raise ValueError(f"{name} must be a finite number")
    if not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return float(value)


def names(values: Any, name: str, *, minimum: int = 0, maximum: int = 100000) -> tuple[str, ...]:
    """Copy bounded unique names in UTF-8 lexicographic order."""
    if not isinstance(values, (tuple, list)) or not minimum <= len(values) <= maximum:
        raise ValueError(f"{name} must be a bounded sequence")
    result = tuple(identifier(value, name) for value in values)
    if len(set(result)) != len(result):
        raise ValueError(f"duplicate {name}")
    return tuple(sorted(result, key=lambda value: value.encode("utf-8")))


@dataclass(frozen=True)
class ConsistencyLimits:
    """Application-owned resource ceilings; serialized inputs cannot raise them."""

    max_ratings: int = 1000000
    max_cases: int = 100000
    max_raters: int = 64
    max_categories: int = 100
    max_groups: int = 50
    max_cells: int = 1000000
    max_pairwise_work: int = 10000000
    max_group_cells: int = 2000000
    max_bootstrap_work: int = 100000000
    max_details: int = 10000
    max_bytes: int = 64 * 1024 * 1024
    max_json_nodes: int = 2000000
    max_json_depth: int = 20

    def __post_init__(self) -> None:
        for key, value in vars(self).items():
            integer(value, key, 1, 2**53 - 1)


@dataclass(frozen=True)
class Rating:
    """One explicit case/rater cell, with an outcome separate from availability."""

    case_id: str
    rater_id: str
    category: str | None = None
    status: str = "scored"

    def __post_init__(self) -> None:
        identifier(self.case_id, "case_id")
        identifier(self.rater_id, "rater_id")
        if self.status not in ("scored", "missing", "error"):
            raise ValueError("rating status must be scored, missing or error")
        if self.status == "scored":
            identifier(self.category, "category")
        elif self.category is not None:
            raise ValueError("unavailable ratings cannot contain a category")


@dataclass(frozen=True)
class ConsistencyGroup:
    """A predeclared subset of case identities, independent of scored outcomes."""

    name: str
    cases: tuple[str, ...]

    def __post_init__(self) -> None:
        identifier(self.name, "group name")
        if self.name == "overall":
            raise ValueError("overall is reserved")
        object.__setattr__(self, "cases", names(self.cases, "group cases"))


@dataclass(frozen=True)
class ConsistencyGate:
    """An observed finite-benchmark requirement, never a confidence guarantee.

    Support coverage uses the statistic's own cohort: pairable cases for raw
    agreement/alpha, complete panels for Fleiss, and evaluated reference cases
    for consensus accuracy. Rating coverage independently counts all cells.
    """

    statistic: str
    minimum: float
    group: str = "overall"
    minimum_support: int = 2
    minimum_coverage: float = 1.0
    minimum_rating_coverage: float = 1.0

    def __post_init__(self) -> None:
        if self.statistic not in STATISTICS:
            raise ValueError("unknown gate statistic")
        identifier(self.group, "gate group")
        object.__setattr__(self, "minimum", finite(self.minimum, "minimum", -1, 1))
        if self.statistic in ("raw_agreement", "consensus_accuracy") and self.minimum < 0:
            raise ValueError("agreement and accuracy thresholds must be nonnegative")
        integer(self.minimum_support, "minimum_support", 1, 100000)
        for key in ("minimum_coverage", "minimum_rating_coverage"):
            object.__setattr__(self, key, finite(getattr(self, key), key, 0, 1))


@dataclass(frozen=True)
class ConsistencyBootstrap:
    """Marginal diagnostic case-bootstrap intervals; disabled by default."""

    resamples: int = 0
    confidence: float = 0.95
    seed: int = 0
    minimum_cases: int = 20
    statistics: tuple[str, ...] = ("raw_agreement", "fleiss_kappa", "nominal_alpha")

    def __post_init__(self) -> None:
        integer(self.resamples, "resamples", 0, 100000)
        integer(self.seed, "seed", 0, 2**32 - 1)
        integer(self.minimum_cases, "minimum_cases", 2, 100000)
        confidence = finite(self.confidence, "confidence", 0, 1)
        if confidence in (0, 1):
            raise ValueError("confidence must be strictly between zero and one")
        object.__setattr__(self, "confidence", confidence)
        statistics = names(self.statistics, "bootstrap statistics", minimum=1, maximum=4)
        if set(statistics) - set(STATISTICS):
            raise ValueError("unknown bootstrap statistic")
        object.__setattr__(self, "statistics", statistics)


@dataclass(frozen=True)
class ConsistencyPlan:
    """Freeze label vocabulary, repeat panel and reporting policy before analysis."""

    score_name: str
    raters: tuple[str, ...]
    categories: tuple[str, ...]
    cases: tuple[str, ...] | None = None
    groups: tuple[ConsistencyGroup, ...] = ()
    gates: tuple[ConsistencyGate, ...] = ()
    minimum_consensus_ratings: int = 2
    detail_limit: int = 100
    bootstrap: ConsistencyBootstrap = field(default_factory=ConsistencyBootstrap)

    def __post_init__(self) -> None:
        identifier(self.score_name, "score_name")
        object.__setattr__(self, "raters", names(self.raters, "raters", minimum=2, maximum=1000))
        object.__setattr__(self, "categories", names(self.categories, "categories", minimum=2, maximum=1000))
        if self.cases is not None:
            object.__setattr__(self, "cases", names(self.cases, "cases"))
        for key, cls in (("groups", ConsistencyGroup), ("gates", ConsistencyGate)):
            values = getattr(self, key)
            if not isinstance(values, (list, tuple)) or len(values) > 1000:
                raise ValueError(f"{key} must be a bounded sequence")
            if not all(isinstance(value, cls) for value in values):
                raise ValueError(f"invalid {key} object")
            object.__setattr__(self, key, tuple(values))
        if len({group.name for group in self.groups}) != len(self.groups):
            raise ValueError("duplicate group name")
        object.__setattr__(self, "groups", tuple(sorted(self.groups, key=lambda group: group.name.encode("utf-8"))))
        known_groups = {"overall", *(group.name for group in self.groups)}
        if any(gate.group not in known_groups for gate in self.gates):
            raise ValueError("gate references an unknown group")
        gate_keys = [(gate.group, gate.statistic) for gate in self.gates]
        if len(set(gate_keys)) != len(gate_keys):
            raise ValueError("duplicate gate statistic/group")
        object.__setattr__(
            self, "gates", tuple(sorted(self.gates, key=lambda gate: (gate.group.encode("utf-8"), gate.statistic)))
        )
        integer(self.minimum_consensus_ratings, "minimum_consensus_ratings", 2, len(self.raters))
        integer(self.detail_limit, "detail_limit", 0, 100000)
        if not isinstance(self.bootstrap, ConsistencyBootstrap):
            raise ValueError("bootstrap must be a ConsistencyBootstrap object")

    def check_limits(self, limits: ConsistencyLimits) -> None:
        """Check fixed dimensions before consuming observations."""
        for actual, limit, name in (
            (len(self.raters), limits.max_raters, "raters"),
            (len(self.categories), limits.max_categories, "categories"),
            (len(self.cases or ()), limits.max_cases, "cases"),
            (len(self.groups), limits.max_groups, "groups"),
            (self.detail_limit, limits.max_details, "details"),
        ):
            if actual > limit:
                raise ValueError(f"consistency {name} limit exceeded")
        if len(self.cases or ()) * len(self.raters) > limits.max_cells:
            raise ValueError("consistency cell limit exceeded")
        if (len(self.cases or ()) + sum(len(group.cases) for group in self.groups)) * len(
            self.raters
        ) > limits.max_group_cells:
            raise ValueError("consistency group cell limit exceeded")
        if self.selected_bytes() > limits.max_bytes:
            raise ValueError("consistency selected byte limit exceeded")

    def selected_bytes(self) -> int:
        """Include declared identities in the direct API's selected-data budget."""
        total = len(self.score_name.encode("utf-8"))
        for values in (self.raters, self.categories, self.cases or ()):
            total += sum(len(value.encode("utf-8")) for value in values)
        for group in self.groups:
            total += len(group.name.encode("utf-8"))
            total += sum(len(value.encode("utf-8")) for value in group.cases)
        return total
