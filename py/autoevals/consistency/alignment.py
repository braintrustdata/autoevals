"""Freeze declared panels before any missing-outcome filtering."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from .config import ConsistencyLimits, ConsistencyPlan, Rating, identifier


@dataclass(frozen=True)
class PanelCase:
    """An immutable row in a declared-rater matrix, including absent slots."""

    case_id: str
    categories: tuple[str | None, ...]
    statuses: tuple[str, ...]
    reference: str | None


@dataclass(frozen=True)
class Panel:
    """Canonical case vectors and fixed group indices for one criterion."""

    cases: tuple[PanelCase, ...]
    groups: tuple[tuple[str, tuple[int, ...]], ...]
    provenance: str


def align(
    ratings: Iterable[Rating],
    plan: ConsistencyPlan,
    references: Mapping[str, str] | None,
    limits: ConsistencyLimits,
) -> Panel:
    """Validate every observation before allocating the complete panel.

    Groups and references cannot introduce cases missing from observed data. A
    planned case manifest is required to account for such unobserved cases.
    Every absent slot remains distinct from an explicitly missing/error result.
    """
    plan.check_limits(limits)
    raters = set(plan.raters)
    categories = set(plan.categories)
    declared = set(plan.cases) if plan.cases is not None else None
    observed: dict[str, dict[str, Rating]] = {}
    selected_bytes = plan.selected_bytes()
    for index, rating in enumerate(ratings):
        if index >= limits.max_ratings:
            raise ValueError("consistency rating limit exceeded")
        if not isinstance(rating, Rating):
            raise ValueError("ratings must contain Rating objects")
        if rating.rater_id not in raters:
            raise ValueError("unknown rater_id")
        if rating.category is not None and rating.category not in categories:
            raise ValueError("unknown rating category")
        if declared is not None and rating.case_id not in declared:
            raise ValueError("case_id is outside the expected manifest")
        cells = observed.setdefault(rating.case_id, {})
        if rating.rater_id in cells:
            raise ValueError("duplicate case/rater cell")
        if len(observed) > limits.max_cases:
            raise ValueError("consistency case limit exceeded")
        selected_bytes += sum(
            len(value.encode("utf-8"))
            for value in (rating.case_id, rating.rater_id, rating.category or "", rating.status)
        )
        if selected_bytes > limits.max_bytes:
            raise ValueError("consistency selected byte limit exceeded")
        cells[rating.rater_id] = rating
    case_ids = (
        plan.cases if plan.cases is not None else tuple(sorted(observed, key=lambda value: value.encode("utf-8")))
    )
    if len(case_ids) * len(plan.raters) > limits.max_cells:
        raise ValueError("consistency cell limit exceeded")
    known_cases = set(case_ids)
    if references is None:
        references = {}
    if not isinstance(references, Mapping) or len(references) > limits.max_cases:
        raise ValueError("references must be a bounded case/category mapping")
    copied_references = {}
    for case_id, category in references.items():
        identifier(case_id, "reference case_id")
        identifier(category, "reference category")
        if case_id not in known_cases:
            raise ValueError("reference case_id is outside the cohort")
        if category not in categories:
            raise ValueError("unknown reference category")
        selected_bytes += len(case_id.encode("utf-8")) + len(category.encode("utf-8"))
        if selected_bytes > limits.max_bytes:
            raise ValueError("consistency selected byte limit exceeded")
        copied_references[case_id] = category
    group_cases = len(case_ids) + sum(len(group.cases) for group in plan.groups)
    if group_cases * len(plan.raters) > limits.max_group_cells:
        raise ValueError("consistency group cell limit exceeded")
    pair_count = len(plan.raters) * (len(plan.raters) - 1) // 2
    if (group_cases + len(plan.groups) + 1) * pair_count > limits.max_pairwise_work:
        raise ValueError("consistency pairwise work limit exceeded")
    bootstrap_work = group_cases * len(plan.raters) * plan.bootstrap.resamples
    if bootstrap_work > limits.max_bootstrap_work:
        raise ValueError("consistency bootstrap work limit exceeded")
    positions = {case_id: index for index, case_id in enumerate(case_ids)}
    groups = [("overall", tuple(range(len(case_ids))))]
    for group in plan.groups:
        if set(group.cases) - known_cases:
            raise ValueError("group cases are outside the cohort; declare expected cases")
        groups.append((group.name, tuple(positions[case_id] for case_id in group.cases)))
    cases = []
    for case_id in case_ids:
        supplied = observed.get(case_id, {})
        values, statuses = [], []
        for rater_id in plan.raters:
            rating = supplied.get(rater_id)
            values.append(rating.category if rating else None)
            statuses.append(rating.status if rating else "absent")
        cases.append(PanelCase(case_id, tuple(values), tuple(statuses), copied_references.get(case_id)))
    return Panel(tuple(cases), tuple(groups), "expected_manifest" if declared is not None else "observed_union")
