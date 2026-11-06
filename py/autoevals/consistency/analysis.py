"""Analyze repeated categorical judgments without running or tuning a scorer."""

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import asdict
from typing import Any, Sequence

from .alignment import PanelCase, align
from .config import ConsistencyGate, ConsistencyLimits, ConsistencyPlan, Rating
from .statistics import (
    Statistic,
    category_counts,
    cohen_kappa,
    fleiss_kappa,
    nominal_alpha,
    pair_agreement,
    raw_agreement,
)


def consensus(case: PanelCase, plan: ConsistencyPlan) -> tuple[str | None, str | None]:
    """Require a strict majority of the entire declared panel, including gaps."""
    counts = category_counts(case)
    if sum(counts.values()) < plan.minimum_consensus_ratings:
        return None, "insufficient_ratings"
    maximum = max(counts.values())
    winners = [category for category, count in counts.items() if count == maximum]
    if len(winners) != 1:
        return None, "tie"
    if maximum * 2 <= len(plan.raters):
        return None, "no_declared_panel_majority"
    return winners[0], None


def reference_accuracy(cases: Sequence[PanelCase], values: Sequence[str | None]) -> dict[str, Any]:
    """Describe evaluated-reference accuracy and abstentions separately."""
    labeled = evaluated = correct = 0
    for case, value in zip(cases, values):
        if case.reference is None:
            continue
        labeled += 1
        if value is not None:
            evaluated += 1
            correct += value == case.reference
    return {
        "labeled_cases": labeled,
        "evaluated_cases": evaluated,
        "correct_cases": correct,
        "unavailable_cases": labeled - evaluated,
        "coverage": evaluated / labeled if labeled else None,
        "accuracy": correct / evaluated if evaluated else None,
        "reason": "no_references" if not labeled else "no_evaluated_references" if not evaluated else None,
    }


def consensus_statistic(cases: Sequence[PanelCase], plan: ConsistencyPlan) -> Statistic:
    """The reference-conditioned consensus statistic used by explicit gates."""
    values = [consensus(case, plan)[0] for case in cases]
    accuracy = reference_accuracy(cases, values)
    support_ratings = sum(
        sum(category is not None for category in case.categories)
        for case, value in zip(cases, values)
        if case.reference is not None and value is not None
    )
    return Statistic(
        accuracy["accuracy"],
        accuracy["evaluated_cases"],
        support_ratings,
        accuracy["labeled_cases"],
        None,
        None,
        "evaluated_reference_cases",
        accuracy["reason"],
    )


def coefficients(cases: Sequence[PanelCase], plan: ConsistencyPlan) -> dict[str, Statistic]:
    """Compute the four group-level statistics on their explicitly different supports."""
    return {
        "raw_agreement": raw_agreement(cases),
        "fleiss_kappa": fleiss_kappa(cases, len(plan.raters)),
        "nominal_alpha": nominal_alpha(cases),
        "consensus_accuracy": consensus_statistic(cases, plan),
    }


def coverage(cases: Sequence[PanelCase], plan: ConsistencyPlan) -> dict[str, Any]:
    """Count additive cell states, including cases never observed at all."""
    states: Counter[str] = Counter(status for case in cases for status in case.statuses)
    counts = [sum(category is not None for category in case.categories) for case in cases]
    expected = len(cases) * len(plan.raters)
    return {
        "total_cases": len(cases),
        "expected_cells": expected,
        "present_cells": expected - states["absent"],
        "scored_cells": states["scored"],
        "missing_cells": states["missing"],
        "error_cells": states["error"],
        "absent_cells": states["absent"],
        "rating_coverage": states["scored"] / expected if expected else None,
        "zero_rating_cases": sum(count == 0 for count in counts),
        "singleton_cases": sum(count == 1 for count in counts),
        "pairable_cases": sum(count >= 2 for count in counts),
        "complete_cases": sum(count == len(plan.raters) for count in counts),
    }


def _rater_report(cases: Sequence[PanelCase], index: int, rater: str) -> dict[str, Any]:
    states = Counter(case.statuses[index] for case in cases)
    values = [case.categories[index] for case in cases]
    reference = reference_accuracy(cases, values)
    for status in ("missing", "error", "absent"):
        reference[f"{status}_cases"] = sum(
            case.reference is not None and case.statuses[index] == status for case in cases
        )
    return {
        "rater_id": rater,
        "scored_cases": states["scored"],
        "missing_cases": states["missing"],
        "error_cases": states["error"],
        "absent_cases": states["absent"],
        "coverage": states["scored"] / len(cases) if cases else None,
        "category_counts": [
            {"category": category, "count": count}
            for category, count in sorted(
                Counter(value for value in values if value is not None).items(),
                key=lambda item: item[0].encode("utf-8"),
            )
        ],
        "reference": reference,
    }


def _case_report(case: PanelCase, plan: ConsistencyPlan) -> dict[str, Any]:
    counts = category_counts(case)
    winner, reason = consensus(case, plan)
    return {
        "case_id": case.case_id,
        "scored_ratings": sum(counts.values()),
        "category_counts": [{"category": category, "count": counts[category]} for category in plan.categories],
        "raw_agreement": pair_agreement(counts),
        "consensus": winner,
        "consensus_reason": reason,
        "reference": case.reference,
        "consensus_correct": winner == case.reference if winner is not None and case.reference is not None else None,
        "unavailable_raters": [
            {"rater_id": rater, "status": status}
            for rater, status in zip(plan.raters, case.statuses)
            if status != "scored"
        ],
    }


def _gate_result(gate: ConsistencyGate, statistic: Statistic, group_coverage: dict) -> dict[str, Any]:
    reasons = []
    support_coverage = statistic.support_cases / statistic.denominator_cases if statistic.denominator_cases else None
    rating_coverage = group_coverage["rating_coverage"]
    if statistic.value is None:
        reasons.append(statistic.reason or "undefined_statistic")
    if statistic.support_cases < gate.minimum_support:
        reasons.append("insufficient_support")
    if support_coverage is None or support_coverage < gate.minimum_coverage:
        reasons.append("insufficient_statistic_coverage")
    if rating_coverage is None or rating_coverage < gate.minimum_rating_coverage:
        reasons.append("insufficient_rating_coverage")
    if reasons:
        verdict = "inconclusive"
    elif statistic.value is not None and statistic.value < gate.minimum:
        verdict = "fail"
        reasons.append("below_observed_threshold")
    else:
        verdict = "pass"
        reasons.append("meets_observed_threshold")
    return {
        **asdict(gate),
        "value": statistic.value,
        "support_cases": statistic.support_cases,
        "denominator_cases": statistic.denominator_cases,
        "support_coverage": support_coverage,
        "rating_coverage": rating_coverage,
        "verdict": verdict,
        "reasons": reasons,
    }


def analyze_consistency(
    ratings: Iterable[Rating],
    *,
    plan: ConsistencyPlan,
    references: Mapping[str, str] | None = None,
    limits: ConsistencyLimits | None = None,
) -> dict[str, Any]:
    """Analyze one categorical criterion across repeated judges or trials.

    No evaluator, model, backend or global random generator is invoked. Missing
    cells stay in coverage denominators. Agreement coefficients use different
    documented supports; consistency alone says nothing about correctness.
    Optional gates describe this finite benchmark. Bootstrap intervals are
    marginal diagnostics, never inputs to gate decisions.
    """
    from .bootstrap import bootstrap_intervals

    if not isinstance(plan, ConsistencyPlan):
        raise ValueError("plan must be a ConsistencyPlan")
    limits = limits or ConsistencyLimits()
    panel = align(ratings, plan, references, limits)
    groups = []
    gates = []
    remaining_details = plan.detail_limit
    for name, indices in panel.groups:
        cases = tuple(panel.cases[index] for index in indices)
        stats = coefficients(cases, plan)
        availability = coverage(cases, plan)
        pairwise = []
        for left, left_name in enumerate(plan.raters):
            for right in range(left + 1, len(plan.raters)):
                statistic, cells = cohen_kappa(cases, left, right)
                pairwise.append(
                    {
                        "left_rater": left_name,
                        "right_rater": plan.raters[right],
                        "statistic": statistic.to_dict(),
                        "confusion": [{"left": a, "right": b, "count": count} for a, b, count in cells],
                    }
                )
        decisions = [consensus(case, plan) for case in cases]
        consensus_reference = reference_accuracy(cases, [value for value, _ in decisions])
        consensus_reference["abstained_cases"] = consensus_reference["unavailable_cases"]
        groups.append(
            {
                "name": name,
                "coverage": availability,
                "statistics": {key: value.to_dict() for key, value in stats.items()},
                "raters": [_rater_report(cases, index, rater) for index, rater in enumerate(plan.raters)],
                "pairwise": pairwise,
                "consensus": {
                    "rule": "strict_majority_of_declared_panel",
                    "decided_cases": sum(value is not None for value, _ in decisions),
                    "abstained_cases": sum(value is None for value, _ in decisions),
                    "reason_counts": [
                        {"reason": reason, "count": count}
                        for reason, count in sorted(Counter(reason for _, reason in decisions if reason).items())
                    ],
                    "reference": consensus_reference,
                },
                "intervals": bootstrap_intervals(cases, plan, name, stats),
                "cases": [_case_report(case, plan) for case in cases[:remaining_details]],
                "omitted_cases": max(0, len(cases) - remaining_details),
            }
        )
        remaining_details = max(0, remaining_details - len(cases))
        gates.extend(
            _gate_result(gate, stats[gate.statistic], availability) for gate in plan.gates if gate.group == name
        )
    verdict = "not_evaluated"
    if gates:
        verdict = (
            "fail"
            if any(gate["verdict"] == "fail" for gate in gates)
            else ("inconclusive" if any(gate["verdict"] == "inconclusive" for gate in gates) else "pass")
        )
    return {
        "format": "autoevals.consistency.report",
        "version": 1,
        "score_name": plan.score_name,
        "cohort_provenance": panel.provenance,
        "categories": list(plan.categories),
        "raters": list(plan.raters),
        "interpretation": "observed_finite_benchmark",
        "interval_interpretation": "marginal_diagnostic_case_bootstrap",
        "groups": groups,
        "gates": gates,
        "verdict": verdict,
    }
