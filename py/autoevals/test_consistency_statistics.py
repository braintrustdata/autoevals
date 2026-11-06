import random

import pytest

from autoevals.consistency import (
    ConsistencyBootstrap,
    ConsistencyGate,
    ConsistencyGroup,
    ConsistencyLimits,
    ConsistencyPlan,
    Rating,
    analyze_consistency,
)
from autoevals.consistency.alignment import PanelCase
from autoevals.consistency.bootstrap import CaseRandom
from autoevals.consistency.statistics import cohen_kappa, nominal_alpha


def case(values):
    return PanelCase(
        "case", tuple(values), tuple("scored" if value is not None else "absent" for value in values), None
    )


def test_published_nominal_alpha_missing_data_example():
    # Krippendorff (2013), Computing Alpha-Reliability, pp4-5.
    rows = [
        [1, 2, 3, 3, 2, 1, 4, 1, 2, None, None, None],
        [1, 2, 3, 3, 2, 2, 4, 1, 2, 5, None, 3],
        [None, 3, 3, 3, 2, 3, 4, 2, 2, 5, 1, None],
        [1, 2, 3, 3, 2, 4, 4, 1, 2, 5, 1, None],
    ]
    cases = [case([str(value) if value is not None else None for value in column]) for column in zip(*rows)]
    result = nominal_alpha(cases)
    assert result.value == pytest.approx(0.7434210526315789)
    assert result.support_cases == 11
    assert result.support_ratings == 40  # singleton's rating is excluded
    assert result.observed_agreement == pytest.approx(0.8)
    assert nominal_alpha(cases[:-1]).value == result.value


def test_constant_varying_cohen_is_defined_but_both_constant_is_not():
    statistic, confusion = cohen_kappa([case(["yes", "yes"]), case(["yes", "no"])], 0, 1)
    assert statistic.value == 0
    assert confusion == (("yes", "no", 1), ("yes", "yes", 1))
    statistic, _ = cohen_kappa([case(["yes", "yes"])], 0, 1)
    assert statistic.value is None
    assert statistic.reason == "zero_expected_disagreement"


def test_systematic_disagreement_remains_negative():
    statistic, _ = cohen_kappa([case(["yes", "no"]), case(["no", "yes"])], 0, 1)
    assert statistic.value == -1
    assert nominal_alpha([case(["yes", "no"]), case(["no", "yes"])]).value == -0.5


def test_missingness_support_and_consensus_use_declared_panel():
    plan = ConsistencyPlan("criterion", ("a", "b", "c", "d", "e"), ("no", "yes"), cases=("x", "y"))
    ratings = [
        Rating("x", "a", "yes"),
        Rating("x", "b", "yes"),
        Rating("y", "a", "yes"),
        Rating("y", "b", "yes"),
        Rating("y", "c", "yes"),
    ]
    result = analyze_consistency(ratings, plan=plan, references={"x": "yes", "y": "yes"})["groups"][0]
    assert result["cases"][0]["consensus"] is None
    assert result["cases"][1]["consensus"] == "yes"
    assert result["consensus"]["reference"]["accuracy"] == 1
    assert result["consensus"]["reference"]["coverage"] == 0.5
    assert result["statistics"]["fleiss_kappa"]["value"] is None


def test_low_reference_coverage_cannot_pass_even_with_perfect_observed_accuracy():
    plan = ConsistencyPlan(
        "criterion",
        ("a", "b", "c"),
        ("no", "yes"),
        cases=("x", "y"),
        gates=(ConsistencyGate("consensus_accuracy", 0.9, minimum_rating_coverage=0.5),),
    )
    ratings = [Rating("x", rater, "yes") for rater in plan.raters]
    report = analyze_consistency(ratings, plan=plan, references={"x": "yes", "y": "yes"})
    assert report["verdict"] == "inconclusive"
    assert "insufficient_statistic_coverage" in report["gates"][0]["reasons"]


def test_whole_case_resampling_is_seeded_without_global_rng_mutation():
    plan = ConsistencyPlan(
        "criterion",
        ("a", "b"),
        ("no", "yes"),
        bootstrap=ConsistencyBootstrap(resamples=1000, confidence=0.9, minimum_cases=2),
    )
    ratings = [
        Rating(str(i), rater, "yes" if (i + j) % 3 else "no") for i in range(30) for j, rater in enumerate(plan.raters)
    ]
    state = random.getstate()
    report = analyze_consistency(ratings, plan=plan)
    assert random.getstate() == state
    assert analyze_consistency(list(reversed(ratings)), plan=plan) == report
    intervals = report["groups"][0]["intervals"]
    assert all(item["attempted"] == 1000 and item["valid"] == 1000 for item in intervals)
    assert all(item["lower"] < item["upper"] for item in intervals)


def test_undefined_bootstrap_draws_do_not_become_conditional_intervals():
    plan = ConsistencyPlan(
        "criterion",
        ("a", "b"),
        ("no", "yes"),
        bootstrap=ConsistencyBootstrap(resamples=1000, confidence=0.9, minimum_cases=2),
    )
    ratings = [Rating("x", rater, "yes") for rater in plan.raters] + [
        Rating("y", rater, "no") for rater in plan.raters
    ]
    results = analyze_consistency(ratings, plan=plan)["groups"][0]["intervals"]
    alpha = next(item for item in results if item["statistic"] == "nominal_alpha")
    assert alpha["invalid"] > 0
    assert alpha["lower"] is None
    assert alpha["reason"] == "undefined_resamples"


def test_pairwise_and_bootstrap_resource_checks_precede_sampling():
    plan = ConsistencyPlan(
        "criterion", ("a", "b", "c"), ("no", "yes"), cases=("x", "y"), groups=(ConsistencyGroup("slice", ("x", "y")),)
    )
    with pytest.raises(ValueError, match="pairwise"):
        analyze_consistency([], plan=plan, limits=ConsistencyLimits(max_pairwise_work=10))


def test_unicode_order_is_utf8_and_invalid_surrogates_reject():
    plan = ConsistencyPlan("criterion", ("a", "b"), ("no", "yes"), cases=("\U00010000", "\ue000"))
    assert plan.cases == ("\ue000", "\U00010000")
    with pytest.raises(ValueError, match="surrogate"):
        Rating("\ud800", "a", "yes")


def test_prng_fixed_vector_for_typescript_parity():
    rng = CaseRandom(17, "overall")
    # Independently calculated with JavaScript Math.imul/unsigned shifts.
    assert [rng.next_u32() for _ in range(5)] == [4291421669, 2916200383, 2632349228, 374880413, 943298360]
