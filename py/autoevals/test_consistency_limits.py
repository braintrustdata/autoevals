import pytest

from autoevals.consistency import ConsistencyGroup, ConsistencyLimits, ConsistencyPlan, Rating, analyze_consistency


def test_declared_identity_bytes_count_even_when_no_scores_arrive():
    plan = ConsistencyPlan("judge", ("a", "b"), ("no", "yes"), cases=("long-case-identifier",))
    with pytest.raises(ValueError, match="byte"):
        analyze_consistency([], plan=plan, limits=ConsistencyLimits(max_bytes=20))


def test_empty_groups_still_charge_pairwise_report_work():
    plan = ConsistencyPlan("j", ("a", "b"), ("n", "y"), cases=(), groups=(ConsistencyGroup("empty", ()),))
    with pytest.raises(ValueError, match="pairwise"):
        analyze_consistency([], plan=plan, limits=ConsistencyLimits(max_pairwise_work=1))


def test_reference_bytes_share_the_selected_evidence_budget():
    plan = ConsistencyPlan("j", ("a", "b"), ("n", "y"), cases=("x",))
    # Plan identifiers total six bytes; reference identifiers add two.
    with pytest.raises(ValueError, match="byte"):
        analyze_consistency([], plan=plan, references={"x": "y"}, limits=ConsistencyLimits(max_bytes=7))


def test_invalid_group_work_rejects_before_consuming_observations():
    plan = ConsistencyPlan("j", ("a", "b"), ("n", "y"), groups=(ConsistencyGroup("slice", ("x", "y")),))

    def ratings():
        raise AssertionError("observations should not be consumed")
        yield Rating("x", "a", "y")

    with pytest.raises(ValueError, match="group cell"):
        analyze_consistency(ratings(), plan=plan, limits=ConsistencyLimits(max_group_cells=3))
