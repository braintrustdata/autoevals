"""Run with uv run python examples/judge_consistency.py; no model calls."""

from autoevals import (
    ConsistencyGate,
    ConsistencyPlan,
    Score,
    ScoreMapping,
    analyze_consistency,
    analyze_consistency_json,
    consistency_scores,
    error_rating,
    input_to_json,
    rating_from_score,
    render_consistency_markdown,
)


def main():
    plan = ConsistencyPlan(
        "supported",
        ("repeat-1", "repeat-2", "repeat-3"),
        ("no", "yes"),
        cases=("case-1", "case-2", "case-3"),
        gates=(ConsistencyGate("raw_agreement", 0.9),),
    )
    mapping = ScoreMapping("supported", ((0, "no"), (1, "yes")))
    # Substitute stored native Scores from your existing judge calls here.
    stored = {
        "case-1": (1, 1, 1),
        "case-2": (0, 1, 0),
        "case-3": (None, 1, None),
    }
    ratings = []
    for case_id, values in stored.items():
        for rater_id, value in zip(plan.raters, values):
            if case_id == "case-3" and rater_id == "repeat-3":
                ratings.append(error_rating(case_id, rater_id))
            else:
                score = Score("supported", value, metadata={"rationale": "not exported"})
                ratings.append(rating_from_score(case_id, rater_id, score, mapping=mapping))
    references = {"case-1": "yes", "case-2": "no", "case-3": "yes"}
    report = analyze_consistency(ratings, plan=plan, references=references)
    bundle = input_to_json(ratings, plan=plan, references=references)
    assert analyze_consistency_json(bundle) == report
    assert "rationale" not in bundle
    assert report["verdict"] == "inconclusive"
    print(render_consistency_markdown(report))
    for score in consistency_scores(report):
        print(score.name, score.score)


if __name__ == "__main__":
    main()
