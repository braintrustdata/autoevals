import json

import pytest

from autoevals import Score
from autoevals.consistency import ConsistencyPlan, Rating, analyze_consistency
from autoevals.consistency.reporting import consistency_scores, render_consistency_markdown
from autoevals.consistency.serialization import analyze_consistency_json, input_to_json, report_to_json


def test_input_bundle_roundtrip_and_strict_schema():
    plan = ConsistencyPlan("judge", ("a", "b"), ("no", "yes"))
    ratings = [Rating("x", "a", "yes"), Rating("x", "b", "no")]
    encoded = input_to_json(ratings, plan=plan, references={"x": "yes"})
    expected = analyze_consistency(ratings, plan=plan, references={"x": "yes"})
    assert analyze_consistency_json(encoded) == expected
    assert json.loads(report_to_json(expected)) == expected
    payload = json.loads(encoded)
    payload["plan"]["unknown"] = 1
    with pytest.raises(ValueError, match="unknown"):
        analyze_consistency_json(json.dumps(payload))


@pytest.mark.parametrize("content", ['{"version":1,"version":1}', '{"x":NaN}', '{"x":Infinity}'])
def test_duplicate_keys_and_nonfinite_json_constants_are_invalid(content):
    with pytest.raises(ValueError):
        analyze_consistency_json(content)


def test_native_summary_does_not_put_signed_kappa_in_score_or_fake_missing_zero():
    plan = ConsistencyPlan("judge", ("a", "b"), ("no", "yes"))
    ratings = [Rating("x", "a", "yes"), Rating("x", "b", "no"), Rating("y", "a", "no"), Rating("y", "b", "yes")]
    report = analyze_consistency(ratings, plan=plan)
    scores = consistency_scores(report)
    assert all(isinstance(score, Score) for score in scores)
    assert {score.name for score in scores} == {
        "judge/overall/agreement",
        "judge/overall/coverage",
        "judge/overall/consensus_accuracy",
    }
    assert scores[-1].score is None
    assert all("report" not in score.metadata for score in scores)


def test_markdown_escapes_untrusted_labels_and_has_no_ansi_controls():
    plan = ConsistencyPlan("judge|<script>\x1b[31m", ("a", "b"), ("no", "yes"))
    text = render_consistency_markdown(analyze_consistency([], plan=plan))
    assert "<script>" not in text
    assert "\x1b" not in text
    assert "judge\\|&lt;script&gt;" in text


def test_markdown_labels_cannot_load_remote_images_or_override_escapes():
    plan = ConsistencyPlan(r"![external](https://example.com/track)\*", ("a", "b"), ("no", "yes"))
    text = render_consistency_markdown(analyze_consistency([], plan=plan))
    assert r"\!\[external\]\(https://example\.com/track\)\\\*" in text
