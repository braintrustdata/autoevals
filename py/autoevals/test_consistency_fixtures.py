import json
from pathlib import Path

import pytest

from autoevals import analyze_consistency_json

STUDIES = json.loads((Path(__file__).parents[2] / "fixtures/consistency/studies.json").read_text())


def subset(actual, expected):
    if type(expected) in (float, int):
        assert actual == pytest.approx(expected, abs=1e-12)
    elif isinstance(expected, list):
        for index, value in enumerate(expected):
            subset(actual[index], value)
    elif isinstance(expected, dict):
        for key, value in expected.items():
            subset(actual[key], value)
    else:
        assert actual == expected


@pytest.mark.parametrize("study", STUDIES, ids=lambda study: study["name"])
def test_shared_studies(study):
    report = analyze_consistency_json(json.dumps(study["input"]))
    subset(report, study["expected"])
    reversed_input = {**study["input"], "ratings": list(reversed(study["input"]["ratings"]))}
    assert analyze_consistency_json(json.dumps(reversed_input)) == report
