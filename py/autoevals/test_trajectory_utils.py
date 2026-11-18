import json
import os

import pytest

from autoevals.trajectory_utils import compute_trajectory_template_vars

with open(os.path.join(os.path.dirname(__file__), "..", "..", "test-data", "trajectories.json")) as f:
    FIXTURES = json.load(f)


@pytest.mark.parametrize("name", FIXTURES)
def test_trajectory_fixture(name):
    fixture = dict(FIXTURES[name])
    trajectories = fixture.pop("trajectories")
    assert compute_trajectory_template_vars(trajectories) == fixture
