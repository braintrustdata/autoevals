import math
import sys

import pytest

from autoevals.number import NumericDiff


@pytest.mark.parametrize(
    "output,expected,score",
    [
        (sys.float_info.max, sys.float_info.max, 1),
        (sys.float_info.max, sys.float_info.max / 2, 2 / 3),
        (sys.float_info.max, -sys.float_info.max, 0),
        (-sys.float_info.max, -sys.float_info.max / 2, 2 / 3),
        (1e-320, 1e-320 / 2, 2 / 3),
        (0, 0, 1),
        (0, sys.float_info.max, 0),
        (10, 5, 2 / 3),
    ],
)
def test_numeric_diff_finite_extremes(output, expected, score):
    result = NumericDiff()(output, expected).score
    assert math.isfinite(result)
    assert result == pytest.approx(score)
