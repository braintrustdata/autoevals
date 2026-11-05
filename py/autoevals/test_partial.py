import asyncio

import pytest

from autoevals.llm import ClosedQA
from autoevals.string import Levenshtein
from autoevals.value import ExactMatch


def test_partial():
    levenshtein_basic = Levenshtein()(output="abc", expected="abcd")
    levenshtein_partial = Levenshtein.partial(expected="abcd")()(output="abc")
    assert levenshtein_partial.score == levenshtein_basic.score
    assert levenshtein_partial.name == levenshtein_basic.name
    assert levenshtein_partial.name == "Levenshtein"

    closedqa_basic = ClosedQA()(criteria="Is the answer correct?", input="What is 1+1?", output="2")
    closedqa_partial = ClosedQA.partial(criteria="Is the answer correct?")()(input="What is 1+1?", output="2")
    assert closedqa_partial.score == closedqa_basic.score
    assert closedqa_partial.name == closedqa_basic.name
    assert closedqa_partial.name == "ClosedQA"


def run_partial(scorer, method, *args, **kwargs):
    result = getattr(scorer, method)(*args, **kwargs)
    return asyncio.run(result) if method == "eval_async" else result


@pytest.mark.parametrize("method", ["eval", "eval_async", "__call__"])
@pytest.mark.parametrize("output", ["approved", None])
@pytest.mark.parametrize("positional", [False, True])
def test_partial_explicit_none_overrides_bound_expected(method, output, positional):
    scorer = ExactMatch.partial(expected="approved")()
    if positional:
        result = run_partial(scorer, method, output, None)
    else:
        result = run_partial(scorer, method, output=output, expected=None)

    assert result.score == ExactMatch().eval(output, expected=None).score
    assert result.name == "ExactMatch"
    assert run_partial(scorer, method, output="approved").score == 1


@pytest.mark.parametrize("method", ["eval", "eval_async", "__call__"])
def test_partial_omitted_expected_retains_bound_value(method):
    scorer = ExactMatch.partial(expected="approved")()
    assert run_partial(scorer, method, output="approved").score == 1


@pytest.mark.parametrize("method", ["eval", "eval_async", "__call__"])
@pytest.mark.parametrize("expected", ["rejected", "", 0, False])
def test_partial_non_null_overrides_remain_supported(method, expected):
    scorer = ExactMatch.partial(expected="approved")()
    assert run_partial(scorer, method, output=expected, expected=expected).score == 1


@pytest.mark.parametrize("method", ["eval", "eval_async", "__call__"])
def test_partial_without_bound_expected_keeps_default_none(method):
    scorer = ExactMatch.partial()()
    assert run_partial(scorer, method, output=None).score == 1


@pytest.mark.parametrize("method", ["eval", "eval_async", "__call__"])
def test_partial_explicit_none_preserves_required_expected_validation(method):
    scorer = Levenshtein.partial(expected="approved")()
    with pytest.raises(ValueError, match="requires an expected value"):
        run_partial(scorer, method, output="approved", expected=None)
