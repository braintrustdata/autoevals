from autoevals.llm import ClosedQA
from autoevals.string import Levenshtein


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
