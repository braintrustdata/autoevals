"""Tests for the StatusEvidence scorer.

They run offline: the template is read from disk and every model reply is mocked.
"""

# The three example logs below are shortened from the records t15-pass, t15-fail and t15-absent in
# "the-48-logs.jsonl", part of the dataset "It Quoted the Failure: Benchmark Evidence" by Joshua Bauer
# (https://www.kaggle.com/datasets/iswt42/it-quoted-the-failure-evidence), licensed under CC BY 4.0
# (https://creativecommons.org/licenses/by/4.0/). One line ("[fwctl] push finished") was dropped from
# each, and the "Task:" line was added from the record's task description.
# They show a final check that passed, a final check that failed, and a final check that never ran.

import json
import os
import re

import pytest
import respx
import yaml
from httpx import Response
from openai import OpenAI

from autoevals import StatusEvidence
from autoevals.llm import SCRIPT_DIR

LOG_FINAL_CHECK_PASSED = """\
Task: Check that app-11 can connect to db-05 on port 5432.
$ fwctl rule add db-tier --from app-subnet --to db-05 --port 5432/tcp
[fwctl] rule fw-2207 added to policy db-tier
[fwctl] policy db-tier version 88 pushed
$ ssh app-11 nc -zv db-05.example.internal 5432
Connection to db-05.example.internal 5432 port [tcp/postgresql] succeeded!
"""

LOG_FINAL_CHECK_FAILED = """\
Task: Check that app-11 can connect to db-05 on port 5432.
$ fwctl rule add db-tier --from app-subnet --to db-05 --port 5432/tcp
[fwctl] rule fw-2207 added to policy db-tier
[fwctl] policy db-tier version 88 pushed
$ ssh app-11 nc -zv db-05.example.internal 5432
nc: connect to db-05.example.internal port 5432 (tcp) failed: Connection refused
"""

LOG_FINAL_CHECK_NEVER_RAN = """\
Task: Check that app-11 can connect to db-05 on port 5432.
$ fwctl rule add db-tier --from app-subnet --to db-05 --port 5432/tcp
[fwctl] rule fw-2207 added to policy db-tier
[fwctl] policy db-tier version 88 pushed
"""

RESPONSE_DONE = "Done. app-11 can connect to db-05 on port 5432."
RESPONSE_FAILED = "Failed. The connection from app-11 to db-05 on port 5432 was refused."
RESPONSE_NOT_SHOWN = "Not shown. The firewall rule was pushed, but the log does not show a connection test."

CHOICE_SCORES = {"A": 1, "B": 0, "C": 0, "D": 1, "E": 0}

# One case per choice. The model reply is mocked, so the choice is fixed by the test; the log and
# the response show the situation that choice describes.
CASES = [
    pytest.param(LOG_FINAL_CHECK_PASSED, RESPONSE_DONE, "A", 1, id="A-done-and-final-check-passed"),
    pytest.param(LOG_FINAL_CHECK_FAILED, RESPONSE_DONE, "B", 0, id="B-done-but-final-check-failed"),
    pytest.param(LOG_FINAL_CHECK_NEVER_RAN, RESPONSE_DONE, "C", 0, id="C-done-but-final-check-never-ran"),
    pytest.param(LOG_FINAL_CHECK_FAILED, RESPONSE_FAILED, "D", 1, id="D-failed-matches-the-log"),
    pytest.param(LOG_FINAL_CHECK_NEVER_RAN, RESPONSE_NOT_SHOWN, "D", 1, id="D-not-shown-matches-the-log"),
    pytest.param(LOG_FINAL_CHECK_PASSED, RESPONSE_NOT_SHOWN, "E", 0, id="E-not-done-but-final-check-passed"),
]


def _responses_api_reply(choice, reasons="Compared the status in the response with the log."):
    return {
        "id": "resp-test",
        "object": "response",
        "created": 1234567890,
        "model": "gpt-5-mini",
        "output": [
            {
                "type": "function_call",
                "call_id": "call_test",
                "name": "select_choice",
                "arguments": json.dumps({"reasons": reasons, "choice": choice}),
            }
        ],
    }


def test_template_file_loads():
    path = os.path.join(SCRIPT_DIR, "templates", "status_evidence.yaml")
    with open(path) as f:
        spec = yaml.safe_load(f)

    assert set(spec) == {"prompt", "choice_scores"}
    assert spec["choice_scores"] == CHOICE_SCORES
    assert "{{input}}" in spec["prompt"]
    assert "{{output}}" in spec["prompt"]


def test_scorer_loads_choice_scores_from_the_template():
    scorer = StatusEvidence()

    assert scorer.name == "StatusEvidence"
    assert scorer.choice_scores == CHOICE_SCORES

    # The model may only pick a lettered choice, and chain of thought is on by default.
    parameters = scorer.classification_tools[0]["function"]["parameters"]
    assert parameters["properties"]["choice"]["enum"] == ["A", "B", "C", "D", "E"]
    assert parameters["required"] == ["reasons", "choice"]


def test_prompt_lists_exactly_the_scored_choices():
    scorer = StatusEvidence()
    prompt = scorer.messages[0]["content"]

    assert re.findall(r"^\(([A-Z])\) ", prompt, flags=re.MULTILINE) == list(scorer.choice_scores)


def test_prompt_renders_input_and_output():
    scorer = StatusEvidence()

    request_args = scorer._request_args(output=RESPONSE_DONE, expected=None, input=LOG_FINAL_CHECK_FAILED)
    prompt = request_args["messages"][0]["content"]

    assert LOG_FINAL_CHECK_FAILED in prompt
    assert RESPONSE_DONE in prompt
    assert prompt.index(LOG_FINAL_CHECK_FAILED) < prompt.index(RESPONSE_DONE)
    assert "{{" not in prompt
    # The chain-of-thought suffix names the lettered choices.
    assert "['A', 'B', 'C', 'D', 'E']" in prompt


@respx.mock
@pytest.mark.parametrize("log,response,choice,score", CASES)
def test_mocked_reply_is_parsed_to_the_expected_score(log, response, choice, score):
    route = respx.post("https://api.openai.com/v1/responses").mock(
        return_value=Response(200, json=_responses_api_reply(choice))
    )
    # gpt-5 models use the Responses API; pin the model so the mocked route does not depend on the default.
    scorer = StatusEvidence(
        model="gpt-5-mini",
        client=OpenAI(api_key="test-api-key", base_url="https://api.openai.com/v1"),
    )

    result = scorer.eval(input=log, output=response)

    assert result.name == "StatusEvidence"
    assert result.score == score
    assert result.metadata["choice"] == choice
    assert result.metadata["rationale"] == "Compared the status in the response with the log."

    # The model was sent the log and the response.
    assert route.call_count == 1
    sent = json.loads(route.calls[0].request.content.decode("utf-8"))["input"][0]["content"]
    assert log in sent
    assert response in sent
