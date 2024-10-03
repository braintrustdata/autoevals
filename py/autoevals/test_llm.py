import asyncio

from unittest import mock

import chevron
import openai

from autoevals.llm import *
from autoevals.llm import build_classification_tools


def test_template_html():
    template_double = "{{output}}"
    template_triple = "{{{output}}}"

    assert chevron.render(template_double, dict(output="Template<Foo>")) == "Template&lt;Foo&gt;"
    assert chevron.render(template_triple, dict(output="Template<Foo>")) == "Template<Foo>"


def test_custom_client():
    custom_client = openai.OpenAI()
    scorer = LLMClassifier(
        name="Comparator",
        prompt_template=(
            "Evaluate two responses and indicate which better answers the user question.\n"
            "Question:{{input}}\n"
            "Response A: {{output}}\nResponse B: {{expected}}\n\n."
            "Your output should be one of A, B, or Tie."
        ),
        choice_scores={"A": 1, "B": 0, "Tie": 0.5},
        use_cot=True,
        client=custom_client,
    )

    class MockResponse:
        def dict(self):
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "function": {
                                        "arguments": '{"reasons":"...","choice":"Tie"}',
                                        "name": "select_choice",
                                    },
                                    "type": "function",
                                }
                            ],
                        },
                    }
                ],
                "usage": {
                    "completion_tokens": 65,
                    "prompt_tokens": 220,
                    "total_tokens": 285,
                },
            }

    with mock.patch.object(
        custom_client.chat.completions,
        "create",
        return_value=MockResponse(),
    ) as m:
        score = scorer.eval(input="What is the capital of France?", output="Paris", expected="Paris")
        assert score.score == 0.5
        m.assert_called_once()


def test_openai():
    e = OpenAILLMClassifier(
        "title",
        messages=[
            {
                "role": "system",
                "content": """\
You are a technical project manager who helps software engineers generate better titles for their GitHub issues.
You will look at the issue description, and pick which of two titles better describes it.""",
            },
            {
                "role": "user",
                "content": """\
I'm going to provide you with the issue description, and two possible titles.

Issue Description: {{page_content}}

1: {{output}}
2: {{expected}}

Please discuss each title briefly (one line for pros, one for cons), and then answer the question by calling
the select_choice function with "1" or "2".""",
            },
        ],
        model="gpt-3.5-turbo",
        choice_scores={"1": 1, "2": 0},
        classification_tools=build_classification_tools(useCoT=True, choice_strings=["1", "2"]),
        max_tokens=500,
    )

    page_content = """
As suggested by Nicolo, we should standardize the error responses coming from GoTrue, postgres, and realtime (and any other/future APIs) so that it's better DX when writing a client,

We can make this change on the servers themselves, but since postgrest and gotrue are fully/partially external may be harder to change, it might be an option to transform the errors within the client libraries/supabase-js, could be messy?

Nicolo also dropped this as a reference: http://spec.openapis.org/oas/v3.0.3#openapi-specification"""

    gen_title = "Standardize error responses from GoTrue, Postgres, and Realtime APIs for better DX"
    original_title = "This title has nothing to do with the content"

    response = e(gen_title, original_title, page_content=page_content)
    print(response.as_json(indent=2))
    assert response.score == 1
    assert response.error is None


def test_llm_classifier():
    for use_cot in [True, False]:
        e = LLMClassifier(
            "title",
            """
You are a technical project manager who helps software engineers generate better titles for their GitHub issues.
You will look at the issue description, and pick which of two titles better describes it.

I'm going to provide you with the issue description, and two possible titles.

Issue Description: {{page_content}}

1: {{output}}
2: {{expected}}""",
            {"1": 1, "2": 0},
            use_cot=use_cot,
        )

        page_content = """
As suggested by Nicolo, we should standardize the error responses coming from GoTrue, postgres, and realtime (and any other/future APIs) so that it's better DX when writing a client,

We can make this change on the servers themselves, but since postgrest and gotrue are fully/partially external may be harder to change, it might be an option to transform the errors within the client libraries/supabase-js, could be messy?

Nicolo also dropped this as a reference: http://spec.openapis.org/oas/v3.0.3#openapi-specification"""

        gen_title = "Standardize error responses from GoTrue, Postgres, and Realtime APIs for better DX"
        original_title = "This title has nothing to do with the content"

        response = e(gen_title, original_title, page_content=page_content)
        print(response.as_json(indent=2))
        assert response.score == 1
        assert response.error is None

        response = e(original_title, gen_title, page_content=page_content)
        print(response.as_json(indent=2))
        assert response.score == 0
        assert response.error is None


def test_nested_async():
    async def nested_async():
        e = Battle()
        e(instructions="Add the following numbers: 1, 2, 3", output="600", expected="6")

    asyncio.run(nested_async())


def test_battle():
    for use_cot in [True, False]:
        print("use_cot", use_cot)
        e = Battle(use_cot=use_cot)
        response = e(
            instructions="Add the following numbers: 1, 2, 3",
            output="600",
            expected="6",
        )

        print(response.as_json(indent=2))
        assert response.score == 0
        assert response.error is None

        response = e(
            instructions="Add the following numbers: 1, 2, 3",
            output="6",
            expected="600",
        )

        print(response.as_json(indent=2))
        assert response.score == 1
        assert response.error is None

        response = e(instructions="Add the following numbers: 1, 2, 3", output="6", expected="6")

        print(response.as_json(indent=2))
        assert response.score == 0
        assert response.error is None
