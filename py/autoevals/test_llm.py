import asyncio
import os

import chevron

from autoevals.llm import *
from autoevals.llm import build_classification_functions


def test_template_html():
    template_double = "{{output}}"
    template_triple = "{{{output}}}"

    assert chevron.render(template_double, dict(output="Template<Foo>")) == "Template&lt;Foo&gt;"
    assert chevron.render(template_triple, dict(output="Template<Foo>")) == "Template<Foo>"


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
        classification_functions=build_classification_functions(useCoT=True, choice_strings=["1", "2"]),
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
        assert response.score == (1 if use_cot else 0)
        assert response.error is None

        response = e(instructions="Add the following numbers: 1, 2, 3", output="6", expected="6")

        print(response.as_json(indent=2))
        assert response.score == 0
        assert response.error is None
