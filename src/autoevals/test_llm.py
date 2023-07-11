import os

import diskcache
from guidance.llms.caches import DiskCache


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(SCRIPT_DIR, "../../.testcache")
os.makedirs(CACHE_DIR, exist_ok=True)


def diskcache_init(self, llm_name: str):
    self._diskcache = diskcache.Cache(os.path.join(CACHE_DIR, f"_{llm_name}.diskcache"))


DiskCache.__init__ = diskcache_init


import re

import guidance

from autoevals.llm import *


def test_guidance():
    def parse_best_title(grade):
        return int(re.findall(r"Winner: (\d+)", grade["summary"])[0])

    e = GuidanceLLMClassifier(
        guidance(
            """
  {{#system~}}
  You are a technical project manager who helps software engineers generate better titles for their GitHub issues.
  You will look at the issue description, and pick which of two titles better describes it.
  {{~/system}}

  {{#user~}}
  I'm going to provide you with the issue description, and two possible titles.

  Issue Description: {{page_content}}

  Title 1: {{output}}
  Title 2: {{expected}}

  Please discuss each title briefly (one line for pros, one for cons), and then pick which one you think more accurately
  summarizes the issue by writing "Winner: 1" or "Winner: 2", and then a short rationale for your choice.
  {{~/user}}

  {{#assistant~}}
  {{gen 'summary' max_tokens=__max_tokens temperature=0}}
  {{~/assistant}}""",
            llm=guidance.llms.OpenAI("gpt-3.5-turbo"),
        ),
        parse_best_title,
        {1: 1, 2: 0},
    )

    page_content = """
    As suggested by Nicolo, we should standardize the error responses coming from GoTrue, postgres, and realtime (and any other/future APIs) so that it's better DX when writing a client,

    We can make this change on the servers themselves, but since postgrest and gotrue are fully/partially external may be harder to change, it might be an option to transform the errors within the client libraries/supabase-js, could be messy?

    Nicolo also dropped this as a reference: http://spec.openapis.org/oas/v3.0.3#openapi-specification"""

    gen_title = "Standardize error responses from GoTrue, Postgres, and Realtime APIs for better DX"
    original_title = "Standardize Error Responses across APIs"

    response = e(gen_title, original_title, page_content=page_content, __max_tokens=500)
    print(response.as_json(indent=2))
    assert response.score == 1
    assert response.error is None
