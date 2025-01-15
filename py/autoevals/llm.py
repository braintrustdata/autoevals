import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional

import chevron
import yaml
from braintrust_core.score import Score

from autoevals.partial import ScorerWithPartial

from .oai import LLMClient, arun_cached_request, run_cached_request

# Disable HTML escaping in chevron.
chevron.renderer._html_escape = lambda x: x  # type: ignore[attr-defined]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

NO_COT_SUFFIX = """\
Answer the question by calling `select_choice` with a single choice from {{__choices}}.
""".strip().replace(
    "\n", " "
)

COT_SUFFIX = """\
Answer the question by calling `select_choice` with your reasoning in a step-by-step matter to be
sure that your conclusion is correct. Avoid simply stating the correct answer at the outset. Select a
single choice by setting the `choice` parameter to a single choice from {{__choices}}.
""".strip().replace(
    "\n", " "
)

DEFAULT_MODEL = "gpt-4o"

PLAIN_RESPONSE_SCHEMA = {
    "properties": {"choice": {"description": "The choice", "title": "Choice", "type": "string"}},
    "required": ["choice"],
    "title": "PlainResponse",
    "type": "object",
}

COT_RESPONSE_SCHEMA = {
    "properties": {
        "reasons": {
            "description": "Write out in a step by step manner your reasoning to be sure that your conclusion is correct. Avoid simply stating the correct answer at the outset.",
            "title": "Reasoning",
            "type": "string",
        },
        "choice": {"description": "The choice", "title": "Choice", "type": "string"},
    },
    "required": ["reasons", "choice"],
    "title": "CoTResponse",
    "type": "object",
}


def build_classification_tools(useCoT, choice_strings):
    params = COT_RESPONSE_SCHEMA if useCoT else PLAIN_RESPONSE_SCHEMA
    enum_params = {
        **params,
        "properties": {
            **params["properties"],
            "choice": {**params["properties"]["choice"], "enum": choice_strings},
        },
    }
    return [
        {
            "type": "function",
            "function": {
                "name": "select_choice",
                "description": "Call this function to select a choice.",
                "parameters": enum_params,
            },
        }
    ]


class OpenAIScorer(ScorerWithPartial):
    def __init__(
        self,
        api_key=None,
        base_url=None,
        client: Optional[LLMClient] = None,
    ):
        self.extra_args = {}
        if api_key:
            self.extra_args["api_key"] = api_key
        if base_url:
            self.extra_args["base_url"] = base_url

        self.client = client


class OpenAILLMScorer(OpenAIScorer):
    def __init__(
        self,
        temperature=None,
        api_key=None,
        base_url=None,
        client: Optional[LLMClient] = None,
    ):
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            client=client,
        )
        self.extra_args["temperature"] = temperature or 0


class OpenAILLMClassifier(OpenAILLMScorer):
    def __init__(
        self,
        name: str,
        messages: List,
        model,
        choice_scores,
        classification_tools,
        render_args=None,
        max_tokens=None,
        temperature=None,
        engine=None,
        api_key=None,
        base_url=None,
        client: Optional[LLMClient] = None,
    ):
        super().__init__(
            client=client,
            api_key=api_key,
            base_url=base_url,
        )

        self.name = name

        self.model = model
        self.engine = engine
        self.messages = messages

        self.extra_args["temperature"] = temperature or 0

        if max_tokens:
            self.extra_args["max_tokens"] = max(max_tokens, 5)

        self.render_args = {}
        if render_args:
            self.render_args.update(render_args)

        self.choice_scores = choice_scores
        self.classification_tools = classification_tools

    def _name(self):
        return self.name

    def _build_args(self, output, expected, **kwargs):
        return dict(
            model=self.model,
            messages=self._render_messages(output=output, expected=expected, **kwargs),
            tools=self.classification_tools,
            tool_choice={"type": "function", "function": {"name": "select_choice"}},
        )

    def _render_messages(self, **kwargs):
        kwargs.update(self.render_args)
        return [
            {
                **m,
                "content": chevron.render(m["content"].strip(), kwargs, warn=True),
            }
            for m in self.messages
        ]

    def _request_args(self, output, expected, **kwargs):
        ret = {
            "client": self.client,
            **self.extra_args,
            **self._build_args(output, expected, **kwargs),
        }

        if self.engine is not None:
            # this parameter has been deprecated (https://help.openai.com/en/articles/6283125-what-happened-to-engines)
            # and is unsupported in openai v1, so only set it if the user has specified it
            ret["engine"] = self.engine

        return ret

    def _process_response(self, resp):
        metadata = {}
        if "tool_calls" not in resp:
            raise ValueError("No tool call found in response")
        tool_call = resp["tool_calls"][0]
        if tool_call["function"]["name"] != "select_choice":
            raise ValueError(f"Unexpected tool call ({tool_call['function']['name']}) found in response")
        args = json.loads(tool_call["function"]["arguments"])

        metadata["choice"] = args["choice"].strip()
        if "reasons" in args:
            metadata["rationale"] = (
                "\n".join(args["reasons"]) if isinstance(args["reasons"], list) else args["reasons"]
            )

        score = self.choice_scores[metadata["choice"]]
        return Score(name=self.name, score=score, metadata=metadata)

    def _postprocess_response(self, resp):
        if len(resp["choices"]) > 0:
            return self._process_response(resp["choices"][0]["message"])
        else:
            raise ValueError("Empty response from OpenAI")

    async def _run_eval_async(self, output, expected, **kwargs):
        return self._postprocess_response(await arun_cached_request(**self._request_args(output, expected, **kwargs)))

    def _run_eval_sync(self, output, expected, **kwargs):
        return self._postprocess_response(run_cached_request(**self._request_args(output, expected, **kwargs)))


@dataclass
class ModelGradedSpec:
    prompt: str
    choice_scores: Dict[str, float]
    model: Optional[str] = None
    engine: Optional[str] = None
    use_cot: Optional[bool] = None
    temperature: Optional[float] = None


class LLMClassifier(OpenAILLMClassifier):
    """
    An LLM-based classifier that wraps `OpenAILLMClassifier` and provides a standard way to
    apply chain of thought, parse the output, and score the result."""

    _SPEC_FILE_CONTENTS: Dict[str, str] = defaultdict(str)

    def __init__(
        self,
        name,
        prompt_template,
        choice_scores,
        model=DEFAULT_MODEL,
        use_cot=True,
        max_tokens=512,
        temperature=0,
        engine=None,
        api_key=None,
        base_url=None,
        client: Optional[LLMClient] = None,
        **extra_render_args,
    ):
        choice_strings = list(choice_scores.keys())

        prompt = prompt_template + "\n" + (COT_SUFFIX if use_cot else NO_COT_SUFFIX)
        messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]

        super().__init__(
            name=name,
            messages=messages,
            model=model,
            choice_scores=choice_scores,
            classification_tools=build_classification_tools(use_cot, choice_strings),
            max_tokens=max_tokens,
            temperature=temperature,
            engine=engine,
            api_key=api_key,
            base_url=base_url,
            render_args={"__choices": choice_strings, **extra_render_args},
            client=client,
        )

    @classmethod
    def from_spec(cls, name: str, spec: ModelGradedSpec, client: Optional[LLMClient] = None, **kwargs):
        return cls(name, spec.prompt, spec.choice_scores, client=client, **kwargs)

    @classmethod
    def from_spec_file(cls, name: str, path: str, client: Optional[LLMClient] = None, **kwargs):
        if cls._SPEC_FILE_CONTENTS[name] == "":
            with open(path) as f:
                cls._SPEC_FILE_CONTENTS[name] = f.read()
        spec = yaml.safe_load(cls._SPEC_FILE_CONTENTS[name])
        return cls.from_spec(name, ModelGradedSpec(**spec), client=client, **kwargs)


class SpecFileClassifier(LLMClassifier):
    def __new__(
        cls,
        model=None,
        engine=None,
        use_cot=None,
        max_tokens=None,
        temperature=None,
        api_key=None,
        base_url=None,
        client: Optional[LLMClient] = None,
    ):
        kwargs = {}
        if model is not None:
            kwargs["model"] = model
        if engine is not None:
            kwargs["engine"] = engine
        if use_cot is not None:
            kwargs["use_cot"] = use_cot
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if temperature is not None:
            kwargs["temperature"] = temperature
        if api_key is not None:
            kwargs["api_key"] = api_key
        if base_url is not None:
            kwargs["base_url"] = base_url

        # convert FooBar to foo_bar
        cls_name = cls.__name__
        template_name = re.sub(r"(?<!^)(?=[A-Z])", "_", cls_name).lower()

        template_path = os.path.join(SCRIPT_DIR, "templates", template_name + ".yaml")
        if not os.path.exists(template_path):
            raise AttributeError(f"Model template {cls_name} not found")

        extra_render_args = cls._partial_args() if hasattr(cls, "_partial_args") else {}

        return LLMClassifier.from_spec_file(cls_name, template_path, client=client, **kwargs, **extra_render_args)


class Battle(SpecFileClassifier):
    """
    Test whether an output _better_ performs the `instructions` than the original
    (`expected`) value."""

    pass


class ClosedQA(SpecFileClassifier):
    """
    Test whether an output answers the `input` using knowledge built into the model. You
    can specify `criteria` to further constrain the answer."""

    pass


class Humor(SpecFileClassifier):
    """
    Test whether an output is funny."""

    pass


class Factuality(SpecFileClassifier):
    """
    Test whether an output is factual, compared to an original (`expected`) value."""

    pass


class Possible(SpecFileClassifier):
    """
    Test whether an output is a possible solution to the challenge posed in the input."""

    pass


class Security(SpecFileClassifier):
    """
    Test whether an output is malicious."""

    pass


class Sql(SpecFileClassifier):
    """
    Test whether a SQL query is semantically the same as a reference (output) query."""

    pass


class Summary(SpecFileClassifier):
    """
    Test whether an output is a better summary of the `input` than the original (`expected`) value."""

    pass


class Translation(SpecFileClassifier):
    """
    Test whether an `output` is as good of a translation of the `input` in the specified `language`
    as an expert (`expected`) value.."""

    pass
