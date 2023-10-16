import json
import os
import re
from dataclasses import dataclass
from typing import List, Optional

import chevron
import openai
import yaml

from .base import Score, Scorer
from .oai import arun_cached_request, run_cached_request
from .util import current_span

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

SUPPORTED_MODELS = [
    "gpt-3.5-turbo",
    "gpt-4",
]


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
            "items": {"type": "string"},
            "title": "Reasons",
            "type": "array",
        },
        "choice": {"description": "The choice", "title": "Choice", "type": "string"},
    },
    "required": ["reasons", "choice"],
    "title": "CoTResponse",
    "type": "object",
}


def build_classification_functions(useCoT):
    return [
        {
            "name": "select_choice",
            "description": "Call this function to select a choice.",
            "parameters": COT_RESPONSE_SCHEMA if useCoT else PLAIN_RESPONSE_SCHEMA,
        }
    ]


class OpenAILLMClassifier(Scorer):
    def __init__(
        self,
        name: str,
        messages: List,
        model,
        choice_scores,
        classification_functions,
        render_args=None,
        max_tokens=None,
        temperature=None,
        engine=None,
    ):
        self.name = name
        self.model = model
        self.engine = engine
        self.messages = messages
        self.choice_scores = choice_scores
        self.classification_functions = classification_functions

        self.extra_args = {"temperature": temperature or 0}
        if max_tokens:
            self.extra_args["max_tokens"] = max(max_tokens, 5)

        self.render_args = {}
        if render_args:
            self.render_args.update(render_args)

    def _name(self):
        return self.name

    def _process_response(self, resp):
        metadata = {}
        try:
            args = json.loads(resp["function_call"]["arguments"])
            if "reasons" in args:
                metadata["rationale"] = "\n".join(args["reasons"])
            if "function_call" not in resp:
                raise ValueError("No function call found in response")
            metadata["choice"] = args["choice"].strip()
            score = self.choice_scores[metadata["choice"]]
            error = None
        except Exception as e:
            score = 0
            error = e

        return Score(name=self.name, score=score, metadata=metadata, error=error)

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
        return dict(
            Completion=openai.ChatCompletion,
            model=self.model,
            engine=self.engine,
            messages=self._render_messages(output=output, expected=expected, **kwargs),
            functions=self.classification_functions,
            function_call={"name": "select_choice"},
            **self.extra_args,
        )

    def _postprocess_response(self, resp):
        if len(resp["choices"]) > 0:
            return self._process_response(resp["choices"][0]["message"])
        else:
            raise ValueError("Empty response from OpenAI")

    async def _run_eval_async(self, output, expected, **kwargs):
        validity_score = 1
        try:
            return self._postprocess_response(
                await arun_cached_request(**self._request_args(output, expected, **kwargs))
            )
        except Exception as e:
            validity_score = 0
            return Score(name=self.name, score=0, error=e)
        finally:
            current_span().log(scores={f"{self._name()} parsed": validity_score})

    def _run_eval_sync(self, output, expected, **kwargs):
        validity_score = 1
        try:
            return self._postprocess_response(run_cached_request(**self._request_args(output, expected, **kwargs)))
        except Exception as e:
            validity_score = 0
            return Score(name=self.name, score=0, error=e)
        finally:
            current_span().log(scores={f"{self._name()} parsed": validity_score})


@dataclass
class ModelGradedSpec:
    prompt: str
    choice_scores: dict[str, float]
    model: Optional[str] = None
    engine: Optional[str] = None
    use_cot: Optional[bool] = None
    temperature: Optional[float] = None


# XXX: Document that prompts are expected to be mustache templates
class LLMClassifier(OpenAILLMClassifier):
    """
    An LLM-based classifier that wraps `OpenAILLMClassifier` and provides a standard way to
    apply chain of thought, parse the output, and score the result."""

    def __init__(
        self,
        name,
        prompt_template,
        choice_scores,
        model="gpt-3.5-turbo",
        use_cot=True,
        max_tokens=512,
        temperature=0,
        engine=None,
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
            name,
            messages,
            model,
            choice_scores,
            classification_functions=build_classification_functions(use_cot),
            max_tokens=max_tokens,
            temperature=temperature,
            engine=engine,
            render_args={"__choices": choice_strings},
        )

    @classmethod
    def from_spec(cls, name: str, spec: ModelGradedSpec, **kwargs):
        return cls(name, spec.prompt, spec.choice_scores, **kwargs)

    @classmethod
    def from_spec_file(cls, name: str, path: str, **kwargs):
        with open(path) as f:
            spec = yaml.safe_load(f)
        return cls.from_spec(name, ModelGradedSpec(**spec), **kwargs)


class SpecFileClassifier(LLMClassifier):
    def __new__(cls, model=None, engine=None, use_cot=None, max_tokens=None, temperature=None):
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

        # convert FooBar to foo_bar
        template_name = re.sub(r"(?<!^)(?=[A-Z])", "_", cls.__name__).lower()

        template_path = os.path.join(SCRIPT_DIR, "templates", template_name + ".yaml")
        if not os.path.exists(template_path):
            raise AttributeError(f"Model template {cls.__name__} not found")

        return LLMClassifier.from_spec_file(cls.__name__, template_path, **kwargs)


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


class Summary(SpecFileClassifier):
    """
    Test whether an output is a better summary of the `input` than the original (`expected`) value."""

    pass


class Translation(SpecFileClassifier):
    """
    Test whether an `output` is as good of a translation of the `input` in the specified `language`
    as an expert (`expected`) value.."""

    pass
