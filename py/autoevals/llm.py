import os
import re
from dataclasses import dataclass
from typing import Annotated, List, Optional

import chevron
import openai
import yaml

from .base import Score, Scorer
from .oai import arun_cached_request, run_cached_request
from .util import current_span

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

NO_COT_SUFFIX = """\
Answer the question by printing only a single choice from {{__choices}} (without quotes or punctuation)
corresponding to the correct answer with no other text.
""".strip().replace(
    "\n", " "
)

COT_SUFFIX = """\
Write out in a step by step manner your reasoning to be sure that your conclusion is correct. Avoid
simply stating the correct answer at the outset. Then print only a single choice from {{__choices}} (without
quotes or punctuation) on its own line corresponding to the correct answer. At the end, repeat just the
answer by itself on a new line formatted as "Answer=X"
""".strip().replace(
    "\n", " "
)

SUPPORTED_MODELS = [
    "gpt-3.5-turbo",
    "gpt-4",
]


class OpenAILLMClassifier(Scorer):
    def __init__(
        self,
        name: str,
        messages: List,
        model,
        parse_score_fn,
        choice_scores,
        render_args=None,
        max_tokens=None,
        temperature=None,
    ):
        found = False
        for m in SUPPORTED_MODELS:
            # Prefixes are ok, because they are just time snapshots
            if model.startswith(m):
                found = True
                break
        if not found:
            raise ValueError(f"Unsupported model: {model}. Currently only supports OpenAI chat models.")

        self.name = name
        self.model = model
        self.messages = messages
        self.parse_score_fn = parse_score_fn
        self.choice_scores = choice_scores

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
            metadata["rationale"] = str(resp)
            metadata["choice"] = self.parse_score_fn(resp)
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

    async def _run_eval_async(self, output, expected, **kwargs):
        validity_score = 1
        try:
            resp = await arun_cached_request(
                openai.ChatCompletion,
                model=self.model,
                messages=self._render_messages(output=output, expected=expected, **kwargs),
                **self.extra_args,
            )
            if len(resp["choices"]) > 0:
                return self._process_response(resp["choices"][0]["message"]["content"])
            else:
                raise ValueError("Empty response from OpenAI")
        except Exception as e:
            validity_score = 0
            return Score(name=self.name, score=0, error=e)
        finally:
            current_span().log(scores={f"{self._name()} parsed": validity_score})

    def _run_eval_sync(self, output, expected, **kwargs):
        validity_score = 1
        try:
            resp = run_cached_request(
                openai.ChatCompletion,
                model=self.model,
                messages=self._render_messages(output=output, expected=expected, **kwargs),
                **self.extra_args,
            )
            if len(resp["choices"]) > 0:
                return self._process_response(resp["choices"][0]["message"]["content"])
            else:
                raise ValueError("Empty response from OpenAI")
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
    ):
        choice_strings = list(choice_scores.keys())

        prompt = prompt_template + "\n" + (COT_SUFFIX if use_cot else NO_COT_SUFFIX)
        if use_cot:

            def parse_score_fn(resp):
                answer = None

                answers = re.findall(r"Answer\s*[=:]\s*(.*)", resp, re.MULTILINE)
                if len(answers) > 0:
                    answer = answers[-1].strip()
                elif resp.strip() in choice_strings:
                    answer = resp.strip()

                if answer is None:
                    raise ValueError("No answer found in response")

                return answer

        else:
            max_tokens = max(len(c) for c in choice_strings)

            def parse_score_fn(resp):
                return resp.strip()

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
            parse_score_fn,
            choice_scores,
            max_tokens=max_tokens,
            temperature=temperature,
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
    def __new__(cls, model=None, use_cot=None, max_tokens=None, temperature=None):
        kwargs = {}
        if model is not None:
            kwargs["model"] = model
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
