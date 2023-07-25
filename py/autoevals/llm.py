import os
import re
from dataclasses import dataclass
from typing import List, Optional

import chevron
import openai
import yaml

from .base import Score, Scorer
from .oai import arun_cached_request, run_cached_request


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
                "content": chevron.render(m["content"].strip(), kwargs),
            }
            for m in self.messages
        ]

    async def _run_eval_async(self, output, expected, **kwargs):
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
            return Score(name=self.name, score=0, error=e)

    def _run_eval_sync(self, output, expected, **kwargs):
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
            return Score(name=self.name, score=0, error=e)


@dataclass
class ModelGradedSpec:
    prompt: str
    choice_scores: dict[str, float]
    model: Optional[str] = None
    use_cot: Optional[bool] = None
    temperature: Optional[float] = None


# XXX: Document that prompts are expected to be mustache templates
class LLMClassifier(OpenAILLMClassifier):
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

        # XXX we should parse the prompt and make sure it has the right variables, and track the
        # required inputs (so we can validate them when running the prompt)
        prompt = prompt_template + "\n" + (COT_SUFFIX if use_cot else NO_COT_SUFFIX)
        if use_cot:

            def parse_score_fn(resp):
                answer = None

                answers = re.findall(r"Answer\s*=\s*(.*)", resp, re.MULTILINE)
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


def _build_template_class(name: str):
    class C(LLMClassifier):
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
            template_name = re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()

            template_path = os.path.join(SCRIPT_DIR, "..", "..", "templates", template_name + ".yaml")
            if not os.path.exists(template_path):
                raise AttributeError(f"Model template {name} not found")

            return LLMClassifier.from_spec_file(template_name, template_path, **kwargs)

    return C


# This makes static analysis tools happier
Battle = _build_template_class("Battle")
ClosedQA = _build_template_class("ClosedQA")
Humor = _build_template_class("Humor")
Factuality = _build_template_class("Factuality")
Possible = _build_template_class("Possible")
Security = _build_template_class("Security")
Summary = _build_template_class("Summary")
Translation = _build_template_class("Translation")
