import os
import re
from dataclasses import dataclass
from typing import Any, Union

import guidance
import yaml

from .base import Evaluation, Evaluator


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_TEMPLATES = set(["Battle", "ClosedQA", "Humor", "Factuality", "Possible", "Security", "Summary", "Translation"])


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


class GuidanceLLMClassifier(Evaluator):
    def __init__(self, program: Union[str, guidance.Program], parse_score_fn, choice_scores):
        if isinstance(program, str):
            program = guidance.Program(program)

        if program.llm is None:
            raise ValueError("GuidanceLLMClassifier requires an LLM")

        self.program = program
        self.parse_score_fn = parse_score_fn
        self.choice_scores = choice_scores

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

        return Evaluation(score=score, metadata=metadata, error=error)

    async def _run_eval_async(self, output, expected, **kwargs):
        try:
            resp = await self.program(output=output, expected=expected, async_mode=True, **kwargs)
            return self._process_response(resp)
        except Exception as e:
            return Evaluation(score=0, error=e)

    def _run_eval_sync(self, output, expected, **kwargs):
        try:
            resp = self.program(output=output, expected=expected, async_mode=False, **kwargs)
            return self._process_response(resp)
        except Exception as e:
            return Evaluation(score=0, error=e)


@dataclass
class ModelGradedSpec:
    prompt: str
    choice_scores: dict[str, float]


# XXX: Document that prompts are expected to be mustache templates
class LLMClassifier(GuidanceLLMClassifier):
    def __init__(
        self,
        prompt_template,
        choice_scores,
        model="gpt-3.5-turbo",
        use_cot=True,
        max_tokens=512,
        temperature=0,
    ):
        found = False
        for m in SUPPORTED_MODELS:
            # Prefixes are ok, because they are just time snapshots
            if model.startswith(m):
                found = True
                break
        if not found:
            raise ValueError(f"Unsupported model: {model}. Currently only supports OpenAI chat models.")

        choice_strings = list(choice_scores.keys())

        # XXX we should parse the prompt and make sure it has the right variables, and track the
        # required inputs (so we can validate them when running the prompt)
        prompt = prompt_template + "\n" + (COT_SUFFIX if use_cot else NO_COT_SUFFIX)
        if use_cot:

            def parse_score_fn(resp):
                answer = None

                answers = re.findall(r"Answer\s*=\s*(.*)", resp["answer"], re.MULTILINE)
                if len(answers) > 0:
                    answer = answers[-1].strip()
                elif resp["answer"].strip() in choice_strings:
                    answer = resp["answer"].strip()

                if answer is None:
                    raise ValueError("No answer found in response")

                return answer

        else:
            max_tokens = max(len(c) for c in choice_strings)

            def parse_score_fn(resp):
                return resp["answer"].strip()

        program = guidance(
            "{{#user~}}\n"
            + prompt
            + "{{~/user}}"
            + """

{{#assistant~}}
{{gen 'answer' max_tokens=__max_tokens temperature=__temperature}}
{{~/assistant}}''')
""".strip(),
            llm=guidance.llms.OpenAI(model),
            __max_tokens=max_tokens,
            __temperature=temperature,
            __choices=choice_strings,
        )

        super().__init__(program, parse_score_fn, choice_scores)

    @classmethod
    def from_spec(cls, spec: ModelGradedSpec, **kwargs):
        return cls(spec.prompt, spec.choice_scores, **kwargs)

    @classmethod
    def from_spec_file(cls, path: str, **kwargs):
        with open(path) as f:
            spec = yaml.safe_load(f)
        return cls.from_spec(ModelGradedSpec(**spec), **kwargs)


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
            template_name = re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower() + ".yaml"

            if not os.path.exists(os.path.join(SCRIPT_DIR, "templates", template_name)):
                raise AttributeError(f"Model template {name} not found")

            return LLMClassifier.from_spec_file(os.path.join(SCRIPT_DIR, "templates", template_name), **kwargs)

    return C


def _init_model_templates():
    for model_template in MODEL_TEMPLATES:
        globals()[model_template] = _build_template_class(model_template)


_init_model_templates()
__all__ = ["GuidanceLLMClassifier", "LLMClassifier"] + list(MODEL_TEMPLATES)
