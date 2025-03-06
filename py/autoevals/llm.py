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

from .oai import Client, arun_cached_request, run_cached_request

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
    """Base class for OpenAI API integration.

    Args:
        api_key: Deprecated. Use client instead.
        base_url: Deprecated. Use client instead.
        client: Optional Client. If not provided, uses global client from init().
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        client: Optional[Client] = None,
    ) -> None:
        """Initialize an OpenAI scorer.

        This is the base class for OpenAI-based scorers. It handles authentication and API configuration.

        Args:
            api_key: Deprecated. Use client instead.
            base_url: Deprecated. Use client instead.
            client: Optional Client. If not provided, uses global client from init().

        Note:
            The api_key and base_url parameters are deprecated and will be removed in a future version.
            Instead, you can either:
            1. Pass a client instance directly to this constructor using the client parameter
            2. Set a global client using autoevals.init(client=your_client)

            The global client can be configured once and will be used by all evaluators that don't have
            a specific client passed to them.
        """
        self.extra_args = {}
        if api_key:
            self.extra_args["api_key"] = api_key
        if base_url:
            self.extra_args["base_url"] = base_url

        self.client = client


class OpenAILLMScorer(OpenAIScorer):
    """Base class for LLM-specific scoring functionality.

    Args:
        temperature: Controls randomness (0 = focused, 1 = creative)
        api_key: Deprecated. Use client.
        base_url: Deprecated. Use client.
        client: Optional Client. If not provided, uses global client from init().
    """

    def __init__(
        self,
        temperature: Optional[float] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        client: Optional[Client] = None,
    ) -> None:
        """Initialize an OpenAI LLM scorer.

        This class extends OpenAIScorer with LLM-specific functionality.

        Args:
            temperature: The sampling temperature to use for the model.
            api_key: Deprecated. See base class for details.
            base_url: Deprecated. See base class for details.
            client: Optional Client. If not provided, uses global client from init().

        See Also:
            OpenAIScorer: Base class that handles authentication and API configuration.
        """
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            client=client,
        )
        self.extra_args["temperature"] = temperature or 0


class OpenAILLMClassifier(OpenAILLMScorer):
    """Base class for LLM-based classification.

    Args:
        name: Classifier name
        messages: Conversation messages for the model
        model: Model to use (e.g. 'gpt-4')
        choice_scores: Maps choices to scores
        classification_tools: Available tools for classification
        render_args: Template rendering arguments
        max_tokens: Max tokens to generate
        temperature: Randomness (0-1)
        engine: Deprecated. Use client.
        api_key: Deprecated. Use client.
        base_url: Deprecated. Use client.
        client: Optional Client. If not provided, uses global client from init().
    """

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
        client: Optional[Client] = None,
    ):
        """Initialize an OpenAI LLM classifier.

        This class extends OpenAILLMScorer to provide classification functionality using LLMs.

        Args:
            name: The name of the classifier.
            messages: List of messages to use for the classification.
            model: The model to use for classification.
            choice_scores: Dictionary mapping choices to their scores.
            classification_tools: List of tools available for classification.
            render_args: Optional dictionary of arguments for message rendering.
            max_tokens: Optional maximum number of tokens to generate.
            temperature: Optional sampling temperature.
            engine: Deprecated. Use client.
            api_key: Deprecated. Use client.
            base_url: Deprecated. Use client.
            client: Optional Client. If not provided, uses global client from init().

        See Also:
            OpenAILLMScorer: Parent class that handles LLM functionality.
            OpenAIScorer: Base class that handles authentication and API configuration.
        """
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
    """High-level classifier for evaluating text using LLMs.

    This is the main class for building custom classifiers. It provides:
    - Chain of thought reasoning for better accuracy
    - Standardized output parsing
    - Template-based prompts
    - YAML configuration support
    - Flexible scoring rules

    Example:
        ```python
        init(OpenAI(api_key="your_api_key"))

        classifier = LLMClassifier(
            name="toxicity",
            prompt_template="Rate if this text is toxic: {{output}}",
            choice_scores={"toxic": 0, "not_toxic": 1},
            model="gpt-4",
            use_cot=True
        )
        result = classifier.eval(output="some text")
        ```

    Args:
        name: Classifier name for tracking
        prompt_template: Template for generating prompts (supports {{output}}, {{expected}}, etc.)
        choice_scores: Dictionary mapping choices to scores (e.g. {"good": 1, "bad": 0})
        model: Model to use (default: gpt-4)
        use_cot: Enable chain of thought reasoning for better accuracy (default: True)
        max_tokens: Maximum tokens to generate (default: 512)
        temperature: Controls randomness (0-1, default: 0)
        engine: Deprecated by OpenAI
        api_key: Deprecated. Use client.
        base_url: Deprecated. Use client.
        client: Optional Client. If not provided, uses global client from init().
        **extra_render_args: Additional template variables
    """

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
        client: Optional[Client] = None,
        **extra_render_args,
    ):
        """Initialize an LLM classifier.

        Args:
            name: Name of the classifier.
            prompt_template: Template string for generating prompts.
            choice_scores: Dictionary mapping choices to their scores.
            model: Model to use for classification (default: gpt-4).
            use_cot: Whether to use chain of thought reasoning (default: True).
            max_tokens: Maximum number of tokens to generate (default: 512).
            temperature: Sampling temperature (default: 0).
            engine: Deprecated. Use client.
            api_key: Deprecated. Use client.
            base_url: Deprecated. Use client.
            client: Optional Client. If not provided, uses global client from init().
            **extra_render_args: Additional arguments to pass to the template renderer.
        """
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
    def from_spec(cls, name: str, spec: ModelGradedSpec, client: Optional[Client] = None, **kwargs):
        return cls(name, spec.prompt, spec.choice_scores, client=client, **kwargs)

    @classmethod
    def from_spec_file(cls, name: str, path: str, client: Optional[Client] = None, **kwargs):
        if cls._SPEC_FILE_CONTENTS[name] == "":
            with open(path) as f:
                cls._SPEC_FILE_CONTENTS[name] = f.read()
        spec = yaml.safe_load(cls._SPEC_FILE_CONTENTS[name])
        return cls.from_spec(name, ModelGradedSpec(**spec), client=client, **kwargs)


class SpecFileClassifier(LLMClassifier):
    """Base class for creating specialized classifiers from YAML templates.

    This class simplifies creating new evaluation types by:
    - Loading config from standardized YAML templates
    - Supporting common evaluation patterns
    - Allowing parameter overrides

    To create a new classifier:
    1. Create a YAML file in 'templates' directory
    2. Subclass SpecFileClassifier
    3. The template name is derived from the class name (FooBar -> foo_bar.yaml)

    Example:
        ```python
        class Toxicity(SpecFileClassifier):
            pass

        classifier = Toxicity(temperature=0.5)  # Override defaults
        ```

    YAML Template Format:
        ```yaml
        prompt: Template string for prompts
        choice_scores:
          choice1: score1
          choice2: score2
        model: gpt-4  # optional
        use_cot: true  # optional
        temperature: 0  # optional
        ```

    Args:
        model: Override template model
        engine: Deprecated. Use client.
        use_cot: Override chain of thought
        max_tokens: Override max tokens
        temperature: Override temperature
        api_key: Deprecated. Use client.
        base_url: Deprecated. Use client.
        client: Optional Client. If not provided, uses global client from init().
    """

    def __new__(
        cls,
        model=None,
        engine=None,
        use_cot=None,
        max_tokens=None,
        temperature=None,
        api_key=None,
        base_url=None,
        client: Optional[Client] = None,
    ):
        """Create a new classifier instance from a YAML template.

        Args:
            model: Optional model override for the template.
            engine: Optional engine override for the template.
            use_cot: Optional chain of thought override for the template.
            max_tokens: Optional maximum tokens override for the template.
            temperature: Optional temperature override for the template.
            api_key: Deprecated. See OpenAIScorer for details.
            base_url: Deprecated. See OpenAIScorer for details.
            client: Optional Client. If not provided, uses global client from init().

        Returns:
            LLMClassifier: A new classifier instance configured from the template.

        Raises:
            AttributeError: If the template file for this class is not found.
        """
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
    """Compare if a solution performs better than a reference.

    Example:
        ```python
        battle = Battle()
        result = battle.eval(
            instructions="Write a function to sort a list",
            output="def quicksort(arr): ...",
            expected="def bubblesort(arr): ..."
        )
        print(result.score)  # 1 if better, 0 if worse
        print(result.metadata["rationale"])  # Explanation
        ```

    Args:
        instructions: Task description
        output: Solution to evaluate
        expected: Reference solution

    See Also:
        SpecFileClassifier: Base class for creating specialized classifiers from YAML templates.
    """

    pass


class ClosedQA(SpecFileClassifier):
    """Evaluate answer correctness using the model's knowledge.

    Example:
        ```python
        qa = ClosedQA()
        result = qa.eval(
            input="What is the capital of France?",
            output="Paris",
            criteria="Must be exact city name"
        )
        print(result.score)  # 1 if correct, 0 if incorrect
        ```

    Args:
        input: Question to evaluate
        output: Answer to assess
        criteria: Optional evaluation criteria

    See Also:
        SpecFileClassifier: Base class for creating specialized classifiers from YAML templates.
    """

    pass


class Humor(SpecFileClassifier):
    """Rate the humor level in text.

    Example:
        ```python
        humor = Humor()
        result = humor.eval(
            output="Why did the developer quit? They didn't get arrays!"
        )
        print(result.score)  # 1 if funny, 0 if not
        print(result.metadata["rationale"])  # Explanation
        ```

    Args:
        output: Text to evaluate for humor
    """

    pass


class Factuality(SpecFileClassifier):
    """Check factual accuracy against a reference.

    Example:
        ```python
        factual = Factuality()
        result = factual.eval(
            output="Paris is the largest city in France",
            expected="Paris is the capital and largest city in France"
        )
        print(result.score)  # 1 if accurate, 0 if inaccurate
        ```

    Args:
        output: Text to check
        expected: Reference text with correct facts
    """

    pass


class Possible(SpecFileClassifier):
    """Evaluate if a solution is feasible and practical.

    Example:
        ```python
        possible = Possible()
        result = possible.eval(
            input="Design a system to handle 1M users",
            output="We'll use a distributed architecture..."
        )
        print(result.score)  # 1 if feasible, 0 if not
        ```

    Args:
        input: Problem description
        output: Proposed solution
    """

    pass


class Security(SpecFileClassifier):
    """Check for security risks in code or text.

    Example:
        ```python
        security = Security()
        result = security.eval(
            output="SELECT * FROM users WHERE id = " + user_input
        )
        print(result.score)  # 0 if risky, 1 if safe
        print(result.metadata["rationale"])  # Found issues
        ```

    Args:
        output: Code/text to analyze
    """

    pass


class Sql(SpecFileClassifier):
    """Compare if two SQL queries are equivalent.

    Example:
        ```python
        sql = Sql()
        result = sql.eval(
            output="SELECT * FROM users WHERE age >= 18",
            expected="SELECT * FROM users WHERE age > 17"
        )
        print(result.score)  # 1 if equivalent, 0 if different
        ```

    Args:
        output: SQL query to check
        expected: Reference SQL query
    """

    pass


class Summary(SpecFileClassifier):
    """Evaluate text summarization quality.

    Example:
        ```python
        summary = Summary()
        result = summary.eval(
            input="Long article text...",
            output="Brief summary...",
            expected="Reference summary..."
        )
        print(result.score)  # Higher is better
        ```

    Args:
        input: Original text
        output: Generated summary
        expected: Reference summary
    """

    pass


class Translation(SpecFileClassifier):
    """Evaluate translation quality.

    Example:
        ```python
        translation = Translation()
        result = translation.eval(
            input="Hello world!",
            output="¡Hola mundo!",
            expected="¡Hola mundo!",
            language="Spanish"
        )
        print(result.score)  # Higher is better
        ```

    Args:
        input: Source text
        output: Translation to evaluate
        expected: Reference translation
        language: Target language
    """

    pass
