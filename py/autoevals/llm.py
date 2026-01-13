"""LLM-based evaluation scorers for assessing model outputs.

This module provides a collection of pre-built LLM scorers for common evaluation tasks.

All evaluators accept the following common arguments:
- model: Model to use (defaults to gpt-4o)
- temperature: Controls randomness (0-1). If not specified, uses the model's default.
- max_tokens: Maximum tokens to generate. If not specified, uses the model's default.
- client: OpenAI client (defaults to global client from init())

Example:
```python
from openai import OpenAI
from autoevals import Battle, Factuality, ClosedQA, init

# Initialize with your OpenAI client (or pass client= to individual scorers)
init(OpenAI())

# Compare solutions
battle = Battle()
result = battle.eval(
    instructions="Write a function to sort a list",
    output="def quicksort(arr): ...",
    expected="def bubblesort(arr): ..."
)
print(result.score)  # 1 if better, 0 if worse
print(result.metadata["rationale"])  # Explanation of comparison

# Check factual accuracy
factual = Factuality()
result = factual.eval(
    output="Paris is the largest city in France",
    expected="Paris is the capital and largest city in France"
)
print(result.score)  # 1 if accurate, 0 if inaccurate

# Evaluate answer correctness
qa = ClosedQA()
result = qa.eval(
    input="What is the capital of France?",
    output="Paris",
    criteria="Must be exact city name"
)
print(result.score)  # 1 if correct, 0 if incorrect
```
"""

import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass

import chevron
import yaml

from autoevals.partial import ScorerWithPartial

from .oai import Client, arun_cached_request, run_cached_request
from .score import Score

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
        api_key: str | None = None,
        base_url: str | None = None,
        client: Client | None = None,
    ) -> None:
        self.extra_args = {}
        if api_key:
            self.extra_args["api_key"] = api_key
        if base_url:
            self.extra_args["base_url"] = base_url

        self.client = client


class OpenAILLMScorer(OpenAIScorer):
    def __init__(
        self,
        temperature: float | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        client: Client | None = None,
    ) -> None:
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            client=client,
        )
        if temperature is not None:
            self.extra_args["temperature"] = temperature


class OpenAILLMClassifier(OpenAILLMScorer):
    def __init__(
        self,
        name: str,
        messages: list,
        model,
        choice_scores,
        classification_tools,
        render_args=None,
        max_tokens=None,
        temperature=None,
        engine=None,
        api_key=None,
        base_url=None,
        client: Client | None = None,
    ):
        super().__init__(
            client=client,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
        )

        self.name = name

        self.model = model
        self.engine = engine
        self.messages = messages

        if max_tokens is not None:
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
    choice_scores: dict[str, float]
    model: str | None = None
    engine: str | None = None
    use_cot: bool | None = None
    temperature: float | None = None
    max_tokens: int | None = None


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
        from openai import OpenAI
        from autoevals import init
        from autoevals.llm import LLMClassifier

        # Create a classifier for toxicity evaluation
        classifier = LLMClassifier(
            name="toxicity",  # Name for tracking
            prompt_template="Rate if this text is toxic: {{output}}",  # Template with variables
            choice_scores={"toxic": 0, "not_toxic": 1},  # Mapping choices to scores
            client=OpenAI()  # Optional: could use init() to set a global client instead
        )

        # Evaluate some text
        result = classifier.eval(output="some text to evaluate")
        print(result.score)  # Score between 0-1 based on choice_scores
        print(result.metadata)  # Additional evaluation details
        ```

    Args:
        name: Classifier name for tracking
        prompt_template: Template for generating prompts (supports `{{output}}`, `{{expected}}`, etc.)
        choice_scores: Mapping of choices to scores (e.g. `{"good": 1, "bad": 0}`)
        model: Model to use. Defaults to DEFAULT_MODEL.
        use_cot: Enable chain of thought reasoning. Defaults to True.
        max_tokens: Maximum tokens to generate. If not specified, uses the model's default.
        temperature: Controls randomness (0-1). If not specified, uses the model's default.
        engine: Deprecated by OpenAI. Use model instead.
        api_key: Deprecated. Use client instead.
        base_url: Deprecated. Use client instead.
        client: OpenAI client. If not provided, uses global client from init().
        **extra_render_args: Additional template variables
    """

    _SPEC_FILE_CONTENTS: dict[str, str] = defaultdict(str)

    def __init__(
        self,
        name,
        prompt_template,
        choice_scores,
        model=DEFAULT_MODEL,
        use_cot=True,
        max_tokens=None,
        temperature=None,
        engine=None,
        api_key=None,
        base_url=None,
        client: Client | None = None,
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
    def from_spec(cls, name: str, spec: ModelGradedSpec, client: Client | None = None, **kwargs):
        spec_kwargs = {}
        if spec.model is not None:
            spec_kwargs["model"] = spec.model
        if spec.engine is not None:
            spec_kwargs["engine"] = spec.engine
        if spec.use_cot is not None:
            spec_kwargs["use_cot"] = spec.use_cot
        if spec.temperature is not None:
            spec_kwargs["temperature"] = spec.temperature
        if spec.max_tokens is not None:
            spec_kwargs["max_tokens"] = spec.max_tokens
        # kwargs can override spec values
        return cls(name, spec.prompt, spec.choice_scores, client=client, **spec_kwargs, **kwargs)

    @classmethod
    def from_spec_file(cls, name: str, path: str, client: Client | None = None, **kwargs):
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
        client: Client | None = None,
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
    """Compare if a solution performs better than a reference solution.

    This evaluator uses LLM-based comparison to determine if a generated solution is better
    than a reference solution, considering factors like:
    - Code quality and readability
    - Algorithm efficiency and complexity
    - Implementation completeness
    - Best practices and patterns
    - Error handling and edge cases

    Example:
        ```python
        import asyncio
        from openai import AsyncOpenAI
        from autoevals import Battle

        async def evaluate_solutions():
            # Initialize with async client
            client = AsyncOpenAI()
            battle = Battle(client=client)

            result = await battle.eval_async(
                instructions="Write a function to sort a list of integers in ascending order",
                output='''
                    def quicksort(arr):
                        if len(arr) <= 1:
                            return arr
                        pivot = arr[len(arr) // 2]
                        left = [x for x in arr if x < pivot]
                        middle = [x for x in arr if x == pivot]
                        right = [x for x in arr if x > pivot]
                        return quicksort(left) + middle + quicksort(right)
                ''',
                expected='''
                    def bubblesort(arr):
                        n = len(arr)
                        for i in range(n):
                            for j in range(0, n - i - 1):
                                if arr[j] > arr[j + 1]:
                                    arr[j], arr[j + 1] = arr[j + 1], arr[j]
                        return arr
                '''
            )

            print(result.score)  # 1 if output is better, 0 if worse
            print(result.metadata["rationale"])  # Detailed comparison explanation
            print(result.metadata["choice"])  # Selected choice (better/worse)

        # Run the async evaluation
        asyncio.run(evaluate_solutions())
        ```

    Args:
        instructions: Problem description or task requirements that both solutions should address
        output: Solution to evaluate (code, text, or other content)
        expected: Reference solution to compare against

    Returns:
        Score object with:
        - score: 1 if output solution is better, 0 if worse
        - metadata.rationale: Detailed explanation of the comparison
        - metadata.choice: Selected choice (better/worse)
    """

    pass


class ClosedQA(SpecFileClassifier):
    """Evaluate answer correctness using the model's knowledge.

    Example:
        ```python
        from autoevals import ClosedQA, init
        from openai import OpenAI

        init(OpenAI())

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
    """

    pass


class Humor(SpecFileClassifier):
    """Rate the humor level in text.

    Example:
        ```python
        from autoevals import Humor, init
        from openai import OpenAI

        init(OpenAI())

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
        from autoevals import Factuality, init
        from openai import OpenAI

        init(OpenAI())

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
        from autoevals import Possible, init
        from openai import OpenAI

        init(OpenAI())

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
    """Evaluate if a solution has security vulnerabilities.

    This evaluator uses LLM-based analysis to identify potential security issues
    in code or system designs, checking for common vulnerabilities like:
    - Injection attacks (SQL, command, etc.)
    - Authentication/authorization flaws
    - Data exposure risks
    - Input validation issues
    - Unsafe dependencies
    - Insecure configurations
    - Common OWASP vulnerabilities

    Example:
        ```python
        import asyncio
        from openai import AsyncOpenAI
        from autoevals import Security

        async def evaluate_security():
            # Initialize with async client
            client = AsyncOpenAI()
            security = Security(client=client)

            result = await security.eval_async(
                instructions="Write a function to execute a SQL query with user input",
                output='''
                    def execute_query(user_input):
                        query = f"SELECT * FROM users WHERE name = '{user_input}'"
                        cursor.execute(query)
                        return cursor.fetchall()
                '''
            )

            print(result.score)  # 0 if vulnerable, 1 if secure
            print(result.metadata["rationale"])  # Detailed security analysis
            print(result.metadata["choice"])  # Selected choice (secure/vulnerable)

        # Run the async evaluation
        asyncio.run(evaluate_security())
        ```

    Args:
        instructions: Context or requirements for the security evaluation
        output: Code or system design to evaluate for security issues

    Returns:
        Score object with:
        - score: 1 if secure, 0 if vulnerable
        - metadata.rationale: Detailed security analysis
        - metadata.choice: Selected choice (secure/vulnerable)
        - metadata.vulnerabilities: List of identified security issues
    """

    pass


class Sql(SpecFileClassifier):
    """Compare if two SQL queries are equivalent.

    Example:
        ```python
        from autoevals import Sql, init
        from openai import OpenAI

        init(OpenAI())

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
        from openai import OpenAI
        from autoevals import Summary, init

        init(OpenAI())

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
        from openai import OpenAI
        from autoevals import Translation

        translation = Translation(client=OpenAI())
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
