# Autoevals

Autoevals is a tool to quickly and easily evaluate AI model outputs.

It bundles together a variety of automatic evaluation methods including:

- LLM-as-a-judge
- Heuristic (e.g. Levenshtein distance)
- Statistical (e.g. BLEU)

Autoevals is developed by the team at [Braintrust](https://braintrust.dev/).

Autoevals uses model-graded evaluation for a variety of subjective tasks including fact checking,
safety, and more. Many of these evaluations are adapted from OpenAI's excellent [evals](https://github.com/openai/evals)
project but are implemented so you can flexibly run them on individual examples, tweak the prompts, and debug
their outputs.

You can also create your own model-graded evaluations with Autoevals. It's easy to add custom prompts, parse outputs,
and manage exceptions.

<div className="hidden">

### Requirements

- Python 3.9 or higher
- Compatible with both OpenAI Python SDK v0.x and v1.x

</div>

## Installation

<div className="tabs">

### TypeScript

```bash
npm install autoevals
```

### Python

```bash
pip install autoevals
```

</div>

## Getting started

Use Autoevals to model-grade an example LLM completion using the [Factuality prompt](templates/factuality.yaml).
By default, Autoevals uses your `OPENAI_API_KEY` environment variable to authenticate with OpenAI's API.

<div className="tabs">

### Python

```python
from autoevals.llm import *
import asyncio

# Create a new LLM-based evaluator
evaluator = Factuality()

# Synchronous evaluation
input = "Which country has the highest population?"
output = "People's Republic of China"
expected = "China"

# Using the synchronous API
result = evaluator(output, expected, input=input)
print(f"Factuality score (sync): {result.score}")
print(f"Factuality metadata (sync): {result.metadata['rationale']}")

# Using the asynchronous API
async def main():
    result = await evaluator.eval_async(output, expected, input=input)
    print(f"Factuality score (async): {result.score}")
    print(f"Factuality metadata (async): {result.metadata['rationale']}")

# Run the async example
asyncio.run(main())
```

### TypeScript

```typescript
import { Factuality } from "autoevals";

(async () => {
  const input = "Which country has the highest population?";
  const output = "People's Republic of China";
  const expected = "China";

  const result = await Factuality({ output, expected, input });
  console.log(`Factuality score: ${result.score}`);
  console.log(`Factuality metadata: ${result.metadata?.rationale}`);
})();
```

</div>

## Using other AI providers

When you use Autoevals, it will look for an `OPENAI_BASE_URL` environment variable to use as the base for requests to an OpenAI compatible API. If `OPENAI_BASE_URL` is not set, it will default to the [AI proxy](https://www.braintrust.dev/docs/guides/proxy).

If you choose to use the proxy, you'll also get:

- Simplified access to many AI providers
- Reduced costs with automatic request caching
- Increased observability when you enable logging to Braintrust

The proxy is free to use, even if you don't have a Braintrust account.

If you have a Braintrust account, you can optionally set the `BRAINTRUST_API_KEY` environment variable instead of `OPENAI_API_KEY` to unlock additional features like logging and monitoring. You can also route requests to [supported AI providers and models](https://www.braintrust.dev/docs/guides/proxy#supported-models) or custom models you have configured in Braintrust.

<div className="tabs">

### Python

```python
# NOTE: ensure BRAINTRUST_API_KEY is set in your environment and OPENAI_API_KEY is not set
from autoevals.llm import *

# Create an LLM-based evaluator using the Claude 3.5 Sonnet model from Anthropic
evaluator = Factuality(model="claude-3-5-sonnet-latest")

# Evaluate an example LLM completion
input = "Which country has the highest population?"
output = "People's Republic of China"
expected = "China"

result = evaluator(output, expected, input=input)

# The evaluator returns a score from [0,1] and includes the raw outputs from the evaluator
print(f"Factuality score: {result.score}")
print(f"Factuality metadata: {result.metadata['rationale']}")
```

### TypeScript

```typescript
// NOTE: ensure BRAINTRUST_API_KEY is set in your environment and OPENAI_API_KEY is not set
import { Factuality } from "autoevals";

(async () => {
  const input = "Which country has the highest population?";
  const output = "People's Republic of China";
  const expected = "China";

  // Run an LLM-based evaluator using the Claude 3.5 Sonnet model from Anthropic
  const result = await Factuality({
    model: "claude-3-5-sonnet-latest",
    output,
    expected,
    input,
  });

  // The evaluator returns a score from [0,1] and includes the raw outputs from the evaluator
  console.log(`Factuality score: ${result.score}`);
  console.log(`Factuality metadata: ${result.metadata?.rationale}`);
})();
```

</div>

## Custom client configuration

There are two ways you can configure a custom client when you need to use a different OpenAI compatible API:

1. **Global configuration**: Initialize a client that will be used by all evaluators
2. **Instance configuration**: Configure a client for a specific evaluator

### Global configuration

Set up a client that all your evaluators will use:

<div className="tabs">

#### Python

```python
import openai
import asyncio
from autoevals import init
from autoevals.llm import Factuality

client = init(openai.AsyncOpenAI(base_url="https://api.openai.com/v1/"))

async def main():
    evaluator = Factuality()
    result = await evaluator.eval_async(
        input="What is the speed of light in a vacuum?",
        output="The speed of light in a vacuum is 299,792,458 meters per second.",
        expected="The speed of light in a vacuum is approximately 300,000 kilometers per second."
    )
    print(f"Factuality score: {result.score}")

asyncio.run(main())
```

#### TypeScript

```typescript
import OpenAI from "openai";
import { init, Factuality } from "autoevals";

const client = new OpenAI({
  baseURL: "https://api.openai.com/v1/",
});

init({ client });

(async () => {
  const result = await Factuality({
    input: "What is the speed of light in a vacuum?",
    output: "The speed of light in a vacuum is 299,792,458 meters per second.",
    expected:
      "The speed of light in a vacuum is approximately 300,000 kilometers per second (or precisely 299,792,458 meters per second).",
  });

  console.log("Factuality Score:", result);
})();
```

</div>

### Instance configuration

Configure a client for a specific evaluator instance:

<div className="tabs">

#### Python

```python
import openai
from autoevals.llm import Factuality

custom_client = openai.OpenAI(base_url="https://custom-api.example.com/v1/")
evaluator = Factuality(client=custom_client)
```

#### TypeScript

```typescript
import OpenAI from "openai";
import { Factuality } from "autoevals";

(async () => {
  const customClient = new OpenAI({
    baseURL: "https://custom-api.example.com/v1/",
  });

  const result = await Factuality({
    client: customClient,
    output: "Paris is the capital of France",
    expected:
      "Paris is the capital of France and has a population of over 2 million",
    input: "Tell me about Paris",
  });
  console.log(result);
})();
```

</div>

## Using Braintrust with Autoevals (optional)

Once you grade an output using Autoevals, you can optionally use [Braintrust](https://www.braintrust.dev/docs/libs/python) to log and compare your evaluation results. This integration is completely optional and not required for using Autoevals.

<div className="tabs">

### TypeScript

Create a file named `example.eval.js` (it must take the form `*.eval.[ts|tsx|js|jsx]`):

```typescript
import { Eval } from "braintrust";
import { Factuality } from "autoevals";

Eval("Autoevals", {
  data: () => [
    {
      input: "Which country has the highest population?",
      expected: "China",
    },
  ],
  task: () => "People's Republic of China",
  scores: [Factuality],
});
```

Then, run

```bash
npx braintrust run example.eval.js
```

### Python

Create a file named `eval_example.py` (it must take the form `eval_*.py`):

```python
import braintrust
from autoevals.llm import Factuality

Eval(
    "Autoevals",
    data=lambda: [
        dict(
            input="Which country has the highest population?",
            expected="China",
        ),
    ],
    task=lambda *args: "People's Republic of China",
    scores=[Factuality],
)
```

</div>

## Supported evaluation methods

### LLM-as-a-judge evaluations

- Battle
- Closed QA
- Humor
- Factuality
- Moderation
- Security
- Summarization
- SQL
- Translation
- Fine-tuned binary classifiers

### RAG evaluations

- Context precision
- Context relevancy
- Context recall
- Context entity recall
- Faithfulness
- Answer relevancy
- Answer similarity
- Answer correctness

### Composite evaluations

- Semantic list contains
- JSON validity

### Embedding evaluations

- Embedding similarity

### Heuristic evaluations

- Levenshtein distance
- Exact match
- Numeric difference
- JSON diff

For detailed documentation on all scorers, including parameters, score ranges, and usage examples, see the [**Scorer Reference**](SCORERS.md).

## Custom evaluation prompts

Autoevals supports custom evaluation prompts for model-graded evaluation. To use them, simply pass in a prompt and scoring mechanism:

<div className="tabs">

### Python

```python
from autoevals import LLMClassifier

# Define a prompt prefix for a LLMClassifier (returns just one answer)
prompt_prefix = """
You are a technical project manager who helps software engineers generate better titles for their GitHub issues.
You will look at the issue description, and pick which of two titles better describes it.

I'm going to provide you with the issue description, and two possible titles.

Issue Description: {{input}}

1: {{output}}
2: {{expected}}
"""

# Define the scoring mechanism
# 1 if the generated answer is better than the expected answer
# 0 otherwise
output_scores = {"1": 1, "2": 0}

evaluator = LLMClassifier(
    name="TitleQuality",
    prompt_template=prompt_prefix,
    choice_scores=output_scores,
    use_cot=True,
)

# Evaluate an example LLM completion
page_content = """
As suggested by Nicolo, we should standardize the error responses coming from GoTrue, postgres, and realtime (and any other/future APIs) so that it's better DX when writing a client,
We can make this change on the servers themselves, but since postgrest and gotrue are fully/partially external may be harder to change, it might be an option to transform the errors within the client libraries/supabase-js, could be messy?
Nicolo also dropped this as a reference: http://spec.openapis.org/oas/v3.0.3#openapi-specification"""
output = "Standardize error responses from GoTrue, Postgres, and Realtime APIs for better DX"
expected = "Standardize Error Responses across APIs"

response = evaluator(output, expected, input=page_content)

print(f"Score: {response.score}")
print(f"Metadata: {response.metadata}")
```

### TypeScript

```typescript
import { LLMClassifierFromTemplate } from "autoevals";

(async () => {
  const promptTemplate = `You are a technical project manager who helps software engineers generate better titles for their GitHub issues.
You will look at the issue description, and pick which of two titles better describes it.

I'm going to provide you with the issue description, and two possible titles.

Issue Description: {{input}}

1: {{output}}
2: {{expected}}`;

  const choiceScores = { 1: 1, 2: 0 };

  const evaluator = LLMClassifierFromTemplate<{ input: string }>({
    name: "TitleQuality",
    promptTemplate,
    choiceScores,
    useCoT: true,
  });

  const input = `As suggested by Nicolo, we should standardize the error responses coming from GoTrue, postgres, and realtime (and any other/future APIs) so that it's better DX when writing a client,
We can make this change on the servers themselves, but since postgrest and gotrue are fully/partially external may be harder to change, it might be an option to transform the errors within the client libraries/supabase-js, could be messy?
Nicolo also dropped this as a reference: http://spec.openapis.org/oas/v3.0.3#openapi-specification`;
  const output = `Standardize error responses from GoTrue, Postgres, and Realtime APIs for better DX`;
  const expected = `Standardize Error Responses across APIs`;

  const response = await evaluator({ input, output, expected });

  console.log("Score", response.score);
  console.log("Metadata", response.metadata);
})();
```

</div>

## Creating custom scorers

You can also create your own scoring functions that do not use LLMs. For example, to test whether the word `'banana'`
is in the output, you can use the following:

<div className="tabs">

### Python

```python
from autoevals import Score

def banana_scorer(output, expected, input):
    return Score(name="banana_scorer", score=1 if "banana" in output else 0)

input = "What is 1 banana + 2 bananas?"
output = "3"
expected = "3 bananas"

result = banana_scorer(output, expected, input)

print(f"Banana score: {result.score}")
```

### TypeScript

```typescript
import { Score } from "autoevals";

const bananaScorer = ({
  output,
  expected,
  input,
}: {
  output: string;
  expected: string;
  input: string;
}): Score => {
  return { name: "banana_scorer", score: output.includes("banana") ? 1 : 0 };
};

(async () => {
  const input = "What is 1 banana + 2 bananas?";
  const output = "3";
  const expected = "3 bananas";

  const result = bananaScorer({ output, expected, input });
  console.log(`Banana score: ${result.score}`);
})();
```

</div>

## Why does this library exist?

There is nothing particularly novel about the evaluation methods in this library. They are all well-known and well-documented. However, there are a few things that are particularly difficult when evaluating in practice:

- Normalizing metrics between 0 and 1 is tough. For example, check out the calculation in [number.py](/py/autoevals/number.py) to see how it's done for numeric differences.
- Parsing the outputs on model-graded evaluations is also challenging. There are frameworks that do this, but it's hard to
  debug one output at a time, propagate errors, and tweak the prompts. Autoevals makes these tasks easy.
- Collecting metrics behind a uniform interface makes it easy to swap out evaluation methods and compare them. Prior to Autoevals, we couldn't find an open source library where you can simply pass in `input`, `output`, and `expected` values through a bunch of different evaluation methods.

<div className="hidden">

## Documentation

The full docs are available [for your reference](https://www.braintrust.dev/docs/reference/autoevals).

## Contributing

We welcome contributions!

To install the development dependencies, run `make develop`, and run `source env.sh` to activate the environment. Make a `.env` file from the `.env.example` file and set the environment variables. Run `direnv allow` to load the environment variables.

To run the tests, run `pytest` from the root directory.

Send a PR and we'll review it! We'll take care of versioning and releasing.

</div>
