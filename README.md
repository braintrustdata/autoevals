# Autoevals

Autoevals is a tool to quickly and easily evaluate AI model outputs.

It bundles together a variety of automatic evaluation methods including:

- LLM-as-a-Judge
- Heuristic (e.g. Levenshtein distance)
- Statistical (e.g. BLEU)

Autoevals is developed by the team at [Braintrust](https://braintrust.dev/).

Autoevals uses model-graded evaluation for a variety of subjective tasks including fact checking,
safety, and more. Many of these evaluations are adapted from OpenAI's excellent [evals](https://github.com/openai/evals)
project but are implemented so you can flexibly run them on individual examples, tweak the prompts, and debug
their outputs.

You can also create your own model-graded evaluations with Autoevals. It's easy to add custom prompts, parse outputs,
and manage exceptions.

## Installation

Autoevals is distributed as a [Python library on PyPI](https://pypi.org/project/autoevals/) and
[Node.js library on NPM](https://www.npmjs.com/package/autoevals).

### Python

```bash
pip install autoevals
```

### Node.js

```bash
npm install autoevals
```

## Example

Use Autoevals to model-grade an example LLM completion using the [factuality prompt](templates/factuality.yaml).
By default, Autoevals uses your `OPENAI_API_KEY` environment variable to authenticate with OpenAI's API.

### Python

```python
from autoevals.llm import *

# Create a new LLM-based evaluator
evaluator = Factuality()

# Evaluate an example LLM completion
input = "Which country has the highest population?"
output = "People's Republic of China"
expected = "China"

result = evaluator(output, expected, input=input)

# The evaluator returns a score from [0,1] and includes the raw outputs from the evaluator
print(f"Factuality score: {result.score}")
print(f"Factuality metadata: {result.metadata['rationale']}")
```

#### Use with other AI providers through the AI proxy

Autoevals will look for an `OPENAI_BASE_URL` environment variable to use as the base for requests to an OpenAI compatible API. If `OPENAI_BASE_URL` is not set, it will default to the [AI proxy](https://www.braintrust.dev/docs/guides/proxy). This provides numerous benefits like simplified access to many AI providers, reduced costs with automatic request caching, and increased observability when you enable logging to Braintrust. The proxy is free to use, even if you don't have a Braintrust account.

If you have a Braintrust account, you can set the `BRAINTUST_API_KEY` environment variable instead of `OPENAI_API_KEY` to unlock additional features like logging and monitoring. Additionally, you can route requests to [supported AI providers and models](https://www.braintrust.dev/docs/guides/proxy#supported-models) or custom models you have configured in Braintrust.

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

#### Custom Client

If you need to use a different OpenAI compatible API or require custom behavior, you can initialize the library with a custom client.

```python
import openai
from autoevals import init
from autoevals.oai import LLMClient

openai_client = openai.OpenAI(base_url="https://api.openai.com/v1/")

class CustomClient(LLMClient):
    openai=openai_client  # you can also pass in openai module and we will instantiate it for you
    embed = openai.embeddings.create
    moderation = openai.moderations.create
    RateLimitError = openai.RateLimitError

    def complete(self, **kwargs):
        # make adjustments as needed
        return self.openai.chat.completions.create(**kwargs)

# Autoevals will now use your custom client
client = init(client=CustomClient)
```

If you only need to use a custom client for a specific evaluator, you can pass in the client to the evaluator.

```python
evaluator = Factuality(client=CustomClient)
```

### Node.js

```javascript
import { Factuality } from "autoevals";

(async () => {
  const input = "Which country has the highest population?";
  const output = "People's Republic of China";
  const expected = "China";

  const result = await Factuality({ output, expected, input });
  console.log(`Factuality score: ${result.score}`);
  console.log(`Factuality metadata: ${result.metadata.rationale}`);
})();
```

#### Use with other AI providers through the AI proxy

Autoevals will look for an `OPENAI_BASE_URL` environment variable to use as the base for requests to an OpenAI compatible API. If `OPENAI_BASE_URL` is not set, it will default to the [AI proxy](https://www.braintrust.dev/docs/guides/proxy). This provides numerous benefits like simplified access to many AI providers, reduced costs with automatic request caching, and increased observability when you enable logging to Braintrust. The proxy is free to use, even if you don't have a Braintrust account.

If you have a Braintrust account, you can set the `BRAINTUST_API_KEY` environment variable instead of `OPENAI_API_KEY` to unlock additional features like logging and monitoring. Additionally, you can route requests to [supported AI providers and models](https://www.braintrust.dev/docs/guides/proxy#supported-models) or custom models you have configured in Braintrust.

```javascript
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

## Using Braintrust with Autoevals

Once you grade an output using Autoevals, it's convenient to use [Braintrust](https://www.braintrust.dev/docs/libs/python) to log and compare your evaluation results.

### Python

```python
from autoevals.llm import *
import braintrust

# Create a new LLM-based evaluator
evaluator = Factuality()

# Set up an example LLM completion
input = "Which country has the highest population?"
output = "People's Republic of China"
expected = "China"

# Set up a BrainTrust experiment to log our eval to
experiment = braintrust.init(
    project="Autoevals", api_key="YOUR_BRAINTRUST_API_KEY"
)

# Start a span and run our evaluator
with experiment.start_span() as span:
    result = evaluator(output, expected, input=input)

    # The evaluator returns a score from [0,1] and includes the raw outputs from the evaluator
    print(f"Factuality score: {result.score}")
    print(f"Factuality metadata: {result.metadata['rationale']}")

    span.log(
        inputs={"query": input},
        output=output,
        expected=expected,
        scores={
            "factuality": result.score,
        },
        metadata={
            "factuality": result.metadata,
        },
    )

print(experiment.summarize())
```

### Node.js

Create a file named `example.eval.js` (it must end with `.eval.js` or `.eval.js`):

```javascript
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

## Supported Evaluation Methods

### LLM-as-a-Judge

- Battle
- ClosedQA
- Humor
- Factuality
- Moderation
- Security
- Summarization
- SQL
- Translation
- Fine-tuned binary classifiers

### RAG

- Context precision
- Context relevancy
- Context recall
- Context entities recall
- Faithfullness
- Answer relevance
- Answer semantic similarity
- Answer correctness
- Aspect critique

### Composite

- Semantic list contains
- JSON validity

### Embeddings

- Embedding similarity
- BERTScore

### Heuristic

- Levenshtein distance
- Exact match
- Numeric difference
- JSON diff
- Jaccard distance

### Statistical

- BLEU
- ROUGE
- METEOR

## Custom Evaluation Prompts

Autoevals supports custom evaluation prompts for model-graded evaluation. To use them, simply pass in a prompt and scoring mechanism:

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
output = (
    "Standardize error responses from GoTrue, Postgres, and Realtime APIs for better DX"
)
expected = "Standardize Error Responses across APIs"

response = evaluator(output, expected, input=page_content)

print(f"Score: {response.score}")
print(f"Metadata: {response.metadata}")
```

### Node.js

```javascript
import { LLMClassifierFromTemplate } from "autoevals";

(async () => {
  const promptTemplate = `You are a technical project manager who helps software engineers generate better titles for their GitHub issues.
You will look at the issue description, and pick which of two titles better describes it.

I'm going to provide you with the issue description, and two possible titles.

Issue Description: {{input}}

1: {{output}}
2: {{expected}}`;

  const choiceScores = { 1: 1, 2: 0 };

  const evaluator =
    LLMClassifierFromTemplate <
    { input: string } >
    {
      name: "TitleQuality",
      promptTemplate,
      choiceScores,
      useCoT: true,
    };

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

## Creating custom scorers

You can also create your own scoring functions that do not use LLMs. For example, to test whether the word `'banana'`
is in the output, you can use the following:

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

### Node.js

```javascript
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

## Why does this library exist?

There is nothing particularly novel about the evaluation methods in this library. They are all well-known and well-documented. However, there are a few things that are particularly difficult when evaluating in practice:

- Normalizing metrics between 0 and 1 is tough. For example, check out the calculation in [number.py](/py/autoevals/number.py) to see how it's done for numeric differences.
- Parsing the outputs on model-graded evaluations is also challenging. There are frameworks that do this, but it's hard to
  debug one output at a time, propagate errors, and tweak the prompts. Autoevals makes these tasks easy.
- Collecting metrics behind a uniform interface makes it easy to swap out evaluation methods and compare them. Prior to Autoevals, we couldn't find an open source library where you can simply pass in `input`, `output`, and `expected` values through a bunch of different evaluation methods.

## Documentation

The full docs are available [here](https://www.braintrust.dev/docs/reference/autoevals).
