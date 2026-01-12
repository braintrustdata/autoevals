"""Autoevals is a comprehensive toolkit for evaluating AI model outputs.

This library provides a collection of specialized scorers for different types of evaluations:

- `string`: Text similarity using edit distance or embeddings
- `llm`: LLM-based evaluation for correctness, complexity, security, etc.
- `moderation`: Content safety and policy compliance checks
- `ragas`: Advanced NLP metrics for RAG system evaluation
- `json`: JSON validation and structural comparison
- `number`: Numeric similarity with relative scaling
- `value`: Exact matching and basic comparisons

**Key features**:

- Both sync and async evaluation support
- Configurable scoring parameters
- Detailed feedback through metadata
- Integration with OpenAI and other LLM providers through Braintrust AI Proxy

**Client setup**:

There are two ways to configure the OpenAI client:

1. Global initialization (recommended):

```python
from autoevals import init
from openai import AsyncOpenAI

# Set up once at the start of your application
client = AsyncOpenAI()
init(client=client)
```

2. Per-evaluator initialization:

```python
from openai import AsyncOpenAI
from autoevals.ragas import ClosedQA

# Pass client directly to evaluator
client = AsyncOpenAI()
evaluator = ClosedQA(client=client)
```

**Multi-provider support via the Braintrust AI Proxy**:

Autoevals supports multiple LLM providers (Anthropic, Azure, etc.) through the Braintrust AI Proxy.
Configure your client to use the proxy and set the default model:

```python
import os
from openai import AsyncOpenAI
from autoevals import init
from autoevals.llm import Factuality

# Configure client to use Braintrust AI Proxy with Claude
client = AsyncOpenAI(
    base_url="https://api.braintrust.dev/v1/proxy",
    api_key=os.getenv("BRAINTRUST_API_KEY"),
)

# Initialize with the client and default model
init(client=client, default_model="claude-3-5-sonnet-20241022")

# All evaluators will now use Claude by default
evaluator = Factuality()
result = evaluator.eval(input="...", output="...", expected="...")
```

**Braintrust integration**:

Autoevals automatically integrates with Braintrust logging when you install the library. If needed, you can manually wrap the client:

```python
from openai import AsyncOpenAI
from braintrust import wrap_openai
from autoevals.ragas import ClosedQA

# Explicitly wrap the client if needed
client = wrap_openai(AsyncOpenAI())
evaluator = ClosedQA(client=client)
```

**Example Autoevals usage**:

```python
from autoevals.ragas import ClosedQA
import asyncio

async def evaluate_qa():
    # Create evaluator for question answering
    evaluator = ClosedQA()

    # Question and context
    question = "What was the purpose of the Apollo missions?"
    context = '''
    The Apollo program was a NASA space program that ran from 1961 to 1972,
    with the goal of landing humans on the Moon and bringing them safely back
    to Earth. The program achieved its most famous success when Apollo 11
    astronauts Neil Armstrong and Buzz Aldrin became the first humans to walk
    on the Moon on July 20, 1969.
    '''

    # Two different answers to evaluate
    answer = "The Apollo program's main goal was to land humans on the Moon and return them safely to Earth."
    expected = "The Apollo missions were designed to achieve human lunar landing and safe return."

    # Evaluate the answer
    result = await evaluator.eval_async(
        question=question,
        context=context,
        output=answer,
        expected=expected
    )

    print(f"Score: {result.score}")  # Semantic similarity score (0-1)
    print(f"Rationale: {result.metadata.rationale}")  # Detailed explanation
    print(f"Faithfulness: {result.metadata.faithfulness}")  # Context alignment

# Run async evaluation
asyncio.run(evaluate_qa())
```

See individual module documentation for detailed usage and options.
"""

from .json import *
from .list import *
from .llm import *
from .moderation import *
from .number import *
from .oai import get_default_model, init
from .ragas import *
from .score import Score, Scorer, SerializableDataClass
from .string import *
from .value import ExactMatch
