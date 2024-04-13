"""
AutoEvals is a tool to quickly and easily evaluate AI model outputs.

### Quickstart

```bash
pip install autoevals
```

### Example

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

"""

from braintrust_core.score import Score, Scorer

from .json import *
from .list import *
from .llm import *
from .moderation import *
from .number import *
from .ragas import *
from .string import *
