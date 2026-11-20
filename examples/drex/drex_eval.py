"""Braintrust experiment that judges answers with Drex.

    pip install autoevals braintrust
    export NACE_API_KEY=nace_sk_...      # https://console.nace.ai
    export BRAINTRUST_API_KEY=...
    braintrust eval examples/drex/drex_eval.py

Drex returns probabilities. The scorer keeps them in each score's metadata, so you can pick a threshold
in Braintrust instead of guessing one up front.
"""

from autoevals.drex import DrexClassifier, DrexQuestion
from braintrust import Eval

DATA = [
    {
        "input": {"question": "What is the capital of France?", "context": "Paris is the capital of France."},
        "output": "Paris.",
    },
    {
        "input": {"question": "What is the capital of France?", "context": "Paris is the capital of France."},
        "output": "Lyon, since 1960.",
    },
]

grounded = DrexClassifier(
    DrexQuestion.noul("Is every claim in the output supported by the context?"),
    name="Grounded",
    # Default model is drex-max. For large text-only rubrics Drex may require model="drex-v2.0".
)

helpful = DrexClassifier(
    DrexQuestion.score(
        ["useless", "partly useful", "useful", "very useful"], instructions="How useful is the output?"
    ),
    name="Helpful",
)


def task(case):
    # Replace with your own application. Here the "application" returns the prepared output.
    return case["output"]


def grounded_scorer(input, output, expected=None):
    return grounded.eval(output, expected, input=input["question"], context=input["context"])


def helpful_scorer(input, output, expected=None):
    return helpful.eval(output, expected, input=input["question"])


Eval(
    "Drex judge example",
    data=lambda: [{"input": {**row["input"], "output": row["output"]}} for row in DATA],
    task=task,
    scores=[grounded_scorer, helpful_scorer],
)
