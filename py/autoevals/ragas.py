import json

import chevron

from . import Score
from .list import ListContains
from .llm import OpenAIScorer
from .oai import arun_cached_request, run_cached_request
from .string import EmbeddingSimilarity

ENTITY_PROMPT = """Given a text, extract unique entities without repetition. Ensure you consider different forms or mentions of the same entity as a single entity.

The output should be a well-formatted JSON instance that conforms to the JSON schema below.

As an example, for the schema {"properties": {"foo": {"title": "Foo", "description": "a list of strings", "type": "array", "items": {"type": "string"}}}, "required": ["foo"]}
the object {"foo": ["bar", "baz"]} is a well-formatted instance of the schema. The object {"properties": {"foo": ["bar", "baz"]}} is not well-formatted.

Here is the output JSON schema:
```
{"type": "object", "properties": {"entities": {"title": "Entities", "type": "array", "items": {"type": "string"}}}, "required": ["entities"]}
```

Do not return any preamble or explanations, return only a pure JSON string surrounded by triple backticks (```).

Examples:

text: "The Eiffel Tower, located in Paris, France, is one of the most iconic landmarks globally.\n            Millions of visitors are attracted to it each year for its breathtaking views of the city.\n            Completed in 1889, it was constructed in time for the 1889 World's Fair."
output: ```{"entities": ["Eiffel Tower", "Paris", "France", "1889", "World's Fair"]}```

text: "The Colosseum in Rome, also known as the Flavian Amphitheatre, stands as a monument to Roman architectural and engineering achievement.\n            Construction began under Emperor Vespasian in AD 70 and was completed by his son Titus in AD 80.\n            It could hold between 50,000 and 80,000 spectators who watched gladiatorial contests and public spectacles."
output: ```{"entities": ["Colosseum", "Rome", "Flavian Amphitheatre", "Vespasian", "AD 70", "Titus", "AD 80"]}```

text: "The Great Wall of China, stretching over 21,196 kilometers from east to west, is a marvel of ancient defensive architecture.\n            Built to protect against invasions from the north, its construction started as early as the 7th century BC.\n            Today, it is a UNESCO World Heritage Site and a major tourist attraction."
output: ```{"entities": ["Great Wall of China", "21,196 kilometers", "7th century BC", "UNESCO World Heritage Site"]}```

Your actual task:

text: {{text}}
output: """

ENTITY_SCHEMA = {
    "type": "object",
    "properties": {"entities": {"title": "Entities", "type": "array", "items": {"type": "string"}}},
    "required": ["entities"],
}


def extract_entities_request(text, **extra_args):
    return dict(
        messages=[{"role": "user", "content": chevron.render(ENTITY_PROMPT, {"text": text})}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "extract_entities",
                    "description": "Extract unique entities from a given text",
                    "parameters": ENTITY_SCHEMA,
                },
            }
        ],
        tool_choice={"type": "function", "function": {"name": "extract_entities"}},
        **extra_args,
    )


async def aextract_entities(text, **extra_args):
    response = await arun_cached_request(**extract_entities_request(text=text, **extra_args))
    return json.loads(response["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"])


def extract_entities(text, **extra_args):
    response = run_cached_request(**extract_entities_request(text=text, **extra_args))
    return json.loads(response["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"])


class ContextEntityRecall(OpenAIScorer):
    def __init__(self, pairwise_scorer=None, model="gpt-3.5-turbo-16k", **kwargs):
        super().__init__(**kwargs)

        self.extraction_model = model
        self.contains_scorer = ListContains(
            pairwise_scorer=pairwise_scorer or EmbeddingSimilarity(), allow_extra_entities=True
        )

    async def _run_eval_async(self, output, expected=None, context=None, **kwargs):
        if expected is None:
            raise ValueError("ContextEntityRecall requires an expected value")
        if context is None:
            raise ValueError("ContextEntityRecall requires a context value")

        context = "\n".join(context) if isinstance(context, list) else context

        expected_entities = [
            e
            for e in (await aextract_entities(text=expected, model=self.extraction_model, **self.extra_args))[
                "entities"
            ]
        ]
        context_entities = [
            e
            for e in (await aextract_entities(text=context, model=self.extraction_model, **self.extra_args))[
                "entities"
            ]
        ]

        score = await self.contains_scorer.eval_async(output=context_entities, expected=expected_entities)

        return Score(
            name=self._name(),
            score=score.score,
            metadata={"context_entities": context_entities, "expected_entities": expected_entities},
        )

    def _run_eval_sync(self, output, expected=None, context=None, **kwargs):
        if expected is None:
            raise ValueError("ContextEntityRecall requires an expected value")
        if context is None:
            raise ValueError("ContextEntityRecall requires a context value")

        context = "\n".join(context) if isinstance(context, list) else context

        expected_entities = [
            e for e in (extract_entities(text=expected, model=self.extraction_model, **self.extra_args))["entities"]
        ]
        context_entities = [
            e for e in (extract_entities(text=context, model=self.extraction_model, **self.extra_args))["entities"]
        ]

        score = self.contains_scorer.eval(output=context_entities, expected=expected_entities)

        return Score(
            name=self._name(),
            score=score.score,
            metadata={"context_entities": context_entities, "expected_entities": expected_entities},
        )


# Tweaked to return an empty array instead of "Insufficient information".
SENTENCE_PROMPT = """Please extract relevant sentences from the provided context that is absolutely required answer the following question. If no relevant sentences are found, or if you believe the question cannot be answered from the given context, return an empty array.  While extracting candidate sentences you're not allowed to make any changes to sentences from given context.

Your actual task:

question: {{question}}
context: {{context}}
candidate sentences: """

SENTENCE_SCHEMA = {
    "$defs": {
        "RelevantSentence": {
            "properties": {
                "sentence": {"description": "The selected sentence", "title": "Sentence", "type": "string"},
                "reasons": {
                    "description": "Reasons why the sentence is relevant. Explain your thinking step by step.",
                    "items": {"type": "string"},
                    "title": "Reasons",
                    "type": "array",
                },
            },
            "required": ["sentence", "reasons"],
            "title": "RelevantSentence",
            "type": "object",
        }
    },
    "properties": {
        "sentences": {
            "description": "List of referenced sentences",
            "items": {"$ref": "#/$defs/RelevantSentence"},
            "title": "Sentences",
            "type": "array",
        }
    },
    "required": ["sentences"],
    "title": "RelevantSentences",
    "type": "object",
}


def extract_sentences_request(question, context, **extra_args):
    return dict(
        messages=[
            {"role": "user", "content": chevron.render(SENTENCE_PROMPT, {"question": question, "context": context})}
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "extract_sentences",
                    "description": "Extract relevant sentences from a given context",
                    "parameters": SENTENCE_SCHEMA,
                },
            }
        ],
        tool_choice={"type": "function", "function": {"name": "extract_sentences"}},
        **extra_args,
    )


class ContextRelevancy(OpenAIScorer):
    def __init__(self, pairwise_scorer=None, model="gpt-3.5-turbo-16k", **kwargs):
        super().__init__(**kwargs)

        self.model = model

    async def _run_eval_async(self, output, expected=None, input=None, context=None, **kwargs):
        if input is None:
            raise ValueError("ContextRelevancy requires an input value")
        if context is None:
            raise ValueError("ContextRelevancy requires a context value")

        if isinstance(context, list):
            context = "\n".join(context)

        response = await arun_cached_request(
            **extract_sentences_request(question=input, context=context, model=self.model, **self.extra_args)
        )
        sentences = json.loads(response["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"])

        return Score(
            name=self._name(),
            score=len("".join([s["sentence"] for s in sentences["sentences"]])) / len(context),
            metadata={
                "relevant_sentences": sentences["sentences"],
            },
        )

    def _run_eval_sync(self, output, expected=None, input=None, context=None, **kwargs):
        if input is None:
            raise ValueError("ContextRelevancy requires an input value")
        if context is None:
            raise ValueError("ContextRelevancy requires a context value")

        if isinstance(context, list):
            context = "\n".join(context)

        response = run_cached_request(
            **extract_sentences_request(question=input, context=context, model=self.model, **self.extra_args)
        )
        sentences = json.loads(response["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"])

        return Score(
            name=self._name(),
            score=len("".join([s["sentence"] for s in sentences["sentences"]])) / len(context),
            metadata={
                "relevant_sentences": sentences["sentences"],
            },
        )
