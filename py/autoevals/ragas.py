# These metrics are ported, with some enhancements, from the [RAGAS](https://github.com/explodinggradients/ragas) project.

import json

import chevron

from . import Score
from .list import ListContains
from .llm import OpenAIScorer
from .oai import arun_cached_request, run_cached_request
from .string import EmbeddingSimilarity

DEFAULT_RAGAS_MODEL = "gpt-3.5-turbo-16k"

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
    """
    Estimates context recall by estimating TP and FN using annotated answer and
    retrieved context.
    """

    def __init__(self, pairwise_scorer=None, model=DEFAULT_RAGAS_MODEL, **kwargs):
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
    """
    Extracts sentences from the context that are relevant to the question with
    self-consistency checks. The number of relevant sentences and is used as the score.
    """

    def __init__(self, pairwise_scorer=None, model=DEFAULT_RAGAS_MODEL, **kwargs):
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
            # Simplify this by just using the string length, rather than the number of sentences.
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
            # Simplify this by just using the string length, rather than the number of sentences.
            score=len("".join([s["sentence"] for s in sentences["sentences"]])) / len(context),
            metadata={
                "relevant_sentences": sentences["sentences"],
            },
        )


CONTEXT_RECALL_PROMPT = """Given a context, and an answer, analyze each sentence in the answer and classify if the sentence can be attributed to the given context or not. Use only "Yes" (1) or "No" (0) as a binary classification. Output json with reason.

The output should be a well-formatted JSON instance that conforms to the JSON schema below.

As an example, for the schema {"properties": {"foo": {"title": "Foo", "description": "a list of strings", "type": "array", "items": {"type": "string"}}}, "required": ["foo"]}
the object {"foo": ["bar", "baz"]} is a well-formatted instance of the schema. The object {"properties": {"foo": ["bar", "baz"]}} is not well-formatted.

Here is the output JSON schema:
```
{"type": "array", "items": {"$ref": "#/definitions/ContextRecallClassificationAnswer"}, "definitions": {"ContextRecallClassificationAnswer": {"title": "ContextRecallClassificationAnswer", "type": "object", "properties": {"statement": {"title": "Statement", "type": "string"}, "attributed": {"title": "Attributed", "type": "integer"}, "reason": {"title": "Reason", "type": "string"}}, "required": ["statement", "attributed", "reason"]}}}
```

Do not return any preamble or explanations, return only a pure JSON string surrounded by triple backticks (```).

Examples:

question: "What can you tell me about albert Albert Einstein?"
context: "Albert Einstein (14 March 1879 - 18 April 1955) was a German-born theoretical physicist, widely held to be one of the greatest and most influential scientists of all time. Best known for developing the theory of relativity, he also made important contributions to quantum mechanics, and was thus a central figure in the revolutionary reshaping of the scientific understanding of nature that modern physics accomplished in the first decades of the twentieth century. His mass-energy equivalence formula E = mc2, which arises from relativity theory, has been called 'the world's most famous equation'. He received the 1921 Nobel Prize in Physics 'for his services to theoretical physics, and especially for his discovery of the law of the photoelectric effect', a pivotal step in the development of quantum theory. His work is also known for its influence on the philosophy of science. In a 1999 poll of 130 leading physicists worldwide by the British journal Physics World, Einstein was ranked the greatest physicist of all time. His intellectual achievements and originality have made Einstein synonymous with genius."
answer: "Albert Einstein born in 14 March 1879 was  German-born theoretical physicist, widely held to be one of the greatest and most influential scientists of all time. He received the 1921 Nobel Prize in Physics for his services to theoretical physics. He published 4 papers in 1905.  Einstein moved to Switzerland in 1895"
classification: ```[{"statement": "Albert Einstein, born on 14 March 1879, was a German-born theoretical physicist, widely held to be one of the greatest and most influential scientists of all time.", "attributed": 1, "reason": "The date of birth of Einstein is mentioned clearly in the context."}, {"statement": "He received the 1921 Nobel Prize in Physics for his services to theoretical physics.", "attributed": 1, "reason": "The exact sentence is present in the given context."}, {"statement": "He published 4 papers in 1905.", "attributed": 0, "reason": "There is no mention about papers he wrote in the given context."}, {"statement": "Einstein moved to Switzerland in 1895.", "attributed": 0, "reason": "There is no supporting evidence for this in the given context."}]```

question: "who won 2020 icc world cup?"
context: "The 2022 ICC Men's T20 World Cup, held from October 16 to November 13, 2022, in Australia, was the eighth edition of the tournament. Originally scheduled for 2020, it was postponed due to the COVID-19 pandemic. England emerged victorious, defeating Pakistan by five wickets in the final to clinch their second ICC Men's T20 World Cup title."
answer: "England"
classification: ```[{"statement": "England won the 2022 ICC Men's T20 World Cup.", "attributed": 1, "reason": "From context it is clear that England defeated Pakistan to win the World Cup."}]```

question: "What is the primary fuel for the Sun?"
context: "NULL"
answer: "Hydrogen"
classification: ```[{"statement": "The Sun's primary fuel is hydrogen.", "attributed": 0, "reason": "The context contains no information"}]```

Your actual task:

question: {{question}}
context: {{context}}
answer: {{answer}}
classification:
"""

CONTEXT_RECALL_SCHEMA = {
    "properties": {
        "statements": {
            "items": {
                "title": "ContextRecallClassificationAnswer",
                "type": "object",
                "properties": {
                    "statement": {"title": "Statement", "type": "string"},
                    "attributed": {"title": "Attributed", "type": "integer"},
                    "reason": {"title": "Reason", "type": "string"},
                },
                "required": ["statement", "attributed", "reason"],
            },
            "type": "array",
        }
    },
    "required": ["statements"],
    "type": "object",
}


def extract_context_recall_request(question, answer, context, **extra_args):
    return dict(
        messages=[
            {
                "role": "user",
                "content": chevron.render(
                    CONTEXT_RECALL_PROMPT, {"question": question, "answer": answer, "context": context}
                ),
            }
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "extract_statements",
                    "parameters": CONTEXT_RECALL_SCHEMA,
                },
            }
        ],
        tool_choice={"type": "function", "function": {"name": "extract_statements"}},
        **extra_args,
    )


class ContextRecall(OpenAIScorer):
    """
    Estimates context recall by estimating TP and FN using annotated answer and
    retrieved context.
    """

    def __init__(self, pairwise_scorer=None, model=DEFAULT_RAGAS_MODEL, **kwargs):
        super().__init__(**kwargs)

        self.model = model

    async def _run_eval_async(self, output, expected=None, input=None, context=None, **kwargs):
        if input is None:
            raise ValueError("ContextRecall requires an input value")
        if expected is None:
            raise ValueError("ContextRecall requires an expected value")
        if context is None:
            raise ValueError("ContextRecall requires a context value")

        if isinstance(context, list):
            context = "\n".join(context)

        response = await arun_cached_request(
            **extract_context_recall_request(
                question=input, answer=expected, context=context, model=self.model, **self.extra_args
            )
        )
        statements = json.loads(response["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"])

        ones = sum([s["attributed"] for s in statements["statements"]])
        total = len(statements["statements"])

        return Score(
            name=self._name(),
            score=ones / total,
            metadata={
                "statements": statements,
            },
        )

    def _run_eval_sync(self, output, expected=None, input=None, context=None, **kwargs):
        if input is None:
            raise ValueError("ContextRecall requires an input value")
        if expected is None:
            raise ValueError("ContextRecall requires an expected value")
        if context is None:
            raise ValueError("ContextRecall requires a context value")

        if isinstance(context, list):
            context = "\n".join(context)

        response = run_cached_request(
            **extract_context_recall_request(
                question=input, answer=expected, context=context, model=self.model, **self.extra_args
            )
        )
        statements = json.loads(response["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"])

        ones = sum([s["attributed"] for s in statements["statements"]])
        total = len(statements["statements"])

        return Score(
            name=self._name(),
            score=ones / total,
            metadata={
                "statements": statements,
            },
        )


CONTEXT_PRECISION_PROMPT = """Given question, answer and context verify if the context was useful in arriving at the given answer. Give verdict as "1" if useful and "0" if not with json output.

The output should be a well-formatted JSON instance that conforms to the JSON schema below.

As an example, for the schema {"properties": {"foo": {"title": "Foo", "description": "a list of strings", "type": "array", "items": {"type": "string"}}}, "required": ["foo"]}
the object {"foo": ["bar", "baz"]} is a well-formatted instance of the schema. The object {"properties": {"foo": ["bar", "baz"]}} is not well-formatted.

Here is the output JSON schema:
```
{"description": "Answer for the verification task whether the context was useful.", "type": "object", "properties": {"reason": {"title": "Reason", "description": "Reason for verification", "type": "string"}, "verdict": {"title": "Verdict", "description": "Binary (0/1) verdict of verification", "type": "integer"}}, "required": ["reason", "verdict"]}
```

Do not return any preamble or explanations, return only a pure JSON string surrounded by triple backticks (```).

Examples:

question: "What can you tell me about albert Albert Einstein?"
context: "Albert Einstein (14 March 1879 – 18 April 1955) was a German-born theoretical physicist, widely held to be one of the greatest and most influential scientists of all time. Best known for developing the theory of relativity, he also made important contributions to quantum mechanics, and was thus a central figure in the revolutionary reshaping of the scientific understanding of nature that modern physics accomplished in the first decades of the twentieth century. His mass–energy equivalence formula E = mc2, which arises from relativity theory, has been called \"the world's most famous equation\". He received the 1921 Nobel Prize in Physics \"for his services to theoretical physics, and especially for his discovery of the law of the photoelectric effect\", a pivotal step in the development of quantum theory. His work is also known for its influence on the philosophy of science. In a 1999 poll of 130 leading physicists worldwide by the British journal Physics World, Einstein was ranked the greatest physicist of all time. His intellectual achievements and originality have made Einstein synonymous with genius."
answer: "Albert Einstein born in 14 March 1879 was German-born theoretical physicist, widely held to be one of the greatest and most influential scientists of all time. He received the 1921 Nobel Prize in Physics for his services to theoretical physics. He published 4 papers in 1905. Einstein moved to Switzerland in 1895"
verification: ```{"reason": "The provided context was indeed useful in arriving at the given answer. The context includes key information about Albert Einstein's life and contributions, which are reflected in the answer.", "verdict": 1}```

question: "who won 2020 icc world cup?"
context: "The 2022 ICC Men's T20 World Cup, held from October 16 to November 13, 2022, in Australia, was the eighth edition of the tournament. Originally scheduled for 2020, it was postponed due to the COVID-19 pandemic. England emerged victorious, defeating Pakistan by five wickets in the final to clinch their second ICC Men's T20 World Cup title."
answer: "England"
verification: ```{"reason": "the context was useful in clarifying the situation regarding the 2020 ICC World Cup and indicating that England was the winner of the tournament that was intended to be held in 2020 but actually took place in 2022.", "verdict": 1}```

question: "What is the tallest mountain in the world?"
context: "The Andes is the longest continental mountain range in the world, located in South America. It stretches across seven countries and features many of the highest peaks in the Western Hemisphere. The range is known for its diverse ecosystems, including the high-altitude Andean Plateau and the Amazon rainforest."
answer: "Mount Everest."
verification: ```{"reason": "the provided context discusses the Andes mountain range, which, while impressive, does not include Mount Everest or directly relate to the question about the world's tallest mountain.", "verdict": 0}```

Your actual task:

question: {{question}}
context: {{context}}
answer: {{answer}}
verification:
"""

CONTEXT_PRECISION_SCHEMA = {
    "title": "ContextPrecisionVerification",
    "description": "Answer for the verification task whether the context was useful.",
    "type": "object",
    "properties": {
        "reason": {"title": "Reason", "description": "Reason for verification", "type": "string"},
        "verdict": {
            "title": "Verdict",
            "description": "Binary (0/1) verdict of verification",
            "type": "integer",
        },
    },
    "required": ["reason", "verdict"],
}


def extract_context_precision_request(question, answer, context, **extra_args):
    return dict(
        messages=[
            {
                "role": "user",
                "content": chevron.render(
                    CONTEXT_PRECISION_PROMPT, {"question": question, "answer": answer, "context": context}
                ),
            }
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "verify",
                    "parameters": CONTEXT_PRECISION_SCHEMA,
                },
            }
        ],
        tool_choice={"type": "function", "function": {"name": "verify"}},
        **extra_args,
    )


class ContextPrecision(OpenAIScorer):
    """
    Average Precision is a metric that evaluates whether all of the
    relevant items selected by the model are ranked higher or not.
    """

    def __init__(self, pairwise_scorer=None, model=DEFAULT_RAGAS_MODEL, **kwargs):
        super().__init__(**kwargs)

        self.model = model

    async def _run_eval_async(self, output, expected=None, input=None, context=None, **kwargs):
        if input is None:
            raise ValueError("ContextPrecision requires an input value")
        if expected is None:
            raise ValueError("ContextPrecision requires an expected value")
        if context is None:
            raise ValueError("ContextPrecision requires a context value")

        if isinstance(context, list):
            context = "\n".join(context)

        response = await arun_cached_request(
            **extract_context_precision_request(
                question=input, answer=expected, context=context, model=self.model, **self.extra_args
            )
        )
        precision = json.loads(response["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"])

        return Score(
            name=self._name(),
            score=precision["verdict"],
            metadata={
                "precision": precision,
            },
        )

    def _run_eval_sync(self, output, expected=None, input=None, context=None, **kwargs):
        if input is None:
            raise ValueError("ContextPrecision requires an input value")
        if expected is None:
            raise ValueError("ContextPrecision requires an expected value")
        if context is None:
            raise ValueError("ContextPrecision requires a context value")

        if isinstance(context, list):
            context = "\n".join(context)

        response = run_cached_request(
            **extract_context_precision_request(
                question=input, answer=expected, context=context, model=self.model, **self.extra_args
            )
        )
        precision = json.loads(response["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"])

        return Score(
            name=self._name(),
            score=precision["verdict"],
            metadata={
                "precision": precision,
            },
        )
