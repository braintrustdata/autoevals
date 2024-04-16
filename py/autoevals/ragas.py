# These metrics are ported, with some enhancements, from the [RAGAS](https://github.com/explodinggradients/ragas) project.

import asyncio
import json

import chevron

from . import Score
from .list import ListContains
from .llm import OpenAILLMScorer
from .oai import arun_cached_request, run_cached_request
from .string import EmbeddingSimilarity


def check_required(name, **kwargs):
    for key, value in kwargs.items():
        if value is None:
            raise ValueError(f"{name} requires {key} value")


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


class ContextEntityRecall(OpenAILLMScorer):
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
        check_required("ContextEntityRecall", expected=expected, context=context)

        context = "\n".join(context) if isinstance(context, list) else context

        expected_entities_future, context_entities_future = (
            aextract_entities(text=expected, model=self.extraction_model, **self.extra_args),
            aextract_entities(text=context, model=self.extraction_model, **self.extra_args),
        )

        expected_entities = [e for e in (await expected_entities_future)["entities"]]
        context_entities = [e for e in (await context_entities_future)["entities"]]

        score = await self.contains_scorer.eval_async(output=context_entities, expected=expected_entities)

        return Score(
            name=self._name(),
            score=score.score,
            metadata={"context_entities": context_entities, "expected_entities": expected_entities},
        )

    def _run_eval_sync(self, output, expected=None, context=None, **kwargs):
        check_required("ContextEntityRecall", expected=expected, context=context)

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


class ContextRelevancy(OpenAILLMScorer):
    """
    Extracts sentences from the context that are relevant to the question with
    self-consistency checks. The number of relevant sentences and is used as the score.
    """

    def __init__(self, pairwise_scorer=None, model=DEFAULT_RAGAS_MODEL, **kwargs):
        super().__init__(**kwargs)

        self.model = model

    def _postprocess(self, context, response):
        sentences = json.loads(response["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"])

        return Score(
            name=self._name(),
            # Simplify this by just using the string length, rather than the number of sentences.
            score=len("".join([s["sentence"] for s in sentences["sentences"]])) / len(context),
            metadata={
                "relevant_sentences": sentences["sentences"],
            },
        )

    async def _run_eval_async(self, output, expected=None, input=None, context=None, **kwargs):
        check_required("ContextRelevancy", input=input, context=context)

        if isinstance(context, list):
            context = "\n".join(context)

        return self._postprocess(
            context,
            await arun_cached_request(
                **extract_sentences_request(question=input, context=context, model=self.model, **self.extra_args)
            ),
        )

    def _run_eval_sync(self, output, expected=None, input=None, context=None, **kwargs):
        check_required("ContextRelevancy", input=input, context=context)

        if isinstance(context, list):
            context = "\n".join(context)

        return self._postprocess(
            context,
            run_cached_request(
                **extract_sentences_request(question=input, context=context, model=self.model, **self.extra_args)
            ),
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


class ContextRecall(OpenAILLMScorer):
    """
    Estimates context recall by estimating TP and FN using annotated answer and
    retrieved context.
    """

    def __init__(self, pairwise_scorer=None, model=DEFAULT_RAGAS_MODEL, **kwargs):
        super().__init__(**kwargs)

        self.model = model

    def _postprocess(self, response):
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

    async def _run_eval_async(self, output, expected=None, input=None, context=None, **kwargs):
        check_required("ContextRecall", input=input, expected=expected, context=context)

        if isinstance(context, list):
            context = "\n".join(context)

        return self._postprocess(
            await arun_cached_request(
                **extract_context_recall_request(
                    question=input, answer=expected, context=context, model=self.model, **self.extra_args
                )
            )
        )

    def _run_eval_sync(self, output, expected=None, input=None, context=None, **kwargs):
        check_required("ContextRecall", input=input, expected=expected, context=context)

        if isinstance(context, list):
            context = "\n".join(context)

        return self._postprocess(
            run_cached_request(
                **extract_context_recall_request(
                    question=input, answer=expected, context=context, model=self.model, **self.extra_args
                )
            )
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


class ContextPrecision(OpenAILLMScorer):
    """
    Average Precision is a metric that evaluates whether all of the
    relevant items selected by the model are ranked higher or not.
    """

    def __init__(self, pairwise_scorer=None, model=DEFAULT_RAGAS_MODEL, **kwargs):
        super().__init__(**kwargs)

        self.model = model

    def _postprocess(self, response):
        precision = json.loads(response["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"])

        return Score(
            name=self._name(),
            score=precision["verdict"],
            metadata={
                "precision": precision,
            },
        )

    async def _run_eval_async(self, output, expected=None, input=None, context=None, **kwargs):
        check_required("ContextPrecision", input=input, expected=expected, context=context)

        if isinstance(context, list):
            context = "\n".join(context)

        return self._postprocess(
            await arun_cached_request(
                **extract_context_precision_request(
                    question=input, answer=expected, context=context, model=self.model, **self.extra_args
                )
            )
        )

    def _run_eval_sync(self, output, expected=None, input=None, context=None, **kwargs):
        check_required("ContextPrecision", input=input, expected=expected, context=context)

        if isinstance(context, list):
            context = "\n".join(context)

        return self._postprocess(
            run_cached_request(
                **extract_context_precision_request(
                    question=input, answer=expected, context=context, model=self.model, **self.extra_args
                )
            )
        )


LONG_FORM_ANSWER_PROMPT = """Create one or more statements from each sentence in the given answer.

The output should be a well-formatted JSON instance that conforms to the JSON schema below.

As an example, for the schema {"properties": {"foo": {"title": "Foo", "description": "a list of strings", "type": "array", "items": {"type": "string"}}}, "required": ["foo"]}
the object {"foo": ["bar", "baz"]} is a well-formatted instance of the schema. The object {"properties": {"foo": ["bar", "baz"]}} is not well-formatted.

Here is the output JSON schema:
```
{"description": "the list of extracted statements", "type": "array", "items": {"type": "string"}}
```

Do not return any preamble or explanations, return only a pure JSON string surrounded by triple backticks (```).

Examples:

question: "Who was  Albert Einstein and what is he best known for?"
answer: "He was a German-born theoretical physicist, widely acknowledged to be one of the greatest and most influential physicists of all time. He was best known for developing the theory of relativity, he also made important contributions to the development of the theory of quantum mechanics."
statements: ```["Albert Einstein, a German-born theoretical physicist, is renowned for being one of the most influential physicists in history.", "Albert Einstein was best known for his theory of relativity.", "Einstein's contributions significantly advanced the field of quantum mechanics", "Recognized globally, Einstein's work has profoundly impacted the scientific community", "Einstein's groundbreaking theories continue to shape our understanding of physics today."]```

question: "Cadmium Chloride is slightly soluble in this chemical, it is also called what?"
answer: "alcohol"
statements: ```["Cadmium Chloride is slightly soluble in alcohol."]```

question: "Were Hitler and Benito Mussolini of the same nationality?"
answer: "Sorry, I can't provide answer to that question."
statements: ```[]```

Your actual task:

question: {{question}}
answer: {{answer}}
statements:
"""

NLI_STATEMENTS_PROMPT = """Your task is to judge the faithfulness of a series of statements based on a given context. For each statement you must return verdict as 1 if the statement can be verified based on the context or 0 if the statement can not be verified based on the context.

The output should be a well-formatted JSON instance that conforms to the JSON schema below.

As an example, for the schema {"properties": {"foo": {"title": "Foo", "description": "a list of strings", "type": "array", "items": {"type": "string"}}}, "required": ["foo"]}
the object {"foo": ["bar", "baz"]} is a well-formatted instance of the schema. The object {"properties": {"foo": ["bar", "baz"]}} is not well-formatted.

Here is the output JSON schema:
```
{"type": "array", "items": {"$ref": "#/definitions/StatementFaithfulnessAnswer"}, "definitions": {"StatementFaithfulnessAnswer": {"title": "StatementFaithfulnessAnswer", "type": "object", "properties": {"statement": {"title": "Statement", "description": "the original statement, word-by-word", "type": "string"}, "verdict": {"title": "Verdict", "description": "the verdict(0/1) of the faithfulness.", "type": "integer"}, "reason": {"title": "Reason", "description": "the reason of the verdict", "type": "string"}}, "required": ["statement", "verdict", "reason"]}}}
```

Do not return any preamble or explanations, return only a pure JSON string surrounded by triple backticks (```).

Examples:

context: "John is a student at XYZ University. He is pursuing a degree in Computer Science. He is enrolled in several courses this semester, including Data Structures, Algorithms, and Database Management. John is a diligent student and spends a significant amount of time studying and completing assignments. He often stays late in the library to work on his projects."
statements: ```["John is majoring in Biology.", "John is taking a course on Artificial Intelligence.", "John is a dedicated student.", "John has a part-time job."]```
answer: ```[{"statement": "John is majoring in Biology.", "verdict": 0, "reason": "John's major is explicitly mentioned as Computer Science. There is no information suggesting he is majoring in Biology."}, {"statement": "John is taking a course on Artificial Intelligence.", "verdict": 0, "reason": "The context mentions the courses John is currently enrolled in, and Artificial Intelligence is not mentioned. Therefore, it cannot be deduced that John is taking a course on AI."}, {"statement": "John is a dedicated student.", "verdict": 1, "reason": "The context states that he spends a significant amount of time studying and completing assignments. Additionally, it mentions that he often stays late in the library to work on his projects, which implies dedication."}, {"statement": "John has a part-time job.", "verdict": 0, "reason": "There is no information given in the context about John having a part-time job."}]```

context: "Photosynthesis is a process used by plants, algae, and certain bacteria to convert light energy into chemical energy."
statements: ```["Albert Einstein was a genius."]```
answer: ```[{"statement": "Albert Einstein was a genius.", "verdict": 0, "reason": "The context and statement are unrelated"}]```

Your actual task:

context: {{context}}
statements: {{statements}}
answer:
"""


EXTRACTED_STATEMENTS_SCHEMA = {
    "properties": {
        "statements": {
            "description": "List of extracted statements",
            "items": {"type": "string"},
            "title": "Statements",
            "type": "array",
        },
    },
    "required": ["statements"],
    "title": "ExtractedStatements",
    "type": "object",
}

STATEMENT_FAITHFULNESS_SCHEMA = {
    "$defs": {
        "StatementFaithfulnessAnswer": {
            "title": "StatementFaithfulnessAnswer",
            "type": "object",
            "properties": {
                "statement": {
                    "title": "Statement",
                    "description": "The original statement, word-for-word",
                    "type": "string",
                },
                "verdict": {
                    "title": "Verdict",
                    "description": "The verdict (0/1) of the faithfulness.",
                    "type": "integer",
                },
                "reason": {"title": "Reason", "description": "The reason for the verdict", "type": "string"},
            },
            "required": ["statement", "verdict", "reason"],
        },
    },
    "properties": {
        "faithfulness": {
            "items": {"$ref": "#/$defs/StatementFaithfulnessAnswer"},
            "type": "array",
        },
    },
    "title": "StatementFaithfulness",
    "type": "object",
}


def extract_statements_request(question, answer, **extra_args):
    return dict(
        messages=[
            {
                "role": "user",
                "content": chevron.render(LONG_FORM_ANSWER_PROMPT, {"question": question, "answer": answer}),
            }
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "extract_statements",
                    "description": "Extract statements from an answer given a question",
                    "parameters": EXTRACTED_STATEMENTS_SCHEMA,
                },
            }
        ],
        tool_choice={"type": "function", "function": {"name": "extract_statements"}},
        **extra_args,
    )


def extract_faithfulness_request(context, statements, **extra_args):
    return dict(
        messages=[
            {
                "role": "user",
                "content": chevron.render(NLI_STATEMENTS_PROMPT, {"context": context, "statements": statements}),
            }
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "judge_statements",
                    "description": "Judge whether the statements are faithful to the context",
                    "parameters": STATEMENT_FAITHFULNESS_SCHEMA,
                },
            }
        ],
        tool_choice={"type": "function", "function": {"name": "judge_statements"}},
        **extra_args,
    )


async def aextract_statements(question, answer, **extra_args):
    response = await arun_cached_request(**extract_statements_request(question=question, answer=answer, **extra_args))
    return load_function_call(response)


def extract_statements(question, answer, **extra_args):
    response = run_cached_request(**extract_statements_request(question=question, answer=answer, **extra_args))
    return load_function_call(response)


async def aextract_faithfulness(context, statements, **extra_args):
    response = await arun_cached_request(
        **extract_faithfulness_request(context=context, statements=statements, **extra_args)
    )
    return load_function_call(response)


def extract_faithfulness(context, statements, **extra_args):
    response = run_cached_request(**extract_faithfulness_request(context=context, statements=statements, **extra_args))
    return load_function_call(response)


class Faithfulness(OpenAILLMScorer):
    """
    Measures factual consistency of a generated answer against the given context.
    """

    def __init__(self, model=DEFAULT_RAGAS_MODEL, **kwargs):
        super().__init__(**kwargs)

        self.model = model

    async def _run_eval_async(self, output, expected=None, input=None, context=None, **kwargs):
        check_required("Faithfulness", input=input, output=output, context=context)

        statements = (await aextract_statements(question=input, answer=expected, model=self.model, **self.extra_args))[
            "statements"
        ]

        faithfulness = (
            await aextract_faithfulness(context=context, statements=statements, model=self.model, **self.extra_args)
        )["faithfulness"]

        return Score(
            name=self._name(),
            score=sum([s["verdict"] for s in faithfulness]) / len(faithfulness),
            metadata={
                "statements": statements,
                "faithfulness": faithfulness,
            },
        )

    def _run_eval_sync(self, output, expected=None, input=None, context=None, **kwargs):
        check_required("Faithfulness", input=input, context=context)

        statements = (extract_statements(question=input, answer=expected, model=self.model, **self.extra_args))[
            "statements"
        ]

        faithfulness = (
            extract_faithfulness(context=context, statements=statements, model=self.model, **self.extra_args)
        )["faithfulness"]

        return Score(
            name=self._name(),
            score=sum([s["verdict"] for s in faithfulness]) / len(faithfulness),
            metadata={
                "statements": statements,
                "faithfulness": faithfulness,
            },
        )


QUESTION_GEN_PROMPT = """Generate a question for the given answer and Identify if answer is noncommittal. Give noncommittal as 1 if the answer is noncommittal and 0 if the answer is committal. A noncommittal answer is one that is evasive, vague, or ambiguous. For example, "I don't know" or "I'm not sure" are noncommittal answers

The output should be a well-formatted JSON instance that conforms to the JSON schema below.

As an example, for the schema {"properties": {"foo": {"title": "Foo", "description": "a list of strings", "type": "array", "items": {"type": "string"}}}, "required": ["foo"]}
the object {"foo": ["bar", "baz"]} is a well-formatted instance of the schema. The object {"properties": {"foo": ["bar", "baz"]}} is not well-formatted.

Here is the output JSON schema:
```
{"type": "object", "properties": {"question": {"title": "Question", "type": "string"}, "noncommittal": {"title": "Noncommittal", "type": "integer"}}, "required": ["question", "noncommittal"]}
```

Do not return any preamble or explanations, return only a pure JSON string surrounded by triple backticks (```).

Examples:

answer: "Albert Einstein was born in Germany."
context: "Albert Einstein was a German-born theoretical physicist who is widely held to be one of the greatest and most influential scientists of all time"
output: ```{"question": "Where was Albert Einstein born?", "noncommittal": 0}```

answer: "It can change its skin color based on the temperature of its environment."
context: "A recent scientific study has discovered a new species of frog in the Amazon rainforest that has the unique ability to change its skin color based on the temperature of its environment."
output: ```{"question": "What unique ability does the newly discovered species of frog have?", "noncommittal": 0}```

answer: "Everest"
context: "The tallest mountain on Earth, measured from sea level, is a renowned peak located in the Himalayas."
output: ```{"question": "What is the tallest mountain on Earth?", "noncommittal": 0}```

answer: "I don't know about the  groundbreaking feature of the smartphone invented in 2023 as am unaware of information beyond 2022. "
context: "In 2023, a groundbreaking invention was announced: a smartphone with a battery life of one month, revolutionizing the way people use mobile technology."
output: ```{"question": "What was the groundbreaking feature of the smartphone invented in 2023?", "noncommittal": 1}```

Your actual task:

answer: {{answer}}
context: {{context}}
output:
"""


QUESTION_GEN_SCHEMA = {
    "properties": {
        "question": {"title": "Question", "type": "string"},
        "noncommittal": {"title": "Noncommittal", "type": "integer"},
    },
    "required": ["question", "noncommittal"],
    "type": "object",
}


def extract_question_gen_request(answer, context, **extra_args):
    return dict(
        messages=[
            {
                "role": "user",
                "content": chevron.render(QUESTION_GEN_PROMPT, {"answer": answer, "context": context}),
            }
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "generate_question",
                    "description": "Generate a question for the given answer and identify if the answer is noncommittal",
                    "parameters": QUESTION_GEN_SCHEMA,
                },
            }
        ],
        tool_choice={"type": "function", "function": {"name": "generate_question"}},
        **extra_args,
    )


class AnswerRelevancy(OpenAILLMScorer):
    """
    Scores the relevancy of the answer according to the given question.
    Answers with incomplete, redundant or unnecessary information are penalized.
    """

    def __init__(self, model=DEFAULT_RAGAS_MODEL, strictness=3, temperature=0.5, **kwargs):
        super().__init__(temperature=temperature, **kwargs)

        self.model = model
        self.strictness = strictness
        self.temperature = temperature

    def _postprocess(self, questions, similarity):
        score = (
            sum([s.score for s in similarity]) / len(questions)
            if not any([q["noncommittal"] for q in questions])
            else 0
        )
        return Score(
            name=self._name(),
            score=score,
            metadata={
                "questions": questions,
                "similarity": similarity,
                "temperature": self.temperature,
            },
        )

    async def _run_eval_async(self, output, expected=None, input=None, context=None, **kwargs):
        check_required("AnswerRelevancy", input=input, output=output, context=context)

        questions = await asyncio.gather(
            *[
                aload_function_call_request(
                    **extract_question_gen_request(answer=output, context=context, model=self.model, **self.extra_args)
                )
                for _ in range(self.strictness)
            ]
        )
        similarity = await asyncio.gather(
            *[
                EmbeddingSimilarity().eval_async(output=q["question"], expected=input, model=self.model)
                for q in questions
            ]
        )

        return self._postprocess(questions, similarity)

    def _run_eval_sync(self, output, expected=None, input=None, context=None, **kwargs):
        check_required("AnswerRelevancy", input=input, output=output, context=context)

        questions = [
            load_function_call_request(
                **extract_question_gen_request(answer=output, context=context, model=self.model, **self.extra_args)
            )
            for _ in range(self.strictness)
        ]
        similarity = [
            EmbeddingSimilarity().eval(output=q["question"], expected=input, model=self.model) for q in questions
        ]

        return self._postprocess(questions, similarity)


class AnswerSimilarity(OpenAILLMScorer):
    """
    Measures the similarity between the generated answer and the expected answer.
    """

    def __init__(self, pairwise_scorer=None, model=DEFAULT_RAGAS_MODEL, **kwargs):
        super().__init__(**kwargs)

        self.model = model

    async def _run_eval_async(self, output, expected=None, input=None, **kwargs):
        check_required("AnswerSimilarity", expected=expected, output=output)

        return await EmbeddingSimilarity().eval_async(
            output=output, expected=expected, model=self.model, **self.extra_args
        )

    def _run_eval_sync(self, output, expected=None, input=None, **kwargs):
        check_required("AnswerSimilarity", expected=expected, output=output)

        return EmbeddingSimilarity().eval(output=output, expected=expected, model=self.model, **self.extra_args)


CORRECTNESS_PROMPT = """Given a ground truth and an answer, analyze each statement in the answer and classify them in one of the following categories:

- TP (true positive): statements that are present in both the answer and the ground truth,
- FP (false positive): statements present in the answer but not found in the ground truth,
- FN (false negative): relevant statements found in the ground truth but omitted in the answer.

A single statement you must classify in exactly one category. Do not try to interpret the meaning of the ground truth or the answer, just compare the presence of the statements in them.

The output should be a well-formatted JSON instance that conforms to the JSON schema below.

As an example, for the schema {"properties": {"foo": {"title": "Foo", "description": "a list of strings", "type": "array", "items": {"type": "string"}}}, "required": ["foo"]}
the object {"foo": ["bar", "baz"]} is a well-formatted instance of the schema. The object {"properties": {"foo": ["bar", "baz"]}} is not well-formatted.

Here is the output JSON schema:
```
{"type": "object", "properties": {"TP": {"title": "Tp", "type": "array", "items": {"type": "string"}}, "FP": {"title": "Fp", "type": "array", "items": {"type": "string"}}, "FN": {"title": "Fn", "type": "array", "items": {"type": "string"}}}, "required": ["TP", "FP", "FN"]}
```

Do not return any preamble or explanations, return only a pure JSON string surrounded by triple backticks (```).

Examples:

question: "What powers the sun and what is its primary function?"
answer: "The sun is powered by nuclear fission, similar to nuclear reactors on Earth, and its primary function is to provide light to the solar system."
ground_truth: "The sun is actually powered by nuclear fusion, not fission. In its core, hydrogen atoms fuse to form helium, releasing a tremendous amount of energy. This energy is what lights up the sun and provides heat and light, essential for life on Earth. The sun's light also plays a critical role in Earth's climate system and helps to drive the weather and ocean currents."
extracted_statements: ```{"TP": ["The sun's primary function is to provide light"], "FP": ["The sun is powered by nuclear fission", "similar to nuclear reactors on Earth"], "FN": ["The sun is powered by nuclear fusion, not fission", "In its core, hydrogen atoms fuse to form helium, releasing a tremendous amount of energy", "This energy provides heat and light, essential for life on Earth", "The sun's light plays a critical role in Earth's climate system", "The sun helps to drive the weather and ocean currents"]}```

question: "What is the boiling point of water?"
answer: "The boiling point of water is 100 degrees Celsius at sea level."
ground_truth: "The boiling point of water is 100 degrees Celsius (212 degrees Fahrenheit) at sea level, but it can change with altitude."
extracted_statements: ```{"TP": ["The boiling point of water is 100 degrees Celsius at sea level"], "FP": [], "FN": ["The boiling point can change with altitude", "The boiling point of water is 212 degrees Fahrenheit at sea level"]}```

Your actual task:

question: {{question}}
answer: {{answer}}
ground_truth: {{ground_truth}}
extracted_statements:
"""


ANSWER_CORRECTNESS_CLASSIFICATION_SCHEMA = {
    "properties": {
        "TP": {"title": "Tp", "type": "array", "items": {"type": "string"}},
        "FP": {"title": "Fp", "type": "array", "items": {"type": "string"}},
        "FN": {"title": "Fn", "type": "array", "items": {"type": "string"}},
    },
    "required": ["TP", "FP", "FN"],
    "type": "object",
}


def extract_correctness_request(question, answer, ground_truth, **extra_args):
    return dict(
        messages=[
            {
                "role": "user",
                "content": chevron.render(
                    CORRECTNESS_PROMPT, {"question": question, "answer": answer, "ground_truth": ground_truth}
                ),
            }
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "classify_statements",
                    "description": "Classify statements as TP, FP, or FN",
                    "parameters": ANSWER_CORRECTNESS_CLASSIFICATION_SCHEMA,
                },
            }
        ],
        tool_choice={"type": "function", "function": {"name": "classify_statements"}},
        **extra_args,
    )


def compute_f1_score(factuality):
    tp, fp, fn = len(factuality["TP"]), len(factuality["FP"]), len(factuality["FN"])
    return tp / (tp + 0.5 * (fp + fn))


class AnswerCorrectness(OpenAILLMScorer):
    """
    Scores the correctness of the answer based on the ground truth.
    """

    def __init__(
        self,
        pairwise_scorer=None,
        model=DEFAULT_RAGAS_MODEL,
        factuality_weight=0.75,
        answer_similarity_weight=0.25,
        answer_similarity=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.model = model
        self.answer_similarity = answer_similarity or AnswerSimilarity()

        if factuality_weight == 0 and answer_similarity_weight == 0:
            raise ValueError("At least one weight must be nonzero")
        if factuality_weight < 0 or answer_similarity_weight < 0:
            raise ValueError("Weights must be non-negative")
        self.factuality_weight = factuality_weight
        self.answer_similarity_weight = answer_similarity_weight

    def _postprocess(self, factuality, similarity):
        factuality_score = compute_f1_score(factuality)
        similarity_score = 0 if similarity is None else similarity.score

        score = (self.factuality_weight * factuality_score + self.answer_similarity_weight * (similarity_score)) / (
            self.factuality_weight + self.answer_similarity_weight
        )

        return Score(
            name=self._name(),
            score=score,
            metadata={
                "factuality": factuality,
                "factuality_score": factuality_score,
                "answer_similarity": similarity,
                "answer_similarity_score": similarity_score,
            },
        )

    async def _run_answer_similarity_async(self, output, expected):
        if self.answer_similarity_weight == 0:
            return None
        return await self.answer_similarity.eval_async(
            output=output, expected=expected, model=self.model, **self.extra_args
        )

    def _run_answer_similarity_sync(self, output, expected):
        if self.answer_similarity_weight == 0:
            return None
        return self.answer_similarity.eval(output=output, expected=expected, model=self.model, **self.extra_args)

    async def _run_eval_async(self, output, expected=None, input=None, **kwargs):
        check_required("AnswerCorrectness", input=input, expected=expected, output=output)

        factuality_future, similarity_future = (
            aload_function_call_request(
                **extract_correctness_request(
                    question=input, answer=output, ground_truth=expected, model=self.model, **self.extra_args
                )
            ),
            self._run_answer_similarity_async(output, expected),
        )

        return self._postprocess(await factuality_future, await similarity_future)

    def _run_eval_sync(self, output, expected=None, input=None, **kwargs):
        check_required("AnswerCorrectness", input=input, expected=expected, output=output)

        factuality, similarity = (
            load_function_call_request(
                **extract_correctness_request(
                    question=input, answer=output, ground_truth=expected, model=self.model, **self.extra_args
                )
            ),
            self._run_answer_similarity_sync(output, expected),
        )

        return self._postprocess(factuality, similarity)


def load_function_call(response):
    return json.loads(response["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"])


async def aload_function_call_request(**kwargs):
    return load_function_call(await arun_cached_request(**kwargs))


def load_function_call_request(**kwargs):
    return load_function_call(run_cached_request(**kwargs))
