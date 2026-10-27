# Autoevals scorer reference

Complete reference for all scorers available in Autoevals, including parameters, score ranges, and usage examples.

## Table of contents

- [LLM-as-a-judge scorers](#llm-as-a-judge-scorers)
- [RAG (Retrieval-Augmented Generation) scorers](#rag-retrieval-augmented-generation-scorers)
- [Heuristic scorers](#heuristic-scorers)
- [JSON scorers](#json-scorers)
- [List scorers](#list-scorers)
- [Voice scorers](#voice-scorers)

---

## LLM-as-a-judge scorers

These scorers use language models to evaluate outputs based on semantic understanding.

### Factuality

Evaluates whether the output is factually consistent with the expected answer.

**Parameters:**

- `input` (string): The input question or prompt
- `output` (string, required): The generated answer to evaluate
- `expected` (string, required): The ground truth answer
- `model` (string, optional): Model to use (default: configured via `init()` or "gpt-5-mini")
- `client` (Client, optional): Custom OpenAI client

**Score Range:** 0-1

- `1.0` = Output is factually accurate
- `0.0` = Output contains factual errors

**Example:**

```typescript
import { Factuality } from "autoevals";

const result = await Factuality({
  input: "What is the capital of France?",
  output: "Paris",
  expected: "The capital of France is Paris",
});
// Score: 1.0 (factually correct)
```

### Battle

Compares two outputs and determines which one is better.

**Parameters:**

- `input` (string): The input question or prompt
- `output` (string, required): First answer to compare
- `expected` (string, required): Second answer to compare
- `model` (string, optional): Model to use
- `client` (Client, optional): Custom OpenAI client

**Score Range:** 0-1

- `1.0` = Output is significantly better than expected
- `0.5` = Both outputs are roughly equal
- `0.0` = Expected is significantly better than output

**Example:**

```python
from autoevals.llm import Battle

evaluator = Battle()
result = evaluator.eval(
    input="Explain photosynthesis",
    output="Plants use sunlight to make food from CO2 and water",
    expected="Photosynthesis is a process"
)
# Score: ~1.0 (first answer is more detailed)
```

### ClosedQA

Evaluates answers to closed-ended questions where there's a clear correct answer.

**Parameters:**

- `input` (string): The question
- `output` (string, required): The generated answer
- `expected` (string, required): The correct answer
- `model` (string, optional): Model to use
- `criteria` (string, optional): Custom evaluation criteria

**Score Range:** 0-1

- `1.0` = Answer is correct
- `0.0` = Answer is incorrect

### Humor

Evaluates whether the output is humorous.

**Parameters:**

- `input` (string): The context or setup
- `output` (string, required): The text to evaluate for humor
- `model` (string, optional): Model to use

**Score Range:** 0-1

- `1.0` = Very humorous
- `0.0` = Not humorous

### Security

Evaluates whether the output contains security vulnerabilities or unsafe content.

**Parameters:**

- `output` (string, required): The content to evaluate
- `model` (string, optional): Model to use

**Score Range:** 0-1

- `1.0` = No security concerns
- `0.0` = Contains security vulnerabilities

### Moderation

Evaluates content for policy violations using OpenAI's moderation API.

**Parameters:**

- `output` (string, required): The content to moderate
- `client` (Client, optional): Custom OpenAI client

**Score Range:** 0-1

- `1.0` = Content is safe
- `0.0` = Content violates policies

**Categories Checked:**

- Sexual content
- Hate speech
- Harassment
- Self-harm
- Violence
- Sexual content involving minors

### Sql

Evaluates SQL query correctness and quality.

**Parameters:**

- `input` (string): The natural language question
- `output` (string, required): The generated SQL query
- `expected` (string, optional): The correct SQL query
- `model` (string, optional): Model to use

**Score Range:** 0-1

### Summary

Evaluates the quality of text summaries.

**Parameters:**

- `input` (string): The original text
- `output` (string, required): The generated summary
- `expected` (string, optional): A reference summary
- `model` (string, optional): Model to use

**Score Range:** 0-1

- `1.0` = Excellent summary (accurate, concise, complete)
- `0.0` = Poor summary

### Translation

Evaluates translation quality.

**Parameters:**

- `input` (string): The source text
- `output` (string, required): The generated translation
- `expected` (string, optional): A reference translation
- `model` (string, optional): Model to use

**Score Range:** 0-1

- `1.0` = Excellent translation
- `0.0` = Poor translation

### VoiceTaskSuccess

Evaluates whether a voice agent correctly completed the caller's request, from the whole call.

**Parameters:**

- `thread_with_system` (array, optional): The call's messages
- `trace` (Trace, optional): The voice call's trace. Used when `thread_with_system` is missing; the conversation comes from `trace.getThread()` / `trace.get_thread()`
- `model` (string, optional): Model to use

The judge sees the messages as JSON, including tool calls and results. With no conversation, the score is `null`.

**Score Range:** 0-1

- `1.0` = Completed correctly
- `0.5` = Completed with minor problems
- `0.0` = Not completed, or wrong or unsupported

---

## RAG (Retrieval-Augmented Generation) scorers

These scorers evaluate RAG systems by assessing both context retrieval and answer generation quality.

All RAG scorers support passing `context` through the `metadata` parameter when used with Braintrust Eval. See the [RAGAS module documentation](js/ragas.ts) for examples.

### ContextRelevancy

Evaluates how relevant the retrieved context is to the input question.

**Parameters:**

- `input` (string, required): The question
- `output` (string, required): The generated answer
- `context` (string[] | string, required): Retrieved context passages
- `model` (string, optional): Model to use (default: "gpt-5-nano")

**Score Range:** 0-1

- `1.0` = All context is highly relevant
- `0.0` = Context is irrelevant

**Example:**

```python
from autoevals.ragas import ContextRelevancy

scorer = ContextRelevancy()
result = scorer.eval(
    input="What is the capital of France?",
    output="Paris",
    context=[
        "Paris is the capital of France.",
        "Berlin is the capital of Germany."
    ]
)
# Score: ~0.5 (only first context item is relevant)
```

### ContextRecall

Measures how well the context supports the expected answer.

**Parameters:**

- `input` (string): The question
- `expected` (string, required): The ground truth answer
- `context` (string[] | string, required): Retrieved context passages
- `model` (string, optional): Model to use

**Score Range:** 0-1

- `1.0` = Context fully supports the expected answer
- `0.0` = Context doesn't support the expected answer

### ContextPrecision

Measures the precision of retrieved context - whether relevant context appears before irrelevant context.

**Parameters:**

- `input` (string, required): The question
- `expected` (string, required): The ground truth answer
- `context` (string[] | string, required): Retrieved context passages (order matters)
- `model` (string, optional): Model to use

**Score Range:** 0-1

- `1.0` = All relevant context appears first
- `0.0` = Relevant context is buried under irrelevant context

### ContextEntityRecall

Measures how well the context contains entities from the expected answer.

**Parameters:**

- `expected` (string, required): The ground truth answer
- `context` (string[] | string, required): Retrieved context passages
- `model` (string, optional): Model to use

**Score Range:** 0-1

- `1.0` = All entities from expected answer are in context
- `0.0` = No entities from expected answer are in context

### Faithfulness

Evaluates whether the generated answer's claims are supported by the context.

**Parameters:**

- `input` (string): The question
- `output` (string, required): The generated answer
- `context` (string[] | string, required): Retrieved context passages
- `model` (string, optional): Model to use

**Score Range:** 0-1

- `1.0` = All claims in the answer are supported by context
- `0.0` = Answer contains unsupported claims (hallucinations)

**Example:**

```typescript
import { Faithfulness } from "autoevals/ragas";

const result = await Faithfulness({
  input: "What is photosynthesis?",
  output:
    "Photosynthesis is how plants make food using sunlight and also they can fly.",
  context: [
    "Photosynthesis is the process by which plants use sunlight to synthesize foods.",
  ],
});
// Score: ~0.5 (first claim supported, "can fly" is not)
```

### AnswerRelevancy

Measures how relevant the answer is to the question.

**Parameters:**

- `input` (string, required): The question
- `output` (string, required): The generated answer
- `context` (string[] | string, optional): Retrieved context passages
- `model` (string, optional): Model to use
- `embedding_model` (string, optional): Model for embeddings (default: "text-embedding-3-small")

**Score Range:** 0-1

- `1.0` = Answer directly addresses the question
- `0.0` = Answer is off-topic

### AnswerSimilarity

Compares semantic similarity between the generated answer and expected answer using embeddings.

**Parameters:**

- `output` (string, required): The generated answer
- `expected` (string, required): The ground truth answer
- `model` (string, optional): Embedding model to use (default: "text-embedding-3-small")

**Score Range:** 0-1

- `1.0` = Answers are semantically identical
- `0.0` = Answers are completely different

### AnswerCorrectness

Combines factual correctness and semantic similarity to evaluate answers.

**Parameters:**

- `input` (string, required): The question
- `output` (string, required): The generated answer
- `expected` (string, required): The ground truth answer
- `model` (string, optional): Model for factuality checking
- `embedding_model` (string, optional): Model for similarity (default: "text-embedding-3-small")
- `factuality_weight` (number, optional): Weight for factuality (default: 0.75)
- `answer_similarity_weight` (number, optional): Weight for similarity (default: 0.25)

**Score Range:** 0-1

**Formula:** `score = (factuality_weight × factuality_score + answer_similarity_weight × similarity_score) / (factuality_weight + answer_similarity_weight)`

---

## Heuristic scorers

Fast, deterministic scorers that don't use LLMs.

### Levenshtein

Calculates Levenshtein (edit) distance between strings, normalized to 0-1.

**Parameters:**

- `output` (string, required): The generated text
- `expected` (string, required): The expected text

**Score Range:** 0-1

- `1.0` = Strings are identical
- `0.0` = Strings are completely different

**Example:**

```python
from autoevals.string import Levenshtein

scorer = Levenshtein()
result = scorer.eval(output="hello", expected="helo")
# Score: ~0.8 (1 character difference)
```

### ExactMatch

Binary scorer that checks for exact string equality.

**Parameters:**

- `output` (any, required): The generated value
- `expected` (any, required): The expected value

**Score Range:** 0 or 1

- `1` = Values are exactly equal
- `0` = Values differ

### NumericDiff

Evaluates numeric differences with configurable thresholds.

**Parameters:**

- `output` (number, required): The generated number
- `expected` (number, required): The expected number
- `max_diff` (number, optional): Maximum acceptable difference (default: 0)
- `relative` (boolean, optional): Use relative difference (default: false)

**Score Range:** 0-1

- `1.0` = Numbers are equal (within threshold)
- `0.0` = Numbers differ significantly

**Formula (absolute):** `score = max(0, 1 - |output - expected| / max_diff)` (when max_diff > 0)

**Formula (relative):** `score = max(0, 1 - |output - expected| / |expected|)`

**Example:**

```typescript
import { NumericDiff } from "autoevals";

// Absolute difference
const result1 = await NumericDiff({
  output: 10.5,
  expected: 10.0,
  maxDiff: 1.0,
});
// Score: 0.5 (difference of 0.5 out of max 1.0)

// Relative difference
const result2 = await NumericDiff({
  output: 100,
  expected: 110,
  relative: true,
});
// Score: ~0.91 (10% difference)
```

### EmbeddingSimilarity

Compares semantic similarity using text embeddings (cosine similarity).

**Parameters:**

- `output` (string, required): The generated text
- `expected` (string, required): The expected text
- `model` (string, optional): Embedding model (default: "text-embedding-3-small")
- `client` (Client, optional): Custom OpenAI client

**Score Range:** -1 to 1 (typically 0-1 for text)

- `1.0` = Semantically identical
- `0.0` = Unrelated
- `-1.0` = Opposite meanings (rare)

---

## JSON scorers

Scorers for evaluating JSON outputs.

### JSONDiff

Recursively compares JSON objects with customizable string and number comparison.

**Parameters:**

- `output` (any, required): The generated JSON
- `expected` (any, required): The expected JSON
- `string_scorer` (Scorer, optional): Scorer for string values (default: Levenshtein)
- `number_scorer` (Scorer, optional): Scorer for numeric values (default: NumericDiff)
- `preserve_strings` (boolean, optional): Don't auto-parse JSON strings (default: false)

**Score Range:** 0-1

- `1.0` = JSON structures are identical
- `0.0` = JSON structures are completely different

**Example:**

```python
from autoevals.json import JSONDiff

scorer = JSONDiff()
result = scorer.eval(
    output={"name": "John", "age": 30},
    expected={"name": "John", "age": 31}
)
# Score: ~0.5 (name matches, age differs slightly)
```

### ValidJSON

Validates JSON syntax and optionally checks against a JSON Schema.

**Parameters:**

- `output` (any, required): The value to validate
- `schema` (object, optional): JSON Schema to validate against

**Score Range:** 0 or 1

- `1` = Valid JSON (and matches schema if provided)
- `0` = Invalid JSON or doesn't match schema

**Example:**

```typescript
import { ValidJSON } from "autoevals/json";

const schema = {
  type: "object",
  properties: {
    name: { type: "string" },
    age: { type: "number" },
  },
  required: ["name", "age"],
};

const result = await ValidJSON({
  output: '{"name": "John", "age": 30}',
  schema,
});
// Score: 1 (valid JSON matching schema)
```

---

## List scorers

Scorers for evaluating lists and arrays.

### ListContains

Checks if all expected items are present in the output list.

**Parameters:**

- `output` (any[], required): The generated list
- `expected` (any[], required): Items that should be present
- `scorer` (Scorer, optional): Scorer for comparing individual items

**Score Range:** 0-1

- `1.0` = All expected items are present
- `0.0` = None of the expected items are present

**Example:**

```python
from autoevals.list import ListContains

scorer = ListContains()
result = scorer.eval(
    output=["apple", "banana", "cherry"],
    expected=["apple", "banana"]
)
# Score: 1.0 (both expected items present)
```

---

## Voice scorers

Deterministic scorers for the timing of a voice call. They take `utterances`, a list of `{ speaker, start_unix_ms, end_unix_ms, interrupted }` objects, where `speaker` is `"user"` or `"agent"`, times are Unix epoch milliseconds, and `interrupted` (optional) means the user cut the agent off. Build the list from your own traces.

Both scorers group the user's speech into turns the same way:

- Consecutive user utterances merge into one turn until the agent starts speaking.
- User speech wholly inside the agent's, like "mm-hmm", isn't a turn.
- The agent talks over a turn when it starts speaking during the turn and overlaps it by more than `minOverlapMs` (default 300).
- The agent's reply to a turn is the first agent utterance that starts after the turn starts and before the next turn, and runs past the turn's end.

Utterances with a missing or invalid time, or another speaker, are skipped and counted in `skipped`.

Timestamps should mark audible speech, on one clock: the user's end is when they stopped speaking (not when the transcript was finalized), and the agent's start is when its first audio played (not when text-to-speech was requested). Times taken on the server leave out network delay, so callers hear slightly longer gaps.

Limits:

- If the agent cuts in while the user pauses mid-sentence and the user carries on, it counts as a fast reply plus a barge-in, not as talking over the user.
- If the agent says nothing and the user speaks again ("hello?"), both utterances merge into one turn.
- Drop intentional agent backchannels when you build the list, or they count as talking over the user.
- The defaults (1500 ms, 300 ms) follow [Hamming's published bands](https://hamming.ai/resources/conversational-flow-measurement-voice-agents), not a standard.

### VoiceLatency

Measures how quickly the agent replies after the user finishes a turn.

**Parameters:**

- `utterances` (Utterance[], required): The call's utterances
- `maxGapMs` / `max_gap_ms` (number, optional): Longest acceptable gap before a reply (default: 1500)
- `minOverlapMs` / `min_overlap_ms` (number, optional): Overlap needed to count as talking over the user (default: 300)

**Score Range:** 0-1, the share of replies that start within `maxGapMs`. `null` when the call has no replies.

Turns the agent talked over aren't counted. A reply that starts slightly before the user stops counts as a gap of 0. Metadata has `replies`, `unanswered_turns`, `max_gap_ms`, `p50_ms`, `p95_ms`, `gaps_ms` and `skipped`; pool `gaps_ms` to get percentiles across a dataset.

**Example:**

```typescript
import { VoiceLatency } from "autoevals";

const result = await VoiceLatency({
  output: null,
  utterances: [
    { speaker: "user", start_unix_ms: 0, end_unix_ms: 2000 },
    { speaker: "agent", start_unix_ms: 2800, end_unix_ms: 5000 },
    { speaker: "user", start_unix_ms: 6000, end_unix_ms: 7000 },
    { speaker: "agent", start_unix_ms: 10000, end_unix_ms: 11000 },
  ],
});
// Score: 0.5 (replies after 0.8 s and 3 s)
```

### VoiceInterruptions

Measures how often the agent talks over the user.

**Parameters:**

- `utterances` (Utterance[], required): The call's utterances
- `minOverlapMs` / `min_overlap_ms` (number, optional): Overlap needed to count as talking over the user (default: 300)

**Score Range:** 0-1, the share of user turns the agent didn't talk over. `null` unless the call has timed user and agent utterances.

Metadata has `user_turns`, `talk_overs`, `min_overlap_ms`, `skipped`, and `barge_ins`: agent utterances marked `interrupted`, where the user cut the agent off. Barge-ins don't affect the score.

**Example:**

```python
from autoevals import VoiceInterruptions

result = VoiceInterruptions().eval(
    output=None,
    utterances=[
        {"speaker": "user", "start_unix_ms": 0, "end_unix_ms": 3000},
        {"speaker": "agent", "start_unix_ms": 2000, "end_unix_ms": 4000},
        {"speaker": "user", "start_unix_ms": 5000, "end_unix_ms": 6000},
        {"speaker": "agent", "start_unix_ms": 6500, "end_unix_ms": 8000, "interrupted": False},
    ],
)
# Score: 0.5 (the agent talked over the first user turn)
```

---

## Custom scorers

You can create custom scorers for domain-specific evaluation needs. See:

- [JSON scorer examples](py/autoevals/json.py) - Combining validators and comparators
- [Creating custom scorers](README.md#creating-custom-scorers) - Basic custom scorer pattern

---

## Score interpretation

General guidelines for interpreting scores:

- **1.0**: Perfect match or complete correctness
- **0.8-0.99**: Very good, minor differences
- **0.6-0.79**: Acceptable, some issues
- **0.4-0.59**: Moderate quality, significant issues
- **0.2-0.39**: Poor quality, major issues
- **0.0-0.19**: Unacceptable or completely wrong

Note: Interpretation varies by scorer type. Binary scorers (ExactMatch, ValidJSON) only return 0 or 1.

---

## Common parameters

Many scorers share these common parameters:

- `model` (string): LLM model to use for evaluation (default: configured via `init()` or "gpt-5-mini")
- `client` (Client): Custom OpenAI-compatible client
- `use_cot` (boolean): Enable chain-of-thought reasoning for LLM scorers (default: true)
- `temperature` (number): LLM temperature setting
- `max_tokens` (number): Maximum tokens for LLM response

## Configuring defaults

Use `init()` to configure default settings for all scorers:

```typescript
import { init } from "autoevals";
import OpenAI from "openai";

init({
  client: new OpenAI({ apiKey: "..." }),
  defaultModel: "gpt-5-mini",
});
```

```python
from autoevals import init
from openai import OpenAI

init(OpenAI(api_key="..."), default_model="gpt-5-mini")
```
