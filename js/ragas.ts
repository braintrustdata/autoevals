/*These metrics are ported, with some enhancements, from the [RAGAS](https://github.com/explodinggradients/ragas) project. */
import mustache from "mustache";

import { Scorer, ScorerArgs } from "@braintrust/core";
import { DEFAULT_MODEL, LLMArgs } from "./llm";
import { buildOpenAIClient } from "./oai";
import OpenAI from "openai";
import { ListContains } from "./list";
import { EmbeddingSimilarity } from "./string";
import { z } from "zod";
import zodToJsonSchema from "zod-to-json-schema";
import { makePartial, ScorerWithPartial } from "./partial";

type RagasArgs = {
  input?: string;
  context?: string | string[];
  model?: string;
} & LLMArgs;

const ENTITY_PROMPT = `Given a text, extract unique entities without repetition. Ensure you consider different forms or mentions of the same entity as a single entity.

The output should be a well-formatted JSON instance that conforms to the JSON schema below.

As an example, for the schema {"properties": {"foo": {"title": "Foo", "description": "a list of strings", "type": "array", "items": {"type": "string"}}}, "required": ["foo"]}
the object {"foo": ["bar", "baz"]} is a well-formatted instance of the schema. The object {"properties": {"foo": ["bar", "baz"]}} is not well-formatted.

Here is the output JSON schema:
\`\`\`
{"type": "object", "properties": {"entities": {"title": "Entities", "type": "array", "items": {"type": "string"}}}, "required": ["entities"]}
\`\`\`

Do not return any preamble or explanations, return only a pure JSON string surrounded by triple backticks (\`\`\`).

Examples:

text: "The Eiffel Tower, located in Paris, France, is one of the most iconic landmarks globally.\n            Millions of visitors are attracted to it each year for its breathtaking views of the city.\n            Completed in 1889, it was constructed in time for the 1889 World's Fair."
output: \`\`\`{"entities": ["Eiffel Tower", "Paris", "France", "1889", "World's Fair"]}\`\`\`

text: "The Colosseum in Rome, also known as the Flavian Amphitheatre, stands as a monument to Roman architectural and engineering achievement.\n            Construction began under Emperor Vespasian in AD 70 and was completed by his son Titus in AD 80.\n            It could hold between 50,000 and 80,000 spectators who watched gladiatorial contests and public spectacles."
output: \`\`\`{"entities": ["Colosseum", "Rome", "Flavian Amphitheatre", "Vespasian", "AD 70", "Titus", "AD 80"]}\`\`\`

text: "The Great Wall of China, stretching over 21,196 kilometers from east to west, is a marvel of ancient defensive architecture.\n            Built to protect against invasions from the north, its construction started as early as the 7th century BC.\n            Today, it is a UNESCO World Heritage Site and a major tourist attraction."
output: \`\`\`{"entities": ["Great Wall of China", "21,196 kilometers", "7th century BC", "UNESCO World Heritage Site"]}\`\`\`

Your actual task:

text: {{text}}
output: `;

const entitySchema = z.object({
  entities: z.array(z.string()),
});

/**
 * Estimates context recall by estimating TP and FN using annotated answer and
 * retrieved context.
 */
export const ContextEntityRecall: ScorerWithPartial<
  string,
  RagasArgs & {
    pairwiseScorer?: Scorer<string, {}>;
  }
> = makePartial(async (args) => {
  const { chatArgs, client, ...inputs } = parseArgs(args);

  const { expected, context } = checkRequired(
    { expected: inputs.expected, context: inputs.context },
    "ContextEntityRecall"
  );

  const makeArgs = (
    text: string
  ): OpenAI.Chat.Completions.ChatCompletionCreateParamsNonStreaming => ({
    ...chatArgs,
    messages: [
      {
        role: "user",
        content: mustache.render(ENTITY_PROMPT, { text }),
      },
    ],
    tools: [
      {
        type: "function",
        function: {
          name: "extract_entities",
          description: "Extract unique entities from a given text",
          parameters: zodToJsonSchema(entitySchema),
        },
      },
    ],
    tool_choice: { type: "function", function: { name: "extract_entities" } },
  });

  const responses = await Promise.all([
    client.chat.completions.create(makeArgs(expected)),
    client.chat.completions.create(makeArgs(context)),
  ]);

  const [expectedEntities, contextEntities] = responses.map(mustParseArgs);

  const score = await ListContains({
    pairwiseScorer: args.pairwiseScorer ?? EmbeddingSimilarity,
    allowExtraEntities: true,
    output: entitySchema.parse(contextEntities).entities,
    expected: entitySchema.parse(expectedEntities).entities,
  });

  return {
    name: "ContextEntityRecall",
    score: score.score,
    metadata: {
      contextEntities: contextEntities.entities,
      expectedEntities: expectedEntities.entities,
    },
  };
}, "ContextEntityRecall");

const SENTENCE_PROMPT = `Please extract relevant sentences from the provided context that is absolutely required answer the following question. If no relevant sentences are found, or if you believe the question cannot be answered from the given context, return an empty array.  While extracting candidate sentences you're not allowed to make any changes to sentences from given context.

Your actual task:

question: {{question}}
context: {{context}}
candidate sentences: `;

const relevantSentencesSchema = z.object({
  sentences: z
    .array(
      z.object({
        sentence: z.string().describe("The selected sentence"),
        reasons: z
          .array(z.string())
          .describe(
            "Reasons why the sentence is relevant. Explain your thinking step by step."
          ),
      })
    )
    .describe("List of referenced sentences"),
});

export const ContextRelevancy: ScorerWithPartial<string, RagasArgs> =
  makePartial(async (args) => {
    const { chatArgs, client, ...inputs } = parseArgs(args);

    const { input, context } = checkRequired(
      { input: inputs.input, context: inputs.context },
      "ContextRelevancy"
    );

    const response = await client.chat.completions.create({
      ...chatArgs,
      messages: [
        {
          role: "user",
          content: mustache.render(SENTENCE_PROMPT, {
            question: input,
            context,
          }),
        },
      ],
      tools: [
        {
          type: "function",
          function: {
            name: "extract_sentences",
            description: "Extract relevant sentences from a given context",
            parameters: zodToJsonSchema(relevantSentencesSchema),
          },
        },
      ],
      tool_choice: {
        type: "function",
        function: { name: "extract_sentences" },
      },
    });

    const sentences = relevantSentencesSchema.parse(mustParseArgs(response));
    return {
      name: "ContextRelevancy",
      score:
        sentences.sentences.map((s) => s.sentence).join("").length /
        context.length,
      metadata: {
        relevantSentences: sentences.sentences,
      },
    };
  }, "ContextRelevancy");

const CONTEXT_RECALL_PROMPT = `Given a context, and an answer, analyze each sentence in the answer and classify if the sentence can be attributed to the given context or not. Use only "Yes" (1) or "No" (0) as a binary classification. Output json with reason.

The output should be a well-formatted JSON instance that conforms to the JSON schema below.

As an example, for the schema {"properties": {"foo": {"title": "Foo", "description": "a list of strings", "type": "array", "items": {"type": "string"}}}, "required": ["foo"]}
the object {"foo": ["bar", "baz"]} is a well-formatted instance of the schema. The object {"properties": {"foo": ["bar", "baz"]}} is not well-formatted.

Here is the output JSON schema:
\`\`\`
{"type": "array", "items": {"$ref": "#/definitions/ContextRecallClassificationAnswer"}, "definitions": {"ContextRecallClassificationAnswer": {"title": "ContextRecallClassificationAnswer", "type": "object", "properties": {"statement": {"title": "Statement", "type": "string"}, "attributed": {"title": "Attributed", "type": "integer"}, "reason": {"title": "Reason", "type": "string"}}, "required": ["statement", "attributed", "reason"]}}}
\`\`\`

Do not return any preamble or explanations, return only a pure JSON string surrounded by triple backticks (\`\`\`).

Examples:

question: "What can you tell me about albert Albert Einstein?"
context: "Albert Einstein (14 March 1879 - 18 April 1955) was a German-born theoretical physicist, widely held to be one of the greatest and most influential scientists of all time. Best known for developing the theory of relativity, he also made important contributions to quantum mechanics, and was thus a central figure in the revolutionary reshaping of the scientific understanding of nature that modern physics accomplished in the first decades of the twentieth century. His mass-energy equivalence formula E = mc2, which arises from relativity theory, has been called 'the world's most famous equation'. He received the 1921 Nobel Prize in Physics 'for his services to theoretical physics, and especially for his discovery of the law of the photoelectric effect', a pivotal step in the development of quantum theory. His work is also known for its influence on the philosophy of science. In a 1999 poll of 130 leading physicists worldwide by the British journal Physics World, Einstein was ranked the greatest physicist of all time. His intellectual achievements and originality have made Einstein synonymous with genius."
answer: "Albert Einstein born in 14 March 1879 was  German-born theoretical physicist, widely held to be one of the greatest and most influential scientists of all time. He received the 1921 Nobel Prize in Physics for his services to theoretical physics. He published 4 papers in 1905.  Einstein moved to Switzerland in 1895"
classification: \`\`\`[{"statement": "Albert Einstein, born on 14 March 1879, was a German-born theoretical physicist, widely held to be one of the greatest and most influential scientists of all time.", "attributed": 1, "reason": "The date of birth of Einstein is mentioned clearly in the context."}, {"statement": "He received the 1921 Nobel Prize in Physics for his services to theoretical physics.", "attributed": 1, "reason": "The exact sentence is present in the given context."}, {"statement": "He published 4 papers in 1905.", "attributed": 0, "reason": "There is no mention about papers he wrote in the given context."}, {"statement": "Einstein moved to Switzerland in 1895.", "attributed": 0, "reason": "There is no supporting evidence for this in the given context."}]\`\`\`

question: "who won 2020 icc world cup?"
context: "The 2022 ICC Men's T20 World Cup, held from October 16 to November 13, 2022, in Australia, was the eighth edition of the tournament. Originally scheduled for 2020, it was postponed due to the COVID-19 pandemic. England emerged victorious, defeating Pakistan by five wickets in the final to clinch their second ICC Men's T20 World Cup title."
answer: "England"
classification: \`\`\`[{"statement": "England won the 2022 ICC Men's T20 World Cup.", "attributed": 1, "reason": "From context it is clear that England defeated Pakistan to win the World Cup."}]\`\`\`

question: "What is the primary fuel for the Sun?"
context: "NULL"
answer: "Hydrogen"
classification: \`\`\`[{"statement": "The Sun's primary fuel is hydrogen.", "attributed": 0, "reason": "The context contains no information"}]\`\`\`

Your actual task:

question: {{question}}
context: {{context}}
answer: {{answer}}
classification:
`;
const contextRecallSchema = z.object({
  statements: z.array(
    z.object({
      statement: z.string(),
      attributed: z.number(),
      reason: z.string(),
    })
  ),
});

export const ContextRecall: ScorerWithPartial<string, RagasArgs> = makePartial(
  async (args) => {
    const { chatArgs, client, ...inputs } = parseArgs(args);
    const { input, expected, context } = checkRequired(
      {
        input: inputs.input,
        expected: inputs.expected,
        context: inputs.context,
      },
      "ContextRecall"
    );

    const response = await client.chat.completions.create({
      ...chatArgs,
      messages: [
        {
          role: "user",
          content: mustache.render(CONTEXT_RECALL_PROMPT, {
            question: input,
            answer: expected,
            context,
          }),
        },
      ],
      tools: [
        {
          type: "function",
          function: {
            name: "extract_statements",
            parameters: zodToJsonSchema(contextRecallSchema),
          },
        },
      ],
      tool_choice: {
        type: "function",
        function: { name: "extract_statements" },
      },
    });

    const statements = contextRecallSchema.parse(mustParseArgs(response));

    return {
      name: "ContextRecall",
      score:
        statements.statements.reduce(
          (acc, { attributed }) => acc + attributed,
          0
        ) / statements.statements.length,
      metadata: {
        statements: statements.statements,
      },
    };
  },
  "ContextRecall"
);

const CONTEXT_PRECISION_PROMPT = `Given question, answer and context verify if the context was useful in arriving at the given answer. Give verdict as "1" if useful and "0" if not with json output.

The output should be a well-formatted JSON instance that conforms to the JSON schema below.

As an example, for the schema {"properties": {"foo": {"title": "Foo", "description": "a list of strings", "type": "array", "items": {"type": "string"}}}, "required": ["foo"]}
the object {"foo": ["bar", "baz"]} is a well-formatted instance of the schema. The object {"properties": {"foo": ["bar", "baz"]}} is not well-formatted.

Here is the output JSON schema:
\`\`\`
{"description": "Answer for the verification task whether the context was useful.", "type": "object", "properties": {"reason": {"title": "Reason", "description": "Reason for verification", "type": "string"}, "verdict": {"title": "Verdict", "description": "Binary (0/1) verdict of verification", "type": "integer"}}, "required": ["reason", "verdict"]}
\`\`\`

Do not return any preamble or explanations, return only a pure JSON string surrounded by triple backticks (\`\`\`).

Examples:

question: "What can you tell me about albert Albert Einstein?"
context: "Albert Einstein (14 March 1879 – 18 April 1955) was a German-born theoretical physicist, widely held to be one of the greatest and most influential scientists of all time. Best known for developing the theory of relativity, he also made important contributions to quantum mechanics, and was thus a central figure in the revolutionary reshaping of the scientific understanding of nature that modern physics accomplished in the first decades of the twentieth century. His mass–energy equivalence formula E = mc2, which arises from relativity theory, has been called \"the world's most famous equation\". He received the 1921 Nobel Prize in Physics \"for his services to theoretical physics, and especially for his discovery of the law of the photoelectric effect\", a pivotal step in the development of quantum theory. His work is also known for its influence on the philosophy of science. In a 1999 poll of 130 leading physicists worldwide by the British journal Physics World, Einstein was ranked the greatest physicist of all time. His intellectual achievements and originality have made Einstein synonymous with genius."
answer: "Albert Einstein born in 14 March 1879 was German-born theoretical physicist, widely held to be one of the greatest and most influential scientists of all time. He received the 1921 Nobel Prize in Physics for his services to theoretical physics. He published 4 papers in 1905. Einstein moved to Switzerland in 1895"
verification: \`\`\`{"reason": "The provided context was indeed useful in arriving at the given answer. The context includes key information about Albert Einstein's life and contributions, which are reflected in the answer.", "verdict": 1}\`\`\`

question: "who won 2020 icc world cup?"
context: "The 2022 ICC Men's T20 World Cup, held from October 16 to November 13, 2022, in Australia, was the eighth edition of the tournament. Originally scheduled for 2020, it was postponed due to the COVID-19 pandemic. England emerged victorious, defeating Pakistan by five wickets in the final to clinch their second ICC Men's T20 World Cup title."
answer: "England"
verification: \`\`\`{"reason": "the context was useful in clarifying the situation regarding the 2020 ICC World Cup and indicating that England was the winner of the tournament that was intended to be held in 2020 but actually took place in 2022.", "verdict": 1}\`\`\`

question: "What is the tallest mountain in the world?"
context: "The Andes is the longest continental mountain range in the world, located in South America. It stretches across seven countries and features many of the highest peaks in the Western Hemisphere. The range is known for its diverse ecosystems, including the high-altitude Andean Plateau and the Amazon rainforest."
answer: "Mount Everest."
verification: \`\`\`{"reason": "the provided context discusses the Andes mountain range, which, while impressive, does not include Mount Everest or directly relate to the question about the world's tallest mountain.", "verdict": 0}\`\`\`

Your actual task:

question: {{question}}
context: {{context}}
answer: {{answer}}
verification:
`;

const contextPrecisionSchema = z.object({
  reason: z.string().describe("Reason for verification"),
  verdict: z.number().describe("Binary (0/1) verdict of verification"),
});

export const ContextPrecision: ScorerWithPartial<string, RagasArgs> =
  makePartial(async (args) => {
    const { chatArgs, client, ...inputs } = parseArgs(args);
    const { input, expected, context } = checkRequired(
      {
        input: inputs.input,
        expected: inputs.expected,
        context: inputs.context,
      },
      "ContextPrecision"
    );

    const response = await client.chat.completions.create({
      ...chatArgs,
      messages: [
        {
          role: "user",
          content: mustache.render(CONTEXT_PRECISION_PROMPT, {
            question: input,
            answer: expected,
            context,
          }),
        },
      ],
      tools: [
        {
          type: "function",
          function: {
            name: "verify",
            description:
              "Verify if context was useful in arriving at the answer",
            parameters: zodToJsonSchema(contextPrecisionSchema),
          },
        },
      ],
      tool_choice: { type: "function", function: { name: "verify" } },
    });

    const precision = contextPrecisionSchema.parse(mustParseArgs(response));

    return {
      name: "ContextPrecision",
      score: precision.verdict,
      metadata: {
        precision,
      },
    };
  }, "ContextPrecision");

const LONG_FORM_ANSWER_PROMPT = `Create one or more statements from each sentence in the given answer.

The output should be a well-formatted JSON instance that conforms to the JSON schema below.

As an example, for the schema {"properties": {"foo": {"title": "Foo", "description": "a list of strings", "type": "array", "items": {"type": "string"}}}, "required": ["foo"]}
the object {"foo": ["bar", "baz"]} is a well-formatted instance of the schema. The object {"properties": {"foo": ["bar", "baz"]}} is not well-formatted.

Here is the output JSON schema:
\`\`\`
{"description": "the list of extracted statements", "type": "array", "items": {"type": "string"}}
\`\`\`

Do not return any preamble or explanations, return only a pure JSON string surrounded by triple backticks (\`\`\`).

Examples:

question: "Who was  Albert Einstein and what is he best known for?"
answer: "He was a German-born theoretical physicist, widely acknowledged to be one of the greatest and most influential physicists of all time. He was best known for developing the theory of relativity, he also made important contributions to the development of the theory of quantum mechanics."
statements: \`\`\`["Albert Einstein, a German-born theoretical physicist, is renowned for being one of the most influential physicists in history.", "Albert Einstein was best known for his theory of relativity.", "Einstein's contributions significantly advanced the field of quantum mechanics", "Recognized globally, Einstein's work has profoundly impacted the scientific community", "Einstein's groundbreaking theories continue to shape our understanding of physics today."]\`\`\`

question: "Cadmium Chloride is slightly soluble in this chemical, it is also called what?"
answer: "alcohol"
statements: \`\`\`["Cadmium Chloride is slightly soluble in alcohol."]\`\`\`

question: "Were Hitler and Benito Mussolini of the same nationality?"
answer: "Sorry, I can't provide answer to that question."
statements: \`\`\`[]\`\`\`

Your actual task:

question: {{question}}
answer: {{answer}}
statements:
`;

const NLI_STATEMENTS_PROMPT = `Your task is to judge the faithfulness of a series of statements based on a given context. For each statement you must return verdict as 1 if the statement can be verified based on the context or 0 if the statement can not be verified based on the context.

The output should be a well-formatted JSON instance that conforms to the JSON schema below.

As an example, for the schema {"properties": {"foo": {"title": "Foo", "description": "a list of strings", "type": "array", "items": {"type": "string"}}}, "required": ["foo"]}
the object {"foo": ["bar", "baz"]} is a well-formatted instance of the schema. The object {"properties": {"foo": ["bar", "baz"]}} is not well-formatted.

Here is the output JSON schema:
\`\`\`
{"type": "array", "items": {"$ref": "#/definitions/StatementFaithfulnessAnswer"}, "definitions": {"StatementFaithfulnessAnswer": {"title": "StatementFaithfulnessAnswer", "type": "object", "properties": {"statement": {"title": "Statement", "description": "the original statement, word-by-word", "type": "string"}, "verdict": {"title": "Verdict", "description": "the verdict(0/1) of the faithfulness.", "type": "integer"}, "reason": {"title": "Reason", "description": "the reason of the verdict", "type": "string"}}, "required": ["statement", "verdict", "reason"]}}}
\`\`\`

Do not return any preamble or explanations, return only a pure JSON string surrounded by triple backticks (\`\`\`).

Examples:

context: "John is a student at XYZ University. He is pursuing a degree in Computer Science. He is enrolled in several courses this semester, including Data Structures, Algorithms, and Database Management. John is a diligent student and spends a significant amount of time studying and completing assignments. He often stays late in the library to work on his projects."
statements: \`\`\`["John is majoring in Biology.", "John is taking a course on Artificial Intelligence.", "John is a dedicated student.", "John has a part-time job."]\`\`\`
answer: \`\`\`[{"statement": "John is majoring in Biology.", "verdict": 0, "reason": "John's major is explicitly mentioned as Computer Science. There is no information suggesting he is majoring in Biology."}, {"statement": "John is taking a course on Artificial Intelligence.", "verdict": 0, "reason": "The context mentions the courses John is currently enrolled in, and Artificial Intelligence is not mentioned. Therefore, it cannot be deduced that John is taking a course on AI."}, {"statement": "John is a dedicated student.", "verdict": 1, "reason": "The context states that he spends a significant amount of time studying and completing assignments. Additionally, it mentions that he often stays late in the library to work on his projects, which implies dedication."}, {"statement": "John has a part-time job.", "verdict": 0, "reason": "There is no information given in the context about John having a part-time job."}]\`\`\`

context: "Photosynthesis is a process used by plants, algae, and certain bacteria to convert light energy into chemical energy."
statements: \`\`\`["Albert Einstein was a genius."]\`\`\`
answer: \`\`\`[{"statement": "Albert Einstein was a genius.", "verdict": 0, "reason": "The context and statement are unrelated"}]\`\`\`

Your actual task:

context: {{context}}
statements: {{statements}}
answer:
`;

const extractedStatementsSchema = z.object({
  statements: z.array(z.string()).describe("the list of extracted statements"),
});
const statementFaithfulnessSchema = z.object({
  faithfulness: z.array(
    z.object({
      statement: z.string().describe("the original statement, word-by-word"),
      verdict: z.number().describe("the verdict(0/1) of the faithfulness."),
      reason: z.string().describe("the reason of the verdict"),
    })
  ),
});

/**
 * Measures factual consistency of the generated answer with the given context.
 */
export const Faithfulness: ScorerWithPartial<string, RagasArgs> = makePartial(
  async (args) => {
    const { chatArgs, client, ...inputs } = parseArgs(args);

    const { input, context, output } = checkRequired(
      { input: inputs.input, context: inputs.context, output: inputs.output },
      "Faithfulness"
    );

    const extractedStatementsResponse = await client.chat.completions.create({
      ...chatArgs,
      messages: [
        {
          role: "user",
          content: mustache.render(LONG_FORM_ANSWER_PROMPT, {
            question: input,
            answer: output,
          }),
        },
      ],
      tools: [
        {
          type: "function",
          function: {
            name: "extract_statements",
            description: "Extract statements from an answer given a question",
            parameters: zodToJsonSchema(extractedStatementsSchema),
          },
        },
      ],
      tool_choice: {
        type: "function",
        function: { name: "extract_statements" },
      },
    });

    const statements = extractedStatementsSchema.parse(
      mustParseArgs(extractedStatementsResponse)
    ).statements;

    const faithfulnessResponse = await client.chat.completions.create({
      ...chatArgs,
      messages: [
        {
          role: "user",
          content: mustache.render(NLI_STATEMENTS_PROMPT, {
            context,
            statements,
          }),
        },
      ],
      tools: [
        {
          type: "function",
          function: {
            name: "judge_statements",
            description:
              "Judge whether the statements are faithful to the context",
            parameters: zodToJsonSchema(statementFaithfulnessSchema),
          },
        },
      ],
      tool_choice: { type: "function", function: { name: "judge_statements" } },
    });

    const faithfulness = statementFaithfulnessSchema.parse(
      mustParseArgs(faithfulnessResponse)
    ).faithfulness;
    const score = faithfulness.length
      ? faithfulness.reduce((acc, { verdict }) => acc + verdict, 0) /
        faithfulness.length
      : 0;

    return {
      name: "Faithfulness",
      score,
      metadata: {
        statements,
        faithfulness,
      },
    };
  },
  "Faithfulness"
);

const QUESTION_GEN_PROMPT = `Generate a question for the given answer and Identify if answer is noncommittal. Give noncommittal as 1 if the answer is noncommittal and 0 if the answer is committal. A noncommittal answer is one that is evasive, vague, or ambiguous. For example, "I don't know" or "I'm not sure" are noncommittal answers

The output should be a well-formatted JSON instance that conforms to the JSON schema below.

As an example, for the schema {"properties": {"foo": {"title": "Foo", "description": "a list of strings", "type": "array", "items": {"type": "string"}}}, "required": ["foo"]}
the object {"foo": ["bar", "baz"]} is a well-formatted instance of the schema. The object {"properties": {"foo": ["bar", "baz"]}} is not well-formatted.

Here is the output JSON schema:
\`\`\`
{"type": "object", "properties": {"question": {"title": "Question", "type": "string"}, "noncommittal": {"title": "Noncommittal", "type": "integer"}}, "required": ["question", "noncommittal"]}
\`\`\`

Do not return any preamble or explanations, return only a pure JSON string surrounded by triple backticks (\`\`\`).

Examples:

answer: "Albert Einstein was born in Germany."
context: "Albert Einstein was a German-born theoretical physicist who is widely held to be one of the greatest and most influential scientists of all time"
output: \`\`\`{"question": "Where was Albert Einstein born?", "noncommittal": 0}\`\`\`

answer: "It can change its skin color based on the temperature of its environment."
context: "A recent scientific study has discovered a new species of frog in the Amazon rainforest that has the unique ability to change its skin color based on the temperature of its environment."
output: \`\`\`{"question": "What unique ability does the newly discovered species of frog have?", "noncommittal": 0}\`\`\`

answer: "Everest"
context: "The tallest mountain on Earth, measured from sea level, is a renowned peak located in the Himalayas."
output: \`\`\`{"question": "What is the tallest mountain on Earth?", "noncommittal": 0}\`\`\`

answer: "I don't know about the  groundbreaking feature of the smartphone invented in 2023 as am unaware of information beyond 2022. "
context: "In 2023, a groundbreaking invention was announced: a smartphone with a battery life of one month, revolutionizing the way people use mobile technology."
output: \`\`\`{"question": "What was the groundbreaking feature of the smartphone invented in 2023?", "noncommittal": 1}\`\`\`

Your actual task:

answer: {{answer}}
context: {{context}}
output:
`;

const questionGenSchema = z.object({
  question: z.string(),
  noncommittal: z.number(),
});

/**
 * Scores the relevancy of the generated answer to the given question.
 * Answers with incomplete, redundant or unnecessary information are penalized.
 */
export const AnswerRelevancy: ScorerWithPartial<
  string,
  RagasArgs & {
    strictness?: number;
  }
> = makePartial(async (args) => {
  const { chatArgs, client, ...inputs } = parseArgs(args);

  const { input, context, output } = checkRequired(
    { input: inputs.input, context: inputs.context, output: inputs.output },
    "AnswerRelevancy"
  );

  const strictness = args.strictness ?? 3;

  const responses = await Promise.all(
    Array.from({ length: strictness }, () =>
      client.chat.completions.create({
        ...chatArgs,
        messages: [
          {
            role: "user",
            content: mustache.render(QUESTION_GEN_PROMPT, {
              answer: output,
              context,
            }),
          },
        ],
        tools: [
          {
            type: "function",
            function: {
              name: "generate_question",
              description:
                "Generate a question for the given answer and identify if the answer is noncommittal",
              parameters: zodToJsonSchema(questionGenSchema),
            },
          },
        ],
        tool_choice: {
          type: "function",
          function: { name: "generate_question" },
        },
      })
    )
  );

  const questions = responses.map((r) =>
    questionGenSchema.parse(mustParseArgs(r))
  );

  const similarity = await Promise.all(
    questions.map(async ({ question }) => {
      const { score } = await EmbeddingSimilarity({
        output: question,
        expected: input,
      });
      return { question, score };
    })
  );

  const score = questions.some(({ noncommittal }) => noncommittal)
    ? 0
    : similarity.reduce((acc, { score }) => acc + (score ?? 0), 0) /
      questions.length;

  return {
    name: "AnswerRelevancy",
    score,
    metadata: {
      questions,
      similarity,
    },
  };
}, "AnswerRelevancy");

/**
 * Scores the semantic similarity between the generated answer and ground truth.
 */
export const AnswerSimilarity: ScorerWithPartial<string, RagasArgs> =
  makePartial(async (args) => {
    const { chatArgs, client, ...inputs } = parseArgs(args);

    const { output, expected } = checkRequired(
      { output: inputs.output, expected: inputs.expected },
      "AnswerSimilarity"
    );

    const { score, error } = await EmbeddingSimilarity({
      output,
      expected,
      expectedMin: 0,
      model: args.model,
    });

    return {
      name: "AnswerSimilarity",
      score,
      error,
    };
  }, "AnswerSimilarity");

const CORRECTNESS_PROMPT = `Given a ground truth and an answer, analyze each statement in the answer and classify them in one of the following categories:

- TP (true positive): statements that are present in both the answer and the ground truth,
- FP (false positive): statements present in the answer but not found in the ground truth,
- FN (false negative): relevant statements found in the ground truth but omitted in the answer.

A single statement you must classify in exactly one category. Do not try to interpret the meaning of the ground truth or the answer, just compare the presence of the statements in them.

The output should be a well-formatted JSON instance that conforms to the JSON schema below.

As an example, for the schema {"properties": {"foo": {"title": "Foo", "description": "a list of strings", "type": "array", "items": {"type": "string"}}}, "required": ["foo"]}
the object {"foo": ["bar", "baz"]} is a well-formatted instance of the schema. The object {"properties": {"foo": ["bar", "baz"]}} is not well-formatted.

Here is the output JSON schema:
\`\`\`
{"type": "object", "properties": {"TP": {"title": "Tp", "type": "array", "items": {"type": "string"}}, "FP": {"title": "Fp", "type": "array", "items": {"type": "string"}}, "FN": {"title": "Fn", "type": "array", "items": {"type": "string"}}}, "required": ["TP", "FP", "FN"]}
\`\`\`

Do not return any preamble or explanations, return only a pure JSON string surrounded by triple backticks (\`\`\`).

Examples:

question: "What powers the sun and what is its primary function?"
answer: "The sun is powered by nuclear fission, similar to nuclear reactors on Earth, and its primary function is to provide light to the solar system."
ground_truth: "The sun is actually powered by nuclear fusion, not fission. In its core, hydrogen atoms fuse to form helium, releasing a tremendous amount of energy. This energy is what lights up the sun and provides heat and light, essential for life on Earth. The sun's light also plays a critical role in Earth's climate system and helps to drive the weather and ocean currents."
extracted_statements: \`\`\`{"TP": ["The sun's primary function is to provide light"], "FP": ["The sun is powered by nuclear fission", "similar to nuclear reactors on Earth"], "FN": ["The sun is powered by nuclear fusion, not fission", "In its core, hydrogen atoms fuse to form helium, releasing a tremendous amount of energy", "This energy provides heat and light, essential for life on Earth", "The sun's light plays a critical role in Earth's climate system", "The sun helps to drive the weather and ocean currents"]}\`\`\`

question: "What is the boiling point of water?"
answer: "The boiling point of water is 100 degrees Celsius at sea level."
ground_truth: "The boiling point of water is 100 degrees Celsius (212 degrees Fahrenheit) at sea level, but it can change with altitude."
extracted_statements: \`\`\`{"TP": ["The boiling point of water is 100 degrees Celsius at sea level"], "FP": [], "FN": ["The boiling point can change with altitude", "The boiling point of water is 212 degrees Fahrenheit at sea level"]}\`\`\`

Your actual task:

question: {{question}}
answer: {{answer}}
ground_truth: {{ground_truth}}
extracted_statements:
`;

const answerCorrectnessClassificationSchema = z.object({
  TP: z.array(z.string()),
  FP: z.array(z.string()),
  FN: z.array(z.string()),
});
type AnswerCorrectnessClassification = z.infer<
  typeof answerCorrectnessClassificationSchema
>;

function computeF1Score(classification: AnswerCorrectnessClassification) {
  const tp = classification.TP.length;
  const fp = classification.FP.length;
  const fn = classification.FN.length;
  return tp / (tp + 0.5 * (fp + fn));
}

/**
 * Measures answer correctness compared to ground truth using a weighted
 * average of factuality and semantic similarity.
 */
export const AnswerCorrectness: ScorerWithPartial<
  string,
  RagasArgs & {
    factualityWeight?: number;
    answerSimilarityWeight?: number;
    answerSimilarity?: Scorer<string, {}>;
  }
> = makePartial(async (args) => {
  const { chatArgs, client, ...inputs } = parseArgs(args);

  const { input, output, expected } = checkRequired(
    { input: inputs.input, output: inputs.output, expected: inputs.expected },
    "AnswerCorrectness"
  );

  const factualityWeight = args.factualityWeight ?? 0.75;
  const answerSimilarityWeight = args.answerSimilarityWeight ?? 0.25;
  const answerSimilarity = args.answerSimilarity ?? AnswerSimilarity;

  if (factualityWeight === 0 && answerSimilarityWeight === 0) {
    throw new Error("At least one weight must be nonzero");
  }
  if (factualityWeight < 0 || answerSimilarityWeight < 0) {
    throw new Error("Weights must be non-negative");
  }

  const [factualityResponse, answerSimilarityResult] = await Promise.all([
    client.chat.completions.create({
      ...chatArgs,
      messages: [
        {
          role: "user",
          content: mustache.render(CORRECTNESS_PROMPT, {
            question: input,
            answer: output,
            ground_truth: expected,
          }),
        },
      ],
      tools: [
        {
          type: "function",
          function: {
            name: "classify_statements",
            description: "Classify statements as TP, FP, or FN",
            parameters: zodToJsonSchema(answerCorrectnessClassificationSchema),
          },
        },
      ],
      tool_choice: {
        type: "function",
        function: { name: "classify_statements" },
      },
    }),
    answerSimilarityWeight === 0
      ? null
      : answerSimilarity({ output, expected }),
  ]);

  const factuality = answerCorrectnessClassificationSchema.parse(
    mustParseArgs(factualityResponse)
  );
  const factualityScore = computeF1Score(factuality);
  const answerSimilarityScore = answerSimilarityResult?.score ?? 0;

  const score =
    (factualityWeight * factualityScore +
      answerSimilarityWeight * answerSimilarityScore) /
    (factualityWeight + answerSimilarityWeight);

  return {
    name: "AnswerCorrectness",
    score,
    error:
      answerSimilarityScore === null ? "AnswerSimilarity failed" : undefined,
    metadata: {
      factuality,
      factualityScore,
      answerSimilarityScore,
    },
  };
}, "AnswerCorrectness");

function parseArgs(args: ScorerArgs<string, RagasArgs>): {
  output: string;
  input?: string;
  expected?: string;
  context?: string;
  chatArgs: Omit<
    OpenAI.Chat.Completions.ChatCompletionCreateParamsNonStreaming,
    "messages"
  >;
  client: OpenAI;
} {
  const {
    input,
    output,
    expected,
    context,
    model,
    temperature,
    maxTokens,
    ...clientArgs
  } = args;
  const chatArgs: Omit<
    OpenAI.Chat.Completions.ChatCompletionCreateParamsNonStreaming,
    "messages"
  > = {
    model: args.model ?? DEFAULT_MODEL,
    temperature: args.temperature ?? 0,
  };
  if (args.maxTokens) {
    chatArgs.max_tokens = args.maxTokens;
  }

  return {
    input,
    output,
    expected,
    context: flatenContext(context),
    chatArgs,
    client: buildOpenAIClient(clientArgs),
  };
}

function flatenContext(context?: string | string[]): string | undefined {
  return context === undefined
    ? context
    : Array.isArray(context)
    ? context.join("\n")
    : context;
}

function checkRequired<T>(
  args: Record<string, T | undefined>,
  name: string
): Record<string, T> {
  for (const [key, value] of Object.entries(args)) {
    if (value === undefined) {
      throw new Error(`${name} requires ${key} value`);
    }
  }

  return args as Record<string, T>;
}

function mustParseArgs(
  resp: OpenAI.Chat.Completions.ChatCompletion
): Record<string, unknown> {
  const args = resp.choices[0]?.message.tool_calls?.[0]?.function.arguments;
  if (!args) {
    throw new Error("No tool call returned");
  }

  return JSON.parse(args);
}
