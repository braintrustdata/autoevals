import * as yaml from "js-yaml";
import mustache from "mustache";

import { Score, Scorer, ScorerArgs } from "./base.js";
import { ChatCompletionRequestMessage } from "openai";
import { ChatCache, cachedChatCompletion } from "./oai.js";
import { templates } from "./templates.js";

const NO_COT_SUFFIX = `Answer the question by printing only a single choice from {{__choices}} (without quotes or punctuation) corresponding to the correct answer with no other text.`;

const COT_SUFFIX = `Write out in a step by step manner your reasoning to be sure that your conclusion is correct. Avoid simply stating the correct answer at the outset. Then print only a single choice from {{__choices}} (without quotes or punctuation) on its own line corresponding to the correct answer. At the end, repeat just the answer by itself on a new line formatted as "Answer=X"`;

const SUPPORTED_MODELS = ["gpt-3.5-turbo", "gpt-4"];

interface LLMArgs {
  maxTokens?: number;
  temperature?: number;
  openAiApiKey?: string;
  openAiOrganizationId?: string;
}

export type OpenAIClassifierArgs<RenderArgs> = {
  name: string;
  model: string;
  messages: ChatCompletionRequestMessage[];
  parseScoreFn: (resp: string) => string;
  choiceScores: Record<string, number>;
  cache?: ChatCache;
} & LLMArgs &
  RenderArgs;

export async function OpenAIClassifier<RenderArgs, Output>(
  args: ScorerArgs<Output, OpenAIClassifierArgs<RenderArgs>>
): Promise<Score> {
  const {
    name,
    output,
    expected,
    messages: messagesArg,
    model,
    parseScoreFn,
    choiceScores,
    maxTokens,
    temperature,
    cache,
    openAiApiKey,
    openAiOrganizationId,
    ...remainingRenderArgs
  } = args;

  let found = false;
  for (const m of SUPPORTED_MODELS) {
    if (model.startsWith(m)) {
      found = true;
      break;
    }
  }
  if (!found) {
    throw new Error(
      `Unsupported model: ${model}. Currently only supports OpenAI chat models.`
    );
  }

  const extraArgs = {
    temperature: temperature || 0,
    max_tokens: maxTokens,
  };

  const renderArgs = {
    output,
    expected,
    ...remainingRenderArgs,
  };

  const messages: ChatCompletionRequestMessage[] = messagesArg.map((m) => ({
    ...m,
    content: m.content && mustache.render(m.content, renderArgs),
  }));

  try {
    const resp = await cachedChatCompletion(
      {
        model,
        messages,
        ...extraArgs,
      },
      {
        cache,
        openAiApiKey,
        openAiOrganizationId,
      }
    );

    if (resp.choices.length > 0) {
      return {
        name,
        ...parseResponse(
          resp.choices[0].message!.content!,
          parseScoreFn,
          choiceScores
        ),
      };
    } else {
      throw new Error("Empty response from OpenAI");
    }
  } catch (error) {
    return {
      name,
      score: 0,
      error,
    };
  }
}

function parseResponse(
  resp: string,
  parseScoreFn: (resp: string) => string,
  choiceScores: Record<string, number>
): Omit<Score, "name"> {
  let score = 0;
  let error = undefined;
  const metadata: Record<string, unknown> = {};
  try {
    metadata["rationale"] = `${resp}`;

    const choice = parseScoreFn(resp);
    metadata["choice"] = choice;
    if (choiceScores[choice] !== undefined) {
      score = choiceScores[choice];
    } else {
      throw new Error(`Unknown score choice ${choice}`);
    }
  } catch (e: unknown) {
    score = 0;
    error = e;
  }

  return {
    score,
    metadata,
    error,
  };
}

export type LLMClassifierArgs<RenderArgs> = {
  model?: string;
  useCoT?: boolean;
} & LLMArgs &
  RenderArgs;

export function LLMClassifierFromTemplate<RenderArgs>({
  name,
  promptTemplate,
  choiceScores,
  model = "gpt-3.5-turbo",
  useCoT: useCoTArg,
  temperature,
}: {
  name: string;
  promptTemplate: string;
  choiceScores: Record<string, number>;
  model?: string;
  useCoT?: boolean;
  temperature?: number;
}): Scorer<string, LLMClassifierArgs<RenderArgs>> {
  const choiceStrings = Object.keys(choiceScores);
  return async (
    runtimeArgs: ScorerArgs<string, LLMClassifierArgs<RenderArgs>>
  ) => {
    const useCoT = runtimeArgs.useCoT ?? useCoTArg ?? true;

    const prompt =
      promptTemplate + "\n" + (useCoT ? COT_SUFFIX : NO_COT_SUFFIX);

    let maxTokens = undefined;
    let parseScoreFn = (resp: string) => resp.trim();
    if (useCoT) {
      parseScoreFn = (resp: string) => {
        const answers = [...resp.matchAll(/Answer\s*=\s*(.*)/g)];
        if (answers && answers.length > 0) {
          return answers[answers.length - 1][1].trim();
        } else if (choiceStrings.includes(resp.trim())) {
          return resp.trim();
        } else {
          throw new Error("No answer found in response");
        }
      };
    } else {
      maxTokens = Math.max(...choiceStrings.map((c) => c.length));
    }

    const messages: ChatCompletionRequestMessage[] = [
      {
        role: "user",
        content: prompt,
      },
    ];

    return await OpenAIClassifier({
      name,
      messages,
      parseScoreFn,
      choiceScores,
      model,
      maxTokens,
      temperature,
      __choices: choiceStrings,
      ...runtimeArgs,

      // Since the logic is a bit funky for computing this, include
      // it at the end to prevent overrides
      useCoT,
    });
  };
}

export interface ModelGradedSpec {
  prompt: string;
  choice_scores: Record<string, number>;
  model?: string;
  use_cot?: boolean;
  temperature?: number;
}

export function LLMClassifierFromSpec<RenderArgs>(
  name: string,
  spec: ModelGradedSpec
): Scorer<any, LLMClassifierArgs<RenderArgs>> {
  return LLMClassifierFromTemplate({
    name,
    promptTemplate: spec.prompt,
    choiceScores: spec.choice_scores,
    model: spec.model,
    useCoT: spec.use_cot,
    temperature: spec.temperature,
  });
}

export function LLMClassifierFromSpecFile<RenderArgs>(
  name: string,
  templateName: keyof typeof templates
): Scorer<any, LLMClassifierArgs<RenderArgs>> {
  const doc = yaml.load(templates[templateName]) as ModelGradedSpec;
  return LLMClassifierFromSpec(name, doc);
}

function buildLLMClassifier<RenderArgs>(
  name: string,
  templateName: keyof typeof templates
) {
  if (!(templateName in templates)) {
    throw new Error(`Model template ${name} not found`);
  }

  return LLMClassifierFromSpecFile<RenderArgs>(
    templateName,
    templateName as keyof typeof templates
  );
}

/**
 * Test whether an output _better_ performs the `instructions` than the original
 * (expected) value.
 */
export const Battle = buildLLMClassifier<{ instructions: string }>(
  "Battle",
  "battle"
);

/**
 * Test whether an output answers the `input` using knowledge built into the model.
 * You can specify `criteria` to further constrain the answer.
 */
export const ClosedQA = buildLLMClassifier<{ input: string; criteria: any }>(
  "ClosedQA",
  "closed_q_a"
);

/**
 * Test whether an output is funny.
 */
export const Humor = buildLLMClassifier<{}>("Humor");

/**
 * Test whether an output is factual, compared to an original (`expected`) value.
 */
export const Factuality = buildLLMClassifier<{
  input: string;
  output: string;
  expected?: string;
}>("Factuality", "factuality");

/**
 * Test whether an output is a possible solution to the challenge posed in the input.
 */
export const Possible = buildLLMClassifier<{ input: string }>(
  "Possible",
  "possible"
);

/**
 * Test whether an output is malicious.
 */
export const Security = buildLLMClassifier<{}>("Security", "security");

/**
 * Test whether a SQL query is semantically the same as a reference (output) query.
 */
export const Sql = buildLLMClassifier<{ input: string }>("Sql", "sql");

/**
 * Test whether an output is a better summary of the `input` than the original (`expected`) value.
 */
export const Summary = buildLLMClassifier<{ input: string }>(
  "Summary",
  "summary"
);

/**
 * Test whether an `output` is as good of a translation of the `input` in the specified `language`
 * as an expert (`expected`) value.
 */
export const Translation = buildLLMClassifier<{
  language: string;
  input: string;
}>("Translation", "translation");
