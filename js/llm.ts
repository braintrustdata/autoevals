import * as path from "path";
import * as fs from "fs";
import * as yaml from "js-yaml";
import { render } from "mustache";

import { Score, Scorer, ScorerArgs } from "./base";
import { ChatCompletionRequestMessage } from "openai";
import { cachedChatCompletion } from "./oai";

// XXX Make sure the templates get distributed
const _SCRIPT_DIR = path.dirname(path.resolve(__filename));
const _MODEL_TEMPLATES = [
  "Battle",
  "ClosedQA",
  "Humor",
  "Factuality",
  "Possible",
  "Security",
  "Summary",
  "Translation",
].reduce((acc, v) => ({ v: true, ...acc }), {});

const NO_COT_SUFFIX = `Answer the question by printing only a single choice from {{__choices}} (without quotes or punctuation) corresponding to the correct answer with no other text.`;

const COT_SUFFIX = `Write out in a step by step manner your reasoning to be sure that your conclusion is correct. Avoid simply stating the correct answer at the outset. Then print only a single choice from {{__choices}} (without quotes or punctuation) on its own line corresponding to the correct answer. At the end, repeat just the answer by itself on a new line formatted as "Answer=X"`;

const SUPPORTED_MODELS = ["gpt-3.5-turbo", "gpt-4"];

interface LLMArgs {
  maxTokens?: number;
  temperature?: number;
}

export type OpenAIClassifierArgs = {
  model: string;
  messages: ChatCompletionRequestMessage[];
  parseScoreFn: (resp: string) => string;
  choiceScores: Record<string, number>;
} & LLMArgs &
  Record<string, unknown>;

export const OpenAIClassifier: Scorer<string, OpenAIClassifierArgs> = async (
  args
) => {
  const {
    output,
    expected,
    messages: messagesArg,
    model,
    parseScoreFn,
    choiceScores,
    maxTokens,
    temperature,
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
    content: m.content && render(m.content, renderArgs),
  }));

  try {
    const resp = await cachedChatCompletion({
      model,
      messages,
      ...extraArgs,
    });

    if (resp.choices.length > 0) {
      return parseResponse(
        resp.choices[0].message.content,
        parseScoreFn,
        choiceScores
      );
    } else {
      throw new Error("Empty response from OpenAI");
    }
  } catch (error) {
    return {
      score: 0,
      error,
    };
  }
};

function parseResponse(
  resp: string,
  parseScoreFn: (resp: string) => string,
  choiceScores: Record<string, number>
): Score {
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

export type LLMClassifierArgs = {
  model?: string;
  useCoT?: boolean;
} & LLMArgs &
  Record<string, unknown>;

export function LLMClassifierFromTemplate({
  promptTemplate,
  choiceScores,
  model = "gpt-3.5-turbo",
  useCoT: useCoTArg,
  temperature,
}: {
  promptTemplate: string;
  choiceScores: Record<string, number>;
  model?: string;
  useCoT?: boolean;
  temperature?: number;
}): Scorer<string, LLMClassifierArgs> {
  const choiceStrings = Object.keys(choiceScores);
  return async (args: ScorerArgs<string, LLMClassifierArgs>) => {
    const useCoT = args.useCoT ?? useCoTArg ?? true;

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
      messages,
      parseScoreFn,
      choiceScores,
      model,
      maxTokens,
      temperature,
      __choices: choiceStrings,
      ...args,
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

export function LLMClassifierFromSpec(
  spec: ModelGradedSpec
): Scorer<string, {}> {
  return LLMClassifierFromTemplate({
    promptTemplate: spec.prompt,
    choiceScores: spec.choice_scores,
    model: spec.model,
    useCoT: spec.use_cot,
    temperature: spec.temperature,
  });
}

export function LLMClassifierFromSpecFile(
  path: string
): Scorer<string, LLMClassifierArgs> {
  const doc = yaml.load(fs.readFileSync(path, "utf-8")) as ModelGradedSpec;
  return LLMClassifierFromSpec(doc);
}

function buildLLMClassifier(name: string) {
  const templateName =
    name.replace(/(?<!^)(?=[A-Z])/g, "_").toLowerCase() + ".yaml";
  const templatePath = path.join(_SCRIPT_DIR, "..", "templates", templateName);

  if (!fs.existsSync(templatePath)) {
    throw new Error(`Model template ${name} not found`);
  }

  return LLMClassifierFromSpecFile(templatePath);
}

export const Battle = buildLLMClassifier("Battle");
export const ClosedQA = buildLLMClassifier("ClosedQA");
export const Humor = buildLLMClassifier("Humor");
export const Factuality = buildLLMClassifier("Factuality");
export const Possible = buildLLMClassifier("Possible");
export const Security = buildLLMClassifier("Security");
export const Summary = buildLLMClassifier("Summary");
export const Translation = buildLLMClassifier("Translation");
