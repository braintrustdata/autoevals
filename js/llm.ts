import { Score, Scorer, ScorerArgs } from "./score";
import {
  ChatCache,
  OpenAIAuth,
  cachedChatCompletion,
  getDefaultModel,
} from "./oai";
import { ModelGradedSpec, templates } from "./templates";
import {
  ChatCompletionMessage,
  ChatCompletionMessageParam,
  ChatCompletionTool,
} from "openai/resources";
import type { ReasoningEffort } from "openai/resources/shared";
import { makePartial, ScorerWithPartial } from "./partial";
import { renderMessages } from "./render-messages";

const NO_COT_SUFFIX =
  "Answer the question by calling `select_choice` with a single choice from {{__choices}}.";

const COT_SUFFIX =
  "Answer the question by calling `select_choice` with your reasoning in a step-by-step manner to be sure that your conclusion is correct. Avoid simply stating the correct answer at the outset. Select a single choice by setting the `choice` parameter to a single choice from {{__choices}}.";

export type LLMArgs = {
  maxTokens?: number;
  temperature?: number;
  reasoningEffort?: ReasoningEffort;
  reasoningEnabled?: boolean;
  reasoningBudget?: number;
} & OpenAIAuth;

/**
 * The default model to use for LLM-based evaluations.
 * @deprecated Use `init({ defaultModel: "..." })` to configure the default model instead.
 */
export const DEFAULT_MODEL = "gpt-4o";

const PLAIN_RESPONSE_SCHEMA = {
  properties: {
    choice: { description: "The choice", title: "Choice", type: "string" },
  },
  required: ["choice"],
  title: "FunctionResponse",
  type: "object",
};

const COT_RESPONSE_SCHEMA = {
  properties: {
    reasons: {
      description:
        "Write out in a step by step manner your reasoning to be sure that your conclusion is correct. Avoid simply stating the correct answer at the outset.",
      title: "Reasoning",
      type: "string",
    },
    choice: { description: "The choice", title: "Choice", type: "string" },
  },
  required: ["reasons", "choice"],
  title: "CoTResponse",
  type: "object",
};

export function buildClassificationTools(
  useCoT: boolean,
  choiceStrings: string[],
): ChatCompletionTool[] {
  const params = useCoT ? COT_RESPONSE_SCHEMA : PLAIN_RESPONSE_SCHEMA;
  const enumParams = {
    ...params,
    properties: {
      ...params.properties,
      choice: { ...params.properties.choice, enum: choiceStrings },
    },
  };
  return [
    {
      type: "function",
      function: {
        name: "select_choice",
        description: "Call this function to select a choice.",
        parameters: enumParams,
      },
    },
  ];
}

export type OpenAIClassifierArgs<RenderArgs> = {
  name: string;
  model: string;
  messages: ChatCompletionMessageParam[];
  choiceScores: Record<string, number>;
  classificationTools: ChatCompletionTool[];
  cache?: ChatCache;
} & LLMArgs &
  RenderArgs;

export async function OpenAIClassifier<RenderArgs, Output>(
  args: ScorerArgs<Output, OpenAIClassifierArgs<RenderArgs>>,
): Promise<Score> {
  const {
    name,
    output,
    expected,
    openAiApiKey,
    openAiOrganizationId,
    openAiBaseUrl,
    openAiDefaultHeaders,
    openAiDangerouslyAllowBrowser,
    azureOpenAi,
    client,
    ...remaining
  } = args;

  const {
    messages: messagesArg,
    model,
    choiceScores,
    classificationTools: classificationTools,
    maxTokens,
    temperature,
    reasoningEffort,
    reasoningEnabled,
    reasoningBudget,
    cache,
    ...remainingRenderArgs
  } = remaining;

  const extraArgs: {
    temperature?: number;
    max_tokens?: number;
    reasoning_effort?: ReasoningEffort;
    reasoning_enabled?: boolean;
    reasoning_budget?: number;
  } = {};
  if (temperature !== undefined) {
    extraArgs.temperature = temperature;
  }
  if (maxTokens !== undefined) {
    extraArgs.max_tokens = maxTokens;
  }
  if (reasoningEffort !== undefined) {
    extraArgs.reasoning_effort = reasoningEffort;
  }
  if (reasoningEnabled !== undefined) {
    extraArgs.reasoning_enabled = reasoningEnabled;
  }
  if (reasoningBudget !== undefined) {
    extraArgs.reasoning_budget = reasoningBudget;
  }

  const renderArgs = {
    output,
    expected,
    ...remainingRenderArgs,
  };

  const messages = renderMessages(messagesArg, renderArgs);

  const resp = await cachedChatCompletion(
    {
      model,
      messages,
      tools: classificationTools,
      tool_choice: {
        type: "function",
        function: {
          name: "select_choice",
        },
      },
      ...extraArgs,
    },
    client
      ? { client }
      : {
          cache,
          openAiApiKey,
          openAiOrganizationId,
          openAiBaseUrl,
          openAiDefaultHeaders,
          openAiDangerouslyAllowBrowser,
          azureOpenAi,
        },
  );

  if (resp.choices.length > 0) {
    return {
      name,
      ...parseResponse(resp.choices[0].message!, choiceScores),
    };
  } else {
    throw new Error("Empty response from OpenAI");
  }
}

function parseResponse(
  resp: ChatCompletionMessage,
  choiceScores: Record<string, number>,
): Omit<Score, "name"> {
  let score = 0;
  const metadata: Record<string, unknown> = {};

  if (!resp.tool_calls || resp.tool_calls.length === 0) {
    throw new Error("No tool calls in response");
  }
  const toolCall = resp.tool_calls[0];
  if (toolCall.type !== "function") {
    throw new Error("Unexpected tool call type");
  }
  if (toolCall.function.name !== "select_choice") {
    throw new Error("Unexpected tool call");
  }

  const args = JSON.parse(toolCall.function.arguments);
  metadata["rationale"] = args["reasons"];
  const choice = args["choice"]?.trim();
  metadata["choice"] = choice;
  if (choice && choiceScores[choice] !== undefined) {
    score = choiceScores[choice];
  } else {
    throw new Error(`Unknown score choice ${choice}`);
  }
  return {
    score,
    metadata,
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
  model: modelArg,
  useCoT: useCoTArg,
  temperature,
  maxTokens: maxTokensArg,
  reasoningEffort,
  reasoningEnabled,
  reasoningBudget,
}: {
  name: string;
  promptTemplate: string;
  choiceScores: Record<string, number>;
  model?: string;
  useCoT?: boolean;
  temperature?: number;
  maxTokens?: number;
  reasoningEffort?: ReasoningEffort;
  reasoningEnabled?: boolean;
  reasoningBudget?: number;
}): Scorer<string, LLMClassifierArgs<RenderArgs>> {
  const choiceStrings = Object.keys(choiceScores);
  const ret = async (
    runtimeArgs: ScorerArgs<string, LLMClassifierArgs<RenderArgs>>,
  ) => {
    const useCoT = runtimeArgs.useCoT ?? useCoTArg ?? true;
    // Use runtime model > template model > configured default model
    const model = runtimeArgs.model ?? modelArg ?? getDefaultModel();

    const prompt =
      promptTemplate + "\n" + (useCoT ? COT_SUFFIX : NO_COT_SUFFIX);

    const maxTokens = runtimeArgs.maxTokens ?? maxTokensArg;
    const messages: ChatCompletionMessageParam[] = [
      {
        role: "user",
        content: prompt,
      },
    ];

    return await OpenAIClassifier({
      name,
      messages,
      choiceScores,
      classificationTools: buildClassificationTools(useCoT, choiceStrings),
      model,
      maxTokens,
      temperature,
      reasoningEffort,
      reasoningEnabled,
      reasoningBudget,
      __choices: choiceStrings,
      ...runtimeArgs,

      // Since the logic is a bit funky for computing this, include
      // it at the end to prevent overrides
      useCoT,
    });
  };
  Object.defineProperty(ret, "name", {
    value: name,
    configurable: true,
  });

  return ret;
}

export function LLMClassifierFromSpec<RenderArgs>(
  name: string,
  spec: ModelGradedSpec,
): Scorer<any, LLMClassifierArgs<RenderArgs>> {
  return LLMClassifierFromTemplate({
    name,
    promptTemplate: spec.prompt,
    choiceScores: spec.choice_scores,
    model: spec.model,
    useCoT: spec.use_cot,
    temperature: spec.temperature,
    maxTokens: spec.max_tokens,
  });
}

export function LLMClassifierFromSpecFile<RenderArgs>(
  name: string,
  templateName: keyof typeof templates,
): Scorer<any, LLMClassifierArgs<RenderArgs>> {
  const doc = templates[templateName];
  return LLMClassifierFromSpec(name, doc);
}

function buildLLMClassifier<RenderArgs>(
  name: string,
  templateName: keyof typeof templates,
): ScorerWithPartial<string, LLMClassifierArgs<RenderArgs>> {
  if (!(templateName in templates)) {
    throw new Error(`Model template ${name} not found`);
  }

  return makePartial(
    LLMClassifierFromSpecFile<RenderArgs>(
      name,
      templateName as keyof typeof templates,
    ),
    name,
  );
}

/**
 * Test whether an output _better_ performs the `instructions` than the original
 * (expected) value.
 */
export const Battle = buildLLMClassifier<{ instructions: string }>(
  "Battle",
  "battle",
);

/**
 * Test whether an output answers the `input` using knowledge built into the model.
 * You can specify `criteria` to further constrain the answer.
 */
export const ClosedQA = buildLLMClassifier<{ input: string; criteria: any }>(
  "ClosedQA",
  "closed_q_a",
);

/**
 * Test whether an output is funny.
 */
export const Humor = buildLLMClassifier<{}>("Humor", "humor");

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
  "possible",
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
  "summary",
);

/**
 * Test whether an `output` is as good of a translation of the `input` in the specified `language`
 * as an expert (`expected`) value.
 */
export const Translation = buildLLMClassifier<{
  language: string;
  input: string;
}>("Translation", "translation");
