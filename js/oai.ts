import {
  ChatCompletion,
  ChatCompletionCreateParams,
  ChatCompletionMessage,
} from "openai/resources/index.mjs";
import { Env } from "./env.js";
import { OpenAI } from "openai";

export interface CachedLLMParams {
  model: string;
  messages: ChatCompletionMessage[];
  functions?: ChatCompletionCreateParams.Function[];
  function_call?: ChatCompletionCreateParams.FunctionCallOption;
  temperature?: number;
  max_tokens?: number;
}

export interface ChatCache {
  get(params: CachedLLMParams): Promise<ChatCompletion | null>;
  set(params: CachedLLMParams, response: ChatCompletion): Promise<void>;
}

export interface OpenAIAuth {
  openAiApiKey?: string;
  openAiOrganizationId?: string;
}

export async function cachedChatCompletion(
  params: CachedLLMParams,
  options: { cache?: ChatCache } & OpenAIAuth
): Promise<ChatCompletion> {
  const { cache, openAiApiKey, openAiOrganizationId } = options;

  const cached = await cache?.get(params);
  if (cached) {
    return cached;
  }

  const openai = new OpenAI({
    apiKey: openAiApiKey || Env.OPENAI_API_KEY,
    organization: openAiOrganizationId,
  });

  if (openai === null) {
    throw new Error("OPENAI_API_KEY not set");
  }

  const completion = await openai.chat.completions.create(params);

  await cache?.set(params, completion);

  return completion;
}
