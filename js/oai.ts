import {
  ChatCompletionFunctions,
  ChatCompletionRequestMessage,
  Configuration,
  CreateChatCompletionResponse,
  OpenAIApi,
} from "openai";
import { Env } from "./env.js";

export interface CachedLLMParams {
  model: string;
  messages: ChatCompletionRequestMessage[];
  functions?: ChatCompletionFunctions[];
  temperature?: number;
  max_tokens?: number;
}

export interface ChatCache {
  get(params: CachedLLMParams): Promise<CreateChatCompletionResponse | null>;
  set(
    params: CachedLLMParams,
    response: CreateChatCompletionResponse
  ): Promise<void>;
}

export interface OpenAIAuth {
  openAiApiKey?: string;
  openAiOrganizationId?: string;
}

export async function cachedChatCompletion(
  params: CachedLLMParams,
  options: { cache?: ChatCache } & OpenAIAuth
): Promise<CreateChatCompletionResponse> {
  const { cache, openAiApiKey, openAiOrganizationId } = options;

  const cached = await cache?.get(params);
  if (cached) {
    return cached;
  }

  const config = new Configuration({
    apiKey: openAiApiKey || Env.OPENAI_API_KEY,
    organization: openAiOrganizationId,
  });
  const openai = new OpenAIApi(config);

  if (openai === null) {
    throw new Error("OPENAI_API_KEY not set");
  }

  const completion = await openai.createChatCompletion(params);
  const data = completion.data;

  await cache?.set(params, data);

  return data;
}
