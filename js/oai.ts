import {
  ChatCompletion,
  ChatCompletionCreateParams,
  ChatCompletionMessageParam,
} from "openai/resources";
import { OpenAI } from "openai";

import { Env } from "./env";
import { currentSpanTraced, SpanLogFn } from "./util";

export interface CachedLLMParams {
  model: string;
  messages: ChatCompletionMessageParam[];
  functions?: ChatCompletionCreateParams.Function[];
  function_call?: ChatCompletionCreateParams["function_call"];
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
  openAiBaseUrl?: string;
  openAiDefaultHeaders?: Record<string, string>;
  openAiDangerouslyAllowBrowser?: boolean;
}

const PROXY_URL = "https://braintrustproxy.com/v1";

export function buildOpenAIClient(options: OpenAIAuth): OpenAI {
  const {
    openAiApiKey,
    openAiOrganizationId,
    openAiBaseUrl,
    openAiDefaultHeaders,
    openAiDangerouslyAllowBrowser,
  } = options;

  return new OpenAI({
    apiKey: openAiApiKey || Env.OPENAI_API_KEY,
    organization: openAiOrganizationId,
    baseURL: openAiBaseUrl || Env.OPENAI_BASE_URL || PROXY_URL,
    defaultHeaders: openAiDefaultHeaders,
    dangerouslyAllowBrowser: openAiDangerouslyAllowBrowser,
  });
}

declare global {
  var __inherited_braintrust_wrap_openai: ((openai: any) => any) | undefined;
}

export async function cachedChatCompletion(
  params: CachedLLMParams,
  options: { cache?: ChatCache } & OpenAIAuth
): Promise<ChatCompletion> {
  let openai = buildOpenAIClient(options);
  if (globalThis.__inherited_braintrust_wrap_openai) {
    openai = globalThis.__inherited_braintrust_wrap_openai(openai);
  }
  return await openai.chat.completions.create(params);
}
