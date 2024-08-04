import {
  ChatCompletion,
  ChatCompletionMessageParam,
  ChatCompletionTool,
  ChatCompletionToolChoiceOption,
} from "openai/resources";
import { OpenAI } from "openai";

import { Env } from "./env";

export interface CachedLLMParams {
  model: string;
  messages: ChatCompletionMessageParam[];
  tools?: ChatCompletionTool[];
  tool_choice?: ChatCompletionToolChoiceOption;
  temperature?: number;
  max_tokens?: number;
  span_info?: {
    span_attributes?: Record<string, string>;
  };
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

export function extractOpenAIArgs<T extends Record<string, unknown>>(
  args: OpenAIAuth & T,
): OpenAIAuth {
  return {
    openAiApiKey: args.openAiApiKey,
    openAiOrganizationId: args.openAiOrganizationId,
    openAiBaseUrl: args.openAiBaseUrl,
    openAiDefaultHeaders: args.openAiDefaultHeaders,
    openAiDangerouslyAllowBrowser: args.openAiDangerouslyAllowBrowser,
  };
}

const PROXY_URL = "https://api.braintrust.dev/v1/proxy";

export function buildOpenAIClient(options: OpenAIAuth): OpenAI {
  const {
    openAiApiKey,
    openAiOrganizationId,
    openAiBaseUrl,
    openAiDefaultHeaders,
    openAiDangerouslyAllowBrowser,
  } = options;

  const client = new OpenAI({
    apiKey: openAiApiKey || Env.OPENAI_API_KEY || Env.BRAINTRUST_API_KEY,
    organization: openAiOrganizationId,
    baseURL: openAiBaseUrl || Env.OPENAI_BASE_URL || PROXY_URL,
    defaultHeaders: openAiDefaultHeaders,
    dangerouslyAllowBrowser: openAiDangerouslyAllowBrowser,
  });

  if (globalThis.__inherited_braintrust_wrap_openai) {
    return globalThis.__inherited_braintrust_wrap_openai(client);
  } else {
    return client;
  }
}

declare global {
  // eslint-disable-next-line
  var __inherited_braintrust_wrap_openai: ((openai: any) => any) | undefined;
}

export async function cachedChatCompletion(
  params: CachedLLMParams,
  options: { cache?: ChatCache } & OpenAIAuth,
): Promise<ChatCompletion> {
  const openai = buildOpenAIClient(options);

  const fullParams = {
    ...params,
    span_info: {
      span_attributes: {
        ...params.span_info?.span_attributes,
        purpose: "scorer",
      },
    },
  };

  return await openai.chat.completions.create(fullParams);
}
