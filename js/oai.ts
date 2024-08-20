import {
  ChatCompletion,
  ChatCompletionMessageParam,
  ChatCompletionTool,
  ChatCompletionToolChoiceOption,
} from "openai/resources";
import { AzureOpenAI, OpenAI } from "openai";

import { Env } from "./env";

export interface CachedLLMParams {
  /**
   Model to use for the completion.
   Note: If using Azure OpenAI, this should be the deployment name..
   */
  model: string;
  messages: ChatCompletionMessageParam[];
  tools?: ChatCompletionTool[];
  tool_choice?: ChatCompletionToolChoiceOption;
  temperature?: number;
  max_tokens?: number;
  span_info?: {
    spanAttributes?: Record<string, string>;
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
  /**
    If present, use [Azure OpenAI Service](https://learn.microsoft.com/en-us/azure/ai-services/openai/)
    instead of OpenAI.
   */
  azureOpenAi?: AzureOpenAiAuth;
}

export interface AzureOpenAiAuth {
  apiKey: string;
  endpoint: string;
  apiVersion: string;
}

export function extractOpenAIArgs<T extends Record<string, unknown>>(
  args: OpenAIAuth & T
): OpenAIAuth {
  return {
    openAiApiKey: args.openAiApiKey,
    openAiOrganizationId: args.openAiOrganizationId,
    openAiBaseUrl: args.openAiBaseUrl,
    openAiDefaultHeaders: args.openAiDefaultHeaders,
    openAiDangerouslyAllowBrowser: args.openAiDangerouslyAllowBrowser,
    azureOpenAi: args.azureOpenAi,
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
    azureOpenAi,
  } = options;

  const client = azureOpenAi
    ? new AzureOpenAI({
        apiKey: azureOpenAi.apiKey,
        endpoint: azureOpenAi.endpoint,
        apiVersion: azureOpenAi.apiVersion,
        defaultHeaders: openAiDefaultHeaders,
        dangerouslyAllowBrowser: openAiDangerouslyAllowBrowser,
      })
    : new OpenAI({
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
  /* eslint-disable no-var */
  var __inherited_braintrust_wrap_openai: ((openai: any) => any) | undefined;
}

export async function cachedChatCompletion(
  params: CachedLLMParams,
  options: { cache?: ChatCache } & OpenAIAuth
): Promise<ChatCompletion> {
  const openai = buildOpenAIClient(options);

  const fullParams = globalThis.__inherited_braintrust_wrap_openai
    ? {
        ...params,
        span_info: {
          spanAttributes: {
            ...params.span_info?.spanAttributes,
            purpose: "scorer",
          },
        },
      }
    : params;

  return await openai.chat.completions.create(fullParams);
}
