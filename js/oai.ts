import {
  ChatCompletion,
  ChatCompletionMessageParam,
  ChatCompletionTool,
  ChatCompletionToolChoiceOption,
} from "openai/resources";
import { AzureOpenAI, OpenAI } from "openai";

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

export type OpenAIAuth =
  | {
      /** @deprecated Use the `client` option instead */
      openAiApiKey?: string;
      /** @deprecated Use the `client` option instead */
      openAiOrganizationId?: string;
      /** @deprecated Use the `client` option instead */
      openAiBaseUrl?: string;
      /** @deprecated Use the `client` option instead */
      openAiDefaultHeaders?: Record<string, string>;
      /** @deprecated Use the `client` option instead */
      openAiDangerouslyAllowBrowser?: boolean;
      /** @deprecated Use the `client` option instead */
      azureOpenAi?: AzureOpenAiAuth;
      client?: never;
    }
  | {
      client: OpenAI;
      /** @deprecated Use the `client` option instead */
      openAiApiKey?: never;
      /** @deprecated Use the `client` option instead */
      openAiOrganizationId?: never;
      /** @deprecated Use the `client` option instead */
      openAiBaseUrl?: never;
      /** @deprecated Use the `client` option instead */
      openAiDefaultHeaders?: never;
      /** @deprecated Use the `client` option instead */
      openAiDangerouslyAllowBrowser?: never;
      /** @deprecated Use the `client` option instead */
      azureOpenAi?: never;
    };

export interface AzureOpenAiAuth {
  apiKey: string;
  endpoint: string;
  apiVersion: string;
}

export function extractOpenAIArgs<T extends Record<string, unknown>>(
  args: OpenAIAuth & T,
): OpenAIAuth {
  return args.client
    ? { client: args.client }
    : {
        openAiApiKey: args.openAiApiKey,
        openAiOrganizationId: args.openAiOrganizationId,
        openAiBaseUrl: args.openAiBaseUrl,
        openAiDefaultHeaders: args.openAiDefaultHeaders,
        openAiDangerouslyAllowBrowser: args.openAiDangerouslyAllowBrowser,
        azureOpenAi: args.azureOpenAi,
      };
}

const PROXY_URL = "https://api.braintrust.dev/v1/proxy";

const resolveOpenAIClient = (options: OpenAIAuth): OpenAI => {
  const {
    openAiApiKey,
    openAiOrganizationId,
    openAiBaseUrl,
    openAiDefaultHeaders,
    openAiDangerouslyAllowBrowser,
    azureOpenAi,
  } = options;

  if (options.client) {
    return options.client;
  }

  if (globalThis.__client) {
    return globalThis.__client;
  }

  if (azureOpenAi) {
    // if not unset will could raise an exception
    delete process.env.OPENAI_BASE_URL;

    return new AzureOpenAI({
      apiKey: azureOpenAi.apiKey,
      endpoint: azureOpenAi.endpoint,
      apiVersion: azureOpenAi.apiVersion,
      defaultHeaders: openAiDefaultHeaders,
      dangerouslyAllowBrowser: openAiDangerouslyAllowBrowser,
    });
  }

  return new OpenAI({
    apiKey:
      openAiApiKey ||
      process.env.OPENAI_API_KEY ||
      process.env.BRAINTRUST_API_KEY,
    organization: openAiOrganizationId,
    baseURL: openAiBaseUrl || process.env.OPENAI_BASE_URL || PROXY_URL,
    defaultHeaders: openAiDefaultHeaders,
    dangerouslyAllowBrowser: openAiDangerouslyAllowBrowser,
  });
};

const isWrapped = (client: OpenAI): boolean => {
  const Constructor = Object.getPrototypeOf(client).constructor;
  const clean = new Constructor({ apiKey: "dummy" });
  return (
    String(client.chat.completions.create) !==
    String(clean.chat.completions.create)
  );
};

export function buildOpenAIClient(options: OpenAIAuth): OpenAI {
  const client = resolveOpenAIClient(options);

  // avoid re-wrapping if the client is already wrapped (proxied)
  if (globalThis.__inherited_braintrust_wrap_openai && !isWrapped(client)) {
    return globalThis.__inherited_braintrust_wrap_openai(client);
  }

  return client;
}

declare global {
  /* eslint-disable no-var */
  var __inherited_braintrust_wrap_openai: ((openai: any) => any) | undefined;
  var __client: OpenAI | undefined;
}

export const init = ({ client }: { client?: OpenAI } = {}) => {
  globalThis.__client = client;
};

export async function cachedChatCompletion(
  params: CachedLLMParams,
  options: { cache?: ChatCache } & OpenAIAuth,
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
