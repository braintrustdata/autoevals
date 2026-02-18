import {
  ChatCompletion,
  ChatCompletionMessageParam,
  ChatCompletionTool,
  ChatCompletionToolChoiceOption,
} from "openai/resources";
import { ReasoningEffort } from "openai/resources/shared";
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
  reasoning_effort?: ReasoningEffort;
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

const isWrapped = (
  client: OpenAI,
  dangerouslyAllowBrowser?: boolean,
): boolean => {
  const Constructor = Object.getPrototypeOf(client).constructor;
  const clean = new Constructor({
    apiKey: "dummy",
    dangerouslyAllowBrowser,
  });
  return (
    String(client.chat.completions.create) !==
    String(clean.chat.completions.create)
  );
};

export function buildOpenAIClient(options: OpenAIAuth): OpenAI {
  const client = resolveOpenAIClient(options);

  // Extract from deprecated options or client instance
  const dangerouslyAllowBrowser =
    options.openAiDangerouslyAllowBrowser ??
    (client as any)._options?.dangerouslyAllowBrowser;

  // avoid re-wrapping if the client is already wrapped (proxied)
  if (
    globalThis.__inherited_braintrust_wrap_openai &&
    !isWrapped(client, dangerouslyAllowBrowser)
  ) {
    return globalThis.__inherited_braintrust_wrap_openai(client);
  }

  return client;
}

declare global {
  /* eslint-disable no-var */
  var __inherited_braintrust_wrap_openai: ((openai: any) => any) | undefined;
  var __client: OpenAI | undefined;
  var __defaultModel: string | undefined;
}

export interface InitOptions {
  /**
   * An OpenAI-compatible client to use for all evaluations.
   * This can be an OpenAI client, or any client that implements the OpenAI API
   * (e.g., configured to use the Braintrust proxy with Anthropic, Gemini, etc.)
   */
  client?: OpenAI;
  /**
   * The default model to use for evaluations when not specified per-call.
   * Defaults to "gpt-4o" if not set.
   *
   * When using non-OpenAI providers via the Braintrust proxy, set this to
   * the appropriate model string (e.g., "claude-3-5-sonnet-20241022").
   */
  defaultModel?: string;
}

/**
 * Initialize autoevals with a custom client and/or default model.
 *
 * @example
 * // Using with OpenAI (default)
 * import { init } from "autoevals";
 * import { OpenAI } from "openai";
 *
 * init({ client: new OpenAI() });
 *
 * @example
 * // Using with Anthropic via Braintrust proxy
 * import { init } from "autoevals";
 * import { OpenAI } from "openai";
 *
 * init({
 *   client: new OpenAI({
 *     apiKey: process.env.BRAINTRUST_API_KEY,
 *     baseURL: "https://api.braintrust.dev/v1/proxy",
 *   }),
 *   defaultModel: "claude-3-5-sonnet-20241022",
 * });
 */
export const init = ({ client, defaultModel }: InitOptions = {}) => {
  globalThis.__client = client;
  globalThis.__defaultModel = defaultModel;
};

/**
 * Get the configured default model, or "gpt-4o" if not set.
 */
export const getDefaultModel = (): string => {
  return globalThis.__defaultModel ?? "gpt-4o";
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
