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
    baseURL: openAiBaseUrl || PROXY_URL,
    defaultHeaders: openAiDefaultHeaders,
    dangerouslyAllowBrowser: openAiDangerouslyAllowBrowser,
  });
}

export async function cachedChatCompletion(
  params: CachedLLMParams,
  options: { cache?: ChatCache } & OpenAIAuth
): Promise<ChatCompletion> {
  const { cache } = options;

  return await currentSpanTraced(
    "OpenAI Completion",
    async (spanLog: SpanLogFn) => {
      let cached = false;
      let ret = await cache?.get(params);
      if (ret) {
        cached = true;
      } else {
        const openai = buildOpenAIClient(options);
        const completion = await openai.chat.completions.create(params);

        await cache?.set(params, completion);
        ret = completion;
      }

      const { messages, ...rest } = params;
      spanLog({
        input: messages,
        metadata: {
          ...rest,
          cached,
        },
        output: ret.choices[0],
        metrics: {
          tokens: ret.usage?.total_tokens,
          prompt_tokens: ret.usage?.prompt_tokens,
          completion_tokens: ret.usage?.completion_tokens,
        },
      });

      return ret;
    }
  );
}
