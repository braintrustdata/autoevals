import {
  ChatCompletion,
  ChatCompletionCreateParams,
  ChatCompletionMessage,
} from "openai/resources/index.mjs";
import { OpenAI } from "openai";

import { Env } from "./env.js";
import { currentSpan } from "./util.js";

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

  return await currentSpan().startSpanWithCallback(
    { name: "OpenAI Completion" },
    async (span: any) => {
      let cached = false;
      let ret = await cache?.get(params);
      if (ret) {
        cached = true;
      } else {
        const openai = new OpenAI({
          apiKey: openAiApiKey || Env.OPENAI_API_KEY,
          organization: openAiOrganizationId,
        });

        if (openai === null) {
          throw new Error("OPENAI_API_KEY not set");
        }

        const completion = await openai.chat.completions.create(params);

        await cache?.set(params, completion);
        ret = completion;
      }

      const { messages, ...rest } = params;
      span.log({
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
