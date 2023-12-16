import { Scorer } from "@braintrust/core";
import levenshtein from "js-levenshtein";
import { OpenAIAuth, buildOpenAIClient } from "./oai";
import { CreateEmbeddingResponse } from "openai/resources/embeddings";
import { SpanLogFn, currentSpanTraced } from "./util";
import { OpenAI } from "openai";
import cossim from "compute-cosine-similarity";

/**
 * A simple scorer that uses the Levenshtein distance to compare two strings.
 */
export const LevenshteinScorer: Scorer<string, {}> = (args) => {
  if (args.expected === undefined) {
    throw new Error("LevenshteinScorer requires an expected value");
  }

  const [output, expected] = [`${args.output}`, `${args.expected}`];
  const maxLen = Math.max(output.length, expected.length);

  let score = 1;
  if (maxLen > 0) {
    score = 1 - levenshtein(output, expected) / maxLen;
  }

  return {
    name: "Levenshtein",
    score,
  };
};

/**
 * A scorer that uses cosine similarity to compare two strings.
 *
 * @param args
 * @param args.prefix A prefix to prepend to the prompt. This is useful for specifying the domain of the inputs.
 * @param args.model The model to use for the embedding distance. Defaults to "text-embedding-ada-002".
 * @param args.expectedMin The minimum expected score. Defaults to 0.7. Values below this will be scored as 0, and
 * values between this and 1 will be scaled linearly.
 * @returns A score between 0 and 1, where 1 is a perfect match.
 */
export const EmbeddingSimilarity: Scorer<
  string,
  {
    prefix?: string;
    expectedMin?: number;
    model?: string;
  } & OpenAIAuth
> = async (args) => {
  if (args.expected === undefined) {
    throw new Error("EmbeddingSimilarity requires an expected value");
  }

  const prefix = args.prefix ?? "";
  const expectedMin = args.expectedMin ?? 0.7;

  const [output, expected] = [
    `${prefix}${args.output}`,
    `${prefix}${args.expected}`,
  ];

  const openai = buildOpenAIClient(args);

  const [outputResult, expectedResult] = await Promise.all(
    [output, expected].map((input) =>
      embed(openai, {
        input,
        model: args.model ?? "text-embedding-ada-002",
      })
    )
  );

  const score = cossim(
    outputResult.data[0].embedding,
    expectedResult.data[0].embedding
  );

  return {
    name: "EmbeddingDistance",
    score: scaleScore(score ?? 0, expectedMin),
    error: score === null ? "EmbeddingDistance failed" : undefined,
  };
};

function scaleScore(score: number, expectedMin: number): number {
  return Math.max((score - expectedMin) / (1 - expectedMin), 0);
}

async function embed(
  openai: OpenAI,
  params: OpenAI.Embeddings.EmbeddingCreateParams
): Promise<CreateEmbeddingResponse> {
  return await currentSpanTraced(
    "OpenAI Embedding",
    async (spanLog: SpanLogFn) => {
      const result = await openai.embeddings.create(params);
      const output = result.data[0].embedding;

      const { input, ...rest } = params;
      spanLog({
        input,
        output,
        metadata: {
          ...rest,
        },
        metrics: {
          tokens: result.usage?.total_tokens,
          prompt_tokens: result.usage?.prompt_tokens,
        },
      });

      return result;
    }
  );
}
