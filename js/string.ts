import { Scorer } from "@braintrust/core";
import levenshtein from "js-levenshtein";
import { OpenAIAuth, buildOpenAIClient } from "./oai";
import cossim from "compute-cosine-similarity";

/**
 * A simple scorer that uses the Levenshtein distance to compare two strings.
 */
export const Levenshtein: Scorer<string, {}> = (args) => {
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
Object.defineProperty(Levenshtein, "name", {
  value: "Levenshtein",
  configurable: true,
});

// For back-compat
export const LevenshteinScorer: Scorer<string, {}> = (args) => {
  return Levenshtein(args);
};

Object.defineProperty(LevenshteinScorer, "name", {
  value: "LevenshteinScorer",
  configurable: true,
});

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
      openai.embeddings.create({
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
    name: "EmbeddingSimilarity",
    score: scaleScore(score ?? 0, expectedMin),
    error: score === null ? "EmbeddingSimilarity failed" : undefined,
  };
};

Object.defineProperty(EmbeddingSimilarity, "name", {
  value: "EmbeddingSimilarity",
  configurable: true,
});

function scaleScore(score: number, expectedMin: number): number {
  return Math.max((score - expectedMin) / (1 - expectedMin), 0);
}
