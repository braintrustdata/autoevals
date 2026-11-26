import { Scorer, ScorerArgs } from "./score";
import levenshtein from "js-levenshtein";
import { OpenAIAuth, buildOpenAIClient, getDefaultEmbeddingModel } from "./oai";
import cossim from "compute-cosine-similarity";
import { makePartial, ScorerWithPartial } from "./partial";

/**
 * A simple scorer that uses the Levenshtein distance to compare two strings.
 */
export const Levenshtein: ScorerWithPartial<string, {}> = makePartial(
  (args) => {
    if (args.expected === undefined) {
      throw new Error("LevenshteinScorer requires an expected value");
    }

    // Compare by Unicode code point, like the Python scorer, so a character
    // outside the BMP (an emoji, say) counts as one edit rather than two.
    const [output, expected] = [
      Array.from(`${args.output}`),
      Array.from(`${args.expected}`),
    ];
    const maxLen = Math.max(output.length, expected.length);

    let score = 1;
    if (maxLen > 0) {
      score = 1 - codePointLevenshtein(output, expected) / maxLen;
    }

    return {
      name: "Levenshtein",
      score,
    };
  },

  "Levenshtein",
);

// js-levenshtein only accepts strings and compares UTF-16 code units. Map each
// distinct code point to a single code unit so that it compares code points.
function codePointLevenshtein(a: string[], b: string[]): number {
  const codes = new Map<string, string>();
  const encode = (chars: string[]) =>
    chars
      .map((c) => {
        let code = codes.get(c);
        if (code === undefined) {
          code = String.fromCharCode(codes.size);
          codes.set(c, code);
        }
        return code;
      })
      .join("");
  const [ea, eb] = [encode(a), encode(b)];
  // More than 65536 distinct code points cannot fit in code units; that is far
  // beyond real text, so fall back to comparing the original UTF-16 strings.
  return codes.size > 0x10000
    ? levenshtein(a.join(""), b.join(""))
    : levenshtein(ea, eb);
}

// For back-compat
export const LevenshteinScorer: ScorerWithPartial<string, {}> = Levenshtein;

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
export const EmbeddingSimilarity: ScorerWithPartial<
  string,
  {
    prefix?: string;
    expectedMin?: number;
    model?: string;
  } & OpenAIAuth
> = makePartial(async (args) => {
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
        model: args.model ?? getDefaultEmbeddingModel(),
      }),
    ),
  );

  const score = cossim(
    outputResult.data[0].embedding,
    expectedResult.data[0].embedding,
  );

  return {
    name: "EmbeddingSimilarity",
    score: scaleScore(score ?? 0, expectedMin),
    error: score === null ? "EmbeddingSimilarity failed" : undefined,
  };
}, "EmbeddingSimilarity");

function scaleScore(score: number, expectedMin: number): number {
  return Math.min(Math.max((score - expectedMin) / (1 - expectedMin), 0), 1);
}
