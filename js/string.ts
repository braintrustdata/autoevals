import { Scorer } from "@braintrust/core";
import levenshtein from "js-levenshtein";

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
    name: "levenshtein",
    score,
  };
};
