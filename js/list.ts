import { Scorer } from "@braintrust/core";
import { Levenshtein } from "./string";
import { linearSumAssignment } from "linear-sum-assignment";
import { makePartial, ScorerWithPartial } from "./util";

/**
 * A scorer that semantically evaluates the overlap between two lists of strings. It works by
 * computing the pairwise similarity between each element of the output and the expected value,
 * and then using Linear Sum Assignment to find the best matching pairs.
 */
export const ListContains: ScorerWithPartial<
  string[],
  {
    pairwiseScorer?: Scorer<string, {}>;
    allowExtraEntities?: boolean;
  }
> = makePartial(async (args) => {
  const { output, expected, allowExtraEntities } = args;
  if (expected === undefined) {
    throw new Error("ListContains requires an expected value");
  }

  if (output.length == 0 && expected.length == 0) {
    return {
      name: "ListContains",
      score: 1,
    };
  } else if (output.length == 0 || expected.length == 0) {
    return {
      name: "ListContains",
      score: 0,
    };
  }

  const pairwiseScorer = args.pairwiseScorer || Levenshtein;

  const similarities = await Promise.all(
    args.output.map(async (output_item) =>
      Promise.all(
        expected.map(
          async (expected_item) =>
            (
              await pairwiseScorer({
                output: output_item,
                expected: expected_item,
              })
            ).score ?? 0
        )
      )
    )
  );

  if (similarities.length === 1 && similarities[0].length === 1) {
    // There appears to be a bug in the linearSumAssignment library when there is only one element
    return {
      name: "ListContains",
      score: similarities[0][0],
    };
  }

  const result = linearSumAssignment(similarities, { maximaze: true });

  const pairs = Array.from(result.rowAssignments)
    .map((c, r) =>
      c >= 0
        ? {
            output: output[r],
            expected: expected[c],
            score: similarities[r][c],
          }
        : null
    )
    .filter((pair) => pair !== null) as Array<{
    output: string;
    expected: string;
    score: number;
  }>;

  const denominator = allowExtraEntities
    ? expected.length
    : Math.max(output.length, expected.length);

  const avgScore =
    pairs.reduce((acc, pair) => acc + pair.score, 0) / denominator;

  return {
    name: "ListContains",
    score: Math.min(Math.max(avgScore, 0), 1),
    metadata: {
      pairs,
    },
  };
}, "ListContains");
