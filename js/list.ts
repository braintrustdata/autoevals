import { Scorer } from "@braintrust/core";
import { Levenshtein } from "./string";
import { linearSumAssignment } from "linear-sum-assignment";

export const ListContains: Scorer<
  string[],
  {
    pairwiseScorer?: Scorer<string, {}>;
  }
> = async (args) => {
  const { output, expected } = args;
  if (expected === undefined) {
    throw new Error("ListContains requires an expected value");
  }

  if (output.length == 0 && expected.length == 0) {
    return {
      name: "ListOverlap",
      score: 1,
    };
  } else if (output.length == 0 || expected.length == 0) {
    return {
      name: "ListOverlap",
      score: 0,
    };
  }

  const pairwiseScorer = args.pairwiseScorer || Levenshtein;

  const distances = await Promise.all(
    args.output.map(async (output_item) =>
      Promise.all(
        expected.map(
          async (expected_item) =>
            1 -
            ((
              await pairwiseScorer({
                output: output_item,
                expected: expected_item,
              })
            ).score ?? 0)
        )
      )
    )
  );

  if (distances.length === 1 && distances[0].length === 1) {
    // There appears to be a bug in the linearSumAssignment library when there is only one element
    return {
      name: "ListOverlap",
      score: 1 - distances[0][0],
    };
  }

  const result = linearSumAssignment(distances, { maximaze: false });

  const pairs = Array.from(result.rowAssignments)
    .map((c, r) =>
      c >= 0
        ? {
            output: output[r],
            expected: expected[c],
            score: 1 - distances[r][c],
          }
        : null
    )
    .filter((pair) => pair !== null) as Array<{
    output: string;
    expected: string;
    score: number;
  }>;

  const avgScore =
    pairs.reduce((acc, pair) => acc + pair.score, 0) / output.length;

  return {
    name: "ListOverlap",
    score: avgScore,
    metadata: {
      pairs,
      lowestDistances: pairs.map((pair) => pair.score),
    },
  };
};
