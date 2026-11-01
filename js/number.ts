import { makePartial, ScorerWithPartial } from "./partial";

/**
 * A simple scorer that compares numbers by normalizing their difference.
 */
export const NumericDiff: ScorerWithPartial<number, {}> = makePartial(
  async (args) => {
    const { output, expected } = args;

    if (expected === undefined) {
      throw new Error("NumericDiff requires an expected value");
    }

    // Scale first so finite inputs cannot overflow the difference or sum.
    const scale = Math.max(Math.abs(expected), Math.abs(output));
    const expectedScaled = scale === 0 ? 0 : expected / scale;
    const outputScaled = scale === 0 ? 0 : output / scale;
    const score =
      scale === 0
        ? 1
        : 1 -
          Math.abs(expectedScaled - outputScaled) /
            (Math.abs(expectedScaled) + Math.abs(outputScaled));

    return {
      name: "NumericDiff",
      score,
    };
  },
  "NumericDiff",
);
