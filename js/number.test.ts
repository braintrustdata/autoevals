import { describe, expect, test } from "vitest";
import { NumericDiff } from "./number";

describe("NumericDiff finite extremes", () => {
  test.each([
    [Number.MAX_VALUE, Number.MAX_VALUE, 1],
    [Number.MAX_VALUE, Number.MAX_VALUE / 2, 2 / 3],
    [Number.MAX_VALUE, -Number.MAX_VALUE, 0],
    [-Number.MAX_VALUE, -Number.MAX_VALUE / 2, 2 / 3],
    [1e-320, 1e-320 / 2, 2 / 3],
    [0, 0, 1],
    [0, Number.MAX_VALUE, 0],
    [10, 5, 2 / 3],
  ])("output %s and expected %s", async (output, expected, score) => {
    const result = await NumericDiff({ output, expected });
    expect(Number.isFinite(result.score)).toBe(true);
    expect(result.score).toBeCloseTo(score, 12);
  });
});
