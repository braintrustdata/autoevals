import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { analyzeConsistencyJSON } from "./consistency";

const studies = JSON.parse(
  readFileSync("fixtures/consistency/studies.json", "utf8"),
);

function subset(actual: unknown, expected: unknown): void {
  if (typeof expected === "number") expect(actual).toBeCloseTo(expected, 12);
  else if (Array.isArray(expected)) {
    expect(Array.isArray(actual)).toBe(true);
    expected.forEach((value, i) => subset((actual as unknown[])[i], value));
  } else if (expected !== null && typeof expected === "object") {
    for (const [key, value] of Object.entries(expected))
      subset((actual as Record<string, unknown>)[key], value);
  } else expect(actual).toEqual(expected);
}

describe("shared Python/TypeScript study fixtures", () => {
  for (const study of studies)
    it(study.name, () => {
      const report = analyzeConsistencyJSON(JSON.stringify(study.input));
      subset(report, study.expected);
      const reversed = {
        ...study.input,
        ratings: [...study.input.ratings].reverse(),
      };
      expect(analyzeConsistencyJSON(JSON.stringify(reversed))).toEqual(report);
    });
});
