import { describe, expect, it } from "vitest";
import { analyzeConsistency, ratingFromScore } from "./consistency";

const plan = {
  score_name: "judge",
  raters: ["a", "b", "c"],
  categories: ["no", "yes"],
  cases: ["one", "two"],
};

describe("categorical judge consistency", () => {
  it("keeps wholly absent cases and declared rater slots", () => {
    const report = analyzeConsistency(
      [{ case_id: "one", rater_id: "a", category: "yes" }],
      { plan },
    );
    expect(report.groups[0].coverage.expected_cells).toBe(6);
    expect(report.groups[0].coverage.absent_cells).toBe(5);
    expect(report.groups[0].cases[0].consensus).toBeNull();
  });
  it("adapts native scores using explicit exact labels without copying metadata", () => {
    const rating = ratingFromScore(
      "one",
      "a",
      { name: "judge", score: 1, metadata: { private: "secret" } },
      {
        score_name: "judge",
        entries: [
          [0, "no"],
          [1, "yes"],
        ],
      },
    );
    expect(rating).toEqual({
      case_id: "one",
      rater_id: "a",
      category: "yes",
      status: "scored",
    });
  });
  it("rejects duplicated cells and boolean scores", () => {
    const rating = { case_id: "one", rater_id: "a", category: "yes" };
    expect(() => analyzeConsistency([rating, rating], { plan })).toThrow(
      /duplicate/,
    );
    expect(() =>
      ratingFromScore("one", "a", { name: "judge", score: true } as never, {
        score_name: "judge",
        entries: [[1, "yes"]],
      }),
    ).toThrow();
  });
});
