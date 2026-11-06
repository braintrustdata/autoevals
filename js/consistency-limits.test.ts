import { describe, expect, it } from "vitest";
import { analyzeConsistency } from "./consistency";

const plan = {
  score_name: "j",
  raters: ["a", "b"],
  categories: ["n", "y"],
  cases: ["x"],
};

describe("bounded selected evidence", () => {
  it("charges empty pairwise report sections", () => {
    expect(() =>
      analyzeConsistency([], {
        plan: { ...plan, cases: [], groups: [{ name: "empty", cases: [] }] },
        limits: { max_pairwise_work: 1 },
      }),
    ).toThrow(/pairwise/);
  });
  it("charges declared names even with no observations", () => {
    expect(() =>
      analyzeConsistency([], {
        plan: { ...plan, cases: ["long-case-identifier"] },
        limits: { max_bytes: 10 },
      }),
    ).toThrow(/byte/);
  });
  it("charges references to the same evidence budget", () => {
    expect(() =>
      analyzeConsistency([], {
        plan,
        references: { x: "y" },
        limits: { max_bytes: 7 },
      }),
    ).toThrow(/byte/);
  });
  it("preflights declared group dimensions before consuming input", () => {
    function* observations(): Generator<never> {
      throw new Error("should not consume observations");
    }
    expect(() =>
      analyzeConsistency(observations(), {
        plan: { ...plan, groups: [{ name: "slice", cases: ["x", "y"] }] },
        limits: { max_group_cells: 3 },
      }),
    ).toThrow(/group cell/);
  });
});
