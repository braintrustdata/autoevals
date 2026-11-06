import { describe, expect, it } from "vitest";
import { analyzeConsistency } from "./consistency";
import { cohenKappa, nominalAlpha } from "./consistency/statistics";
import { CaseRandom } from "./consistency/bootstrap";
import type { PanelCase } from "./consistency/types";

function row(categories: (string | null)[]): PanelCase {
  return {
    case_id: "case",
    categories,
    statuses: categories.map((value) => (value === null ? "absent" : "scored")),
    reference: null,
  };
}

describe("statistical interpretation", () => {
  it("reproduces the published nominal alpha missing-data example", () => {
    const raters = [
      [1, 2, 3, 3, 2, 1, 4, 1, 2, null, null, null],
      [1, 2, 3, 3, 2, 2, 4, 1, 2, 5, null, 3],
      [null, 3, 3, 3, 2, 3, 4, 2, 2, 5, 1, null],
      [1, 2, 3, 3, 2, 4, 4, 1, 2, 5, 1, null],
    ];
    const cases = raters[0].map((_, i) =>
      row(raters.map((rater) => (rater[i] === null ? null : String(rater[i])))),
    );
    const result = nominalAlpha(cases);
    expect(result.value).toBeCloseTo(0.7434210526315789, 12);
    expect(result.support_cases).toBe(11);
    expect(result.support_ratings).toBe(40);
    expect(nominalAlpha(cases.slice(0, -1)).value).toBe(result.value);
  });
  it("distinguishes defined constant-varying kappa from zero expected disagreement", () => {
    expect(
      cohenKappa([row(["yes", "yes"]), row(["yes", "no"])], 0, 1).statistic
        .value,
    ).toBe(0);
    expect(cohenKappa([row(["yes", "yes"])], 0, 1).statistic.reason).toBe(
      "zero_expected_disagreement",
    );
  });
  it("preserves signed systematic disagreement", () => {
    const cases = [row(["yes", "no"]), row(["no", "yes"])];
    expect(cohenKappa(cases, 0, 1).statistic.value).toBe(-1);
    expect(nominalAlpha(cases).value).toBe(-0.5);
  });
  it("does not call a minority of declared slots a consensus", () => {
    const plan = {
      score_name: "judge",
      raters: ["a", "b", "c", "d", "e"],
      categories: ["no", "yes"],
      cases: ["x", "y"],
    };
    const ratings = ["a", "b"]
      .map((rater_id) => ({ case_id: "x", rater_id, category: "yes" }))
      .concat(
        ["a", "b", "c"].map((rater_id) => ({
          case_id: "y",
          rater_id,
          category: "yes",
        })),
      );
    const group = analyzeConsistency(ratings, {
      plan,
      references: { x: "yes", y: "yes" },
    }).groups[0];
    expect(group.cases.map((item) => item.consensus)).toEqual([null, "yes"]);
    expect(group.consensus.reference.coverage).toBe(0.5);
    expect(group.consensus.reference.accuracy).toBe(1);
    expect(group.statistics.fleiss_kappa.value).toBeNull();
  });
  it("uses the specified cross-language PRNG vector", () => {
    const rng = new CaseRandom(17, "overall");
    expect(Array.from({ length: 5 }, () => rng.nextU32())).toEqual([
      4291421669, 2916200383, 2632349228, 374880413, 943298360,
    ]);
  });
  it("preserves undefined bootstrap draws instead of conditioning on valid draws", () => {
    const plan = {
      score_name: "judge",
      raters: ["a", "b"],
      categories: ["no", "yes"],
      bootstrap: { resamples: 1000, confidence: 0.9, minimum_cases: 2 },
    };
    const ratings = ["x", "y"].flatMap((case_id) =>
      plan.raters.map((rater_id) => ({
        case_id,
        rater_id,
        category: case_id === "x" ? "yes" : "no",
      })),
    );
    const intervals = analyzeConsistency(ratings, { plan }).groups[0].intervals;
    const alpha = intervals.find((item) => item.statistic === "nominal_alpha")!;
    expect(alpha.invalid).toBeGreaterThan(0);
    expect(alpha.lower).toBeNull();
    expect(alpha.reason).toBe("undefined_resamples");
  });
  it("bounds work before any bootstrap draws", () => {
    const plan = {
      score_name: "judge",
      raters: ["a", "b", "c"],
      categories: ["no", "yes"],
      cases: ["x", "y"],
      groups: [{ name: "slice", cases: ["x", "y"] }],
    };
    expect(() =>
      analyzeConsistency([], { plan, limits: { max_pairwise_work: 10 } }),
    ).toThrow(/pairwise/);
  });
});
