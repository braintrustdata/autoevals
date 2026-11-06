import { describe, expect, it } from "vitest";
import {
  analyzeConsistency,
  analyzeConsistencyJSON,
  inputToJSON,
  reportToJSON,
  consistencyScores,
  renderConsistencyMarkdown,
} from "./consistency";

const plan = {
  score_name: "judge",
  raters: ["a", "b"],
  categories: ["no", "yes"],
};
const ratings = [
  { case_id: "x", rater_id: "a", category: "yes" },
  { case_id: "x", rater_id: "b", category: "no" },
];

describe("portable consistency evidence", () => {
  it("renders Markdown image syntax and backslashes literally", () => {
    const report = analyzeConsistency([], {
      plan: {
        ...plan,
        score_name: "![external](https://example.com/track)\\*",
      },
    });
    expect(renderConsistencyMarkdown(report)).toContain(
      "\\!\\[external\\]\\(https://example\\.com/track\\)\\\\\\*",
    );
  });
  it("round trips selected data and recomputes its report", () => {
    const references = new Map([["x", "yes"]]);
    const report = analyzeConsistency(ratings, { plan, references });
    expect(
      analyzeConsistencyJSON(inputToJSON(ratings, { plan, references })),
    ).toEqual(report);
    expect(JSON.parse(reportToJSON(report))).toEqual(report);
  });
  it.each([
    '{"version":1,"version":1}',
    '{"x":NaN}',
    '{"x":1e999}',
    '{"x":{"a":1,"\\u0061":2}}',
    '{"x":"\\ud800"}',
  ])("rejects invalid JSON evidence: %s", (content) => {
    expect(() => analyzeConsistencyJSON(content)).toThrow();
  });
  it("rejects unknown fields and excessive structure", () => {
    const data = JSON.parse(inputToJSON(ratings, { plan }));
    data.plan.unknown = 1;
    expect(() => analyzeConsistencyJSON(JSON.stringify(data))).toThrow(
      /unknown/,
    );
    expect(() =>
      analyzeConsistencyJSON(inputToJSON(ratings, { plan }), {
        max_json_nodes: 2,
      }),
    ).toThrow(/structure/);
  });
  it("keeps prototype-looking identifiers as ordinary data", () => {
    const unusual = { ...plan, cases: ["__proto__"] };
    const references = new Map([["__proto__", "yes"]]);
    const content = inputToJSON([], { plan: unusual, references });
    expect(
      analyzeConsistencyJSON(content).groups[0].consensus.reference
        .labeled_cases,
    ).toBe(1);
  });
  it("projects bounded scores and leaves absent accuracy undefined", () => {
    const report = analyzeConsistency(ratings, { plan });
    const scores = consistencyScores(report);
    expect(scores.map((score) => score.score)).toEqual([0, 1, null]);
    expect(scores.every((score) => !score.metadata?.report)).toBe(true);
    const included = consistencyScores(report, { includeReport: true });
    report.score_name = "changed";
    expect((included[0].metadata?.report as typeof report).score_name).toBe(
      "judge",
    );
  });
  it("escapes labels in human reports", () => {
    const report = analyzeConsistency([], {
      plan: { ...plan, score_name: "judge|<script>\u001b[31m" },
    });
    const text = renderConsistencyMarkdown(report);
    expect(text).toContain("judge\\|&lt;script&gt;");
    expect(text).not.toContain("\u001b");
  });
});
