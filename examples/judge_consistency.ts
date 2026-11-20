/** Run with pnpm exec tsx examples/judge_consistency.ts. No model calls. */
import {
  analyzeConsistency,
  analyzeConsistencyJSON,
  consistencyScores,
  errorRating,
  inputToJSON,
  ratingFromScore,
  renderConsistencyMarkdown,
  type ConsistencyPlan,
  type Rating,
  type Score,
  type ScoreMapping,
} from "autoevals";

const plan: ConsistencyPlan = {
  score_name: "supported",
  raters: ["repeat-1", "repeat-2", "repeat-3"],
  categories: ["no", "yes"],
  cases: ["case-1", "case-2", "case-3"],
  gates: [{ statistic: "raw_agreement", minimum: 0.9 }],
};
const mapping: ScoreMapping = {
  score_name: "supported",
  entries: [
    [0, "no"],
    [1, "yes"],
  ],
};
const stored: [string, (number | null)[]][] = [
  ["case-1", [1, 1, 1]],
  ["case-2", [0, 1, 0]],
  ["case-3", [null, 1, null]],
];
const ratings: Rating[] = [];
for (const [caseId, values] of stored) {
  plan.raters.forEach((raterId, i) => {
    if (caseId === "case-3" && raterId === "repeat-3") {
      ratings.push(errorRating(caseId, raterId));
    } else {
      const score: Score = {
        name: "supported",
        score: values[i],
        metadata: { rationale: "not exported" },
      };
      ratings.push(ratingFromScore(caseId, raterId, score, mapping));
    }
  });
}
const references = { "case-1": "yes", "case-2": "no", "case-3": "yes" };
const report = analyzeConsistency(ratings, { plan, references });
const bundle = inputToJSON(ratings, { plan, references });
if (JSON.stringify(analyzeConsistencyJSON(bundle)) !== JSON.stringify(report))
  throw new Error("replay changed evidence");
if (bundle.includes("rationale"))
  throw new Error("unexpected provider metadata");
if (report.verdict !== "inconclusive")
  throw new Error("missing coverage must not pass");
console.log(renderConsistencyMarkdown(report));
console.log(consistencyScores(report));
