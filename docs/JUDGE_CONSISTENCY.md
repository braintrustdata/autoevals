# Repeated judge consistency

Similar average scores can hide different behavior: one judge can repeatedly
change its answer on the same case while another gives stable answers. The
consistency utilities analyze **one categorical criterion** over an explicit
case-by-rater panel. A rater can mean a named model, human annotator, or fixed
repeat slot of the same judge. This is an offline analysis API, not a Scorer
that sends another model request.

Consistency is not correctness. A panel can agree perfectly on an incorrect
answer. Optional reference labels provide a separate accuracy report. Shared
errors, correlated repeat slots and unrepresentative benchmarks remain study
design concerns; the library cannot establish an independent ground truth.

## Python

```python
from autoevals import (
    ConsistencyGate, ConsistencyPlan, Rating, analyze_consistency,
    consistency_scores, render_consistency_markdown,
)

plan = ConsistencyPlan(
    score_name="answer_supported",
    raters=("repeat-1", "repeat-2", "repeat-3"),
    categories=("no", "yes"),
    cases=("case-1", "case-2"),
    gates=(ConsistencyGate("raw_agreement", minimum=0.9),),
)
ratings = [
    Rating("case-1", "repeat-1", "yes"),
    Rating("case-1", "repeat-2", "yes"),
    Rating("case-1", "repeat-3", "no"),
    Rating("case-2", "repeat-1", status="error"),
]
report = analyze_consistency(
    ratings, plan=plan, references={"case-1": "yes", "case-2": "no"}
)
print(render_consistency_markdown(report))
scores = consistency_scores(report)
```

## TypeScript

```typescript
import {
  analyzeConsistency,
  consistencyScores,
  renderConsistencyMarkdown,
} from "autoevals";

const report = analyzeConsistency(
  [
    { case_id: "case-1", rater_id: "repeat-1", category: "yes" },
    { case_id: "case-1", rater_id: "repeat-2", category: "yes" },
    { case_id: "case-1", rater_id: "repeat-3", category: "no" },
    { case_id: "case-2", rater_id: "repeat-1", status: "error" },
  ],
  {
    plan: {
      score_name: "answer_supported",
      raters: ["repeat-1", "repeat-2", "repeat-3"],
      categories: ["no", "yes"],
      cases: ["case-1", "case-2"],
      gates: [{ statistic: "raw_agreement", minimum: 0.9 }],
    },
    references: { "case-1": "yes", "case-2": "no" },
  },
);
console.log(renderConsistencyMarkdown(report));
const scores = consistencyScores(report);
```

Both examples return an `inconclusive` requirement verdict: the missing
observations prevent the stated support and coverage requirements from being
met. They are not silently discarded to make the panel look reliable.

## Collecting native Scores

Use `ScoreMapping` and `rating_from_score` in Python, or `ratingFromScore` with a
`ScoreMapping` in TypeScript. Supply exact numeric/category pairs, for example
`[(0, "no"), (1, "yes")]` / `[[0, "no"], [1, "yes"]]`. The adapter checks the
Score name and rejects unmapped scores, booleans and numeric strings. It does
not infer thresholds or round continuous scores into labels. Define any
discretization policy explicitly in your application before using this API.

A native Score with `score=None` / `null` becomes `missing`. A non-null legacy
Score `error` takes precedence over its numeric value and becomes `error`.
Modern scorers throw exceptions: catch them in your collection code and call
`error_rating` / `errorRating`. Error messages and provider metadata are never
copied into a Rating. Do not use the same rater slot twice for a case; record a
new repeat with a different declared rater identity.

An explicit `cases` manifest retains cases for which **no judge returned**.
Omitting it uses the observed union and records `cohort_provenance=observed_union`;
the API cannot discover entirely absent cases. References and groups must be
subsets of the cohort, so use a manifest if they include unobserved cases.
`missing`, `error` and an absent cell have separate counts. Only `scored` cells
enter categorical statistics. Inputs must use valid Unicode strings; identifiers
are compared exactly and sorted by UTF-8 bytes without Unicode normalization.

## Statistics and denominators

| Output               | Supporting observations                          | Interpretation                                                                                                          |
| -------------------- | ------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------- |
| Raw agreement        | Cases with at least two scored ratings           | Mean per-case ordered-pair agreement; each supporting case has equal weight                                             |
| Fleiss kappa         | Complete declared panels only                    | Fixed-panel coefficient with pooled category marginals                                                                  |
| Nominal alpha        | All cases with at least two scored ratings       | Krippendorff nominal coincidence weighting; singleton ratings are excluded from both observed and expected disagreement |
| Pairwise Cohen kappa | Overlap for that fixed rater pair                | Each marginal uses the same pairwise overlap; confusion cells are sparse                                                |
| Consensus accuracy   | Reference-labeled cases with a decided consensus | Conditional accuracy, reported together with labeled/evaluated counts and reference coverage                            |

Each coefficient includes supporting cases and ratings, denominator cases,
weighting, and an explicit undefined reason. Zero expected disagreement is
undefined, not perfect agreement. A constant-varying pair may have defined
zero Cohen kappa. Negative coefficients remain negative.

Fleiss follows the fixed-panel formulation in
[statsmodels](https://www.statsmodels.org/stable/generated/statsmodels.stats.inter_rater.fleiss_kappa.html).
Nominal alpha follows the coincidence construction in
[Krippendorff, Computing Alpha-Reliability](https://www.asc.upenn.edu/sites/default/files/2021-03/Computing%20Krippendorff's%20Alpha-Reliability.pdf).
The shared tests include that paper's missing-data example.

Consensus requires strictly more than half the **declared** rater slots to
select one category, plus `minimum_consensus_ratings` (default 2) scored
ratings. Missing slots cannot make one remaining judge unanimous. Ties,
insufficient ratings, and absence of a declared-panel majority are distinct
abstention reasons. Reference labels never break a tie.

## Requirements, groups and diagnostic intervals

Predeclare `groups` as named lists of case IDs; `overall` is reserved. Groups
may overlap. A gate selects one of `raw_agreement`, `fleiss_kappa`,
`nominal_alpha`, or `consensus_accuracy`, a `minimum` value, and optionally a
group. Defaults require at least two supporting cases, 100% statistic coverage
and 100% scored-cell coverage. Override them explicitly for a different study.
Accuracy's statistic coverage denominator is reference-labeled cases, not all
cases; its rating coverage denominator still includes every planned cell.

An undefined value or insufficient support/coverage produces `inconclusive`.
Only sufficiently supported observations can pass or fail their value threshold.
Any failed gate makes the overall verdict `fail`; otherwise any inconclusive
gate makes it `inconclusive`. With no gates the verdict is `not_evaluated`.
These are requirements on the supplied finite benchmark, not population-level
confidence claims or a universal definition of an acceptable judge.

`ConsistencyBootstrap` / the plan's `bootstrap` enables marginal diagnostic
percentile intervals (`resamples=0` disables them). Whole cases, including their
missingness and references, are resampled with replacement. Duplicate sampled
occurrences remain distinct. Each group's UTF-8 name and the configured seed
initialize a specified FNV1a/xorshift32 generator identically in both languages;
global RNG state is untouched. Intervals use linear interpolation between
ordered samples. Defaults require at least 20 supporting cases and at least
20 expected draws in each tail. Any undefined draw or degenerate distribution
makes the interval unavailable, with attempted/valid/invalid counts retained.
Intervals never determine gate decisions and are not simultaneous intervals
across multiple groups or statistics.

## Replaying and sharing evidence

`input_to_json` / `inputToJSON` serializes selected observations, the normalized
plan and references. `analyze_consistency_json` / `analyzeConsistencyJSON`
validates that versioned bundle and recomputes its report offline. Unknown
fields, duplicate keys (including escaped duplicates), invalid Unicode and
nonfinite values are rejected. The wire format uses snake_case in both
languages. `report_to_json` / `reportToJSON` exports a bounded finite report;
report export is not a signature or authenticity proof.

`consistency_scores` / `consistencyScores` exposes agreement, rating coverage
and consensus accuracy as native Scores. Undefined values stay null. Signed
kappa and alpha are not squeezed into Score's [0,1] range. Compact support
metadata is included; full case-level report metadata requires explicit
`include_report=True` / `{ includeReport: true }`. Case IDs, labels and reference
answers can themselves be sensitive even when provider metadata is excluded.

`ConsistencyLimits` is application-owned and is not read from portable input.
Default ceilings cover rating count, panel dimensions, group membership work,
pairwise and bootstrap work, selected bytes, JSON bytes/depth/nodes and report
details. Analysis checks declared dimensions before reading observations and
checks cumulative work before constructing matrices or sampling. `detail_limit`
is shared across overall and group sections, so omitted cases are counted.
Callers processing unusually large studies should set their own limits and
partition the study intentionally. Runtime depends on cases, raters, groups
and resamples; the API performs no scheduling or concurrent model execution.

Runnable offline examples are in `examples/judge_consistency.py` and
`examples/judge_consistency.ts`. They exercise native Score adaptation,
unavailable results, requirements, replay and summary projection without
requiring API credentials.
