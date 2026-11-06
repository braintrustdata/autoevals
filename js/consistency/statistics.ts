import { compareNames } from "./config";
import type { PanelCase, PairwiseReport, Statistic } from "./types";

export function counts(
  values: readonly (string | null)[],
): Map<string, number> {
  const result = new Map<string, number>();
  for (const value of values)
    if (value !== null) result.set(value, (result.get(value) ?? 0) + 1);
  return result;
}

export function sum(values: Iterable<number>): number {
  // Explicit Neumaier accumulation, shared with Python across its versions.
  // Native sum implementations differ enough to change boundary gate verdicts.
  let result = 0,
    correction = 0;
  for (const value of values) {
    const next = result + value;
    correction +=
      Math.abs(result) >= Math.abs(value)
        ? result - next + value
        : value - next + result;
    result = next;
  }
  return result + correction;
}

/** Equal unordered-pair agreement within one case. */
export function pairAgreement(categories: Map<string, number>): number | null {
  const n = sum(categories.values());
  if (n < 2) return null;
  return (
    sum([...categories.values()].map((count) => count * (count - 1))) /
    (n * (n - 1))
  );
}

function result(
  value: number | null,
  support: number,
  ratings: number,
  denominator: number,
  observed: number | null,
  expected: number | null,
  weighting: string,
  reason: string | null = null,
): Statistic {
  return {
    value,
    support_cases: support,
    support_ratings: ratings,
    denominator_cases: denominator,
    observed_agreement: observed,
    expected_agreement: expected,
    weighting,
    reason,
  };
}

/** Equal-case raw agreement, distinct from alpha's observation weighting. */
export function rawAgreement(cases: readonly PanelCase[]): Statistic {
  const eligible = cases
    .map((item) => counts(item.categories))
    .filter((item) => sum(item.values()) >= 2);
  if (!eligible.length)
    return result(
      null,
      0,
      0,
      cases.length,
      null,
      null,
      "equal_pairable_cases",
      "no_pairable_cases",
    );
  const observed =
    sum(eligible.map((item) => pairAgreement(item) ?? 0)) / eligible.length;
  return result(
    observed,
    eligible.length,
    sum(eligible.map((item) => sum(item.values()))),
    cases.length,
    observed,
    null,
    "equal_pairable_cases",
  );
}

/** Fleiss pooled-marginal coefficient uses complete, fixed-size panels only. */
export function fleissKappa(
  cases: readonly PanelCase[],
  raterCount: number,
): Statistic {
  const complete = cases.filter((item) =>
    item.categories.every((value) => value !== null),
  );
  if (!complete.length)
    return result(
      null,
      0,
      0,
      cases.length,
      null,
      null,
      "complete_panels",
      "no_complete_panels",
    );
  const pooled = new Map<string, number>();
  const agreements: number[] = [];
  for (const item of complete) {
    const categories = counts(item.categories);
    for (const [category, count] of categories)
      pooled.set(category, (pooled.get(category) ?? 0) + count);
    agreements.push(pairAgreement(categories) ?? 0);
  }
  const n = complete.length * raterCount;
  const observed = sum(agreements) / complete.length;
  const expected = sum(
    [...pooled.values()].map((count) => (count / n) * (count / n)),
  );
  const value = expected >= 1 ? null : (observed - expected) / (1 - expected);
  return result(
    value,
    complete.length,
    n,
    cases.length,
    observed,
    expected,
    "complete_panels",
    value === null ? "zero_expected_disagreement" : null,
  );
}

/**
 * Nominal Krippendorff alpha: fractional within-case coincidences and N(N−1)
 * finite-sample expected disagreement. Singletons contribute to neither term.
 * Summed disagreement avoids allocating a vocabulary-by-vocabulary matrix.
 */
export function nominalAlpha(cases: readonly PanelCase[]): Statistic {
  const pooled = new Map<string, number>();
  const disagreements: number[] = [];
  let support = 0;
  for (const item of cases) {
    const categories = counts(item.categories),
      n = sum(categories.values());
    if (n < 2) continue;
    support++;
    for (const [category, count] of categories)
      pooled.set(category, (pooled.get(category) ?? 0) + count);
    disagreements.push(
      (n * n - sum([...categories.values()].map((count) => count * count))) /
        (n - 1),
    );
  }
  const n = sum(pooled.values());
  if (!support)
    return result(
      null,
      0,
      0,
      cases.length,
      null,
      null,
      "pairable_observations",
      "no_pairable_cases",
    );
  const observedDisagreement = sum(disagreements) / n;
  const expectedDisagreement =
    (n * n - sum([...pooled.values()].map((count) => count * count))) /
    (n * (n - 1));
  const value =
    expectedDisagreement <= 0
      ? null
      : 1 - observedDisagreement / expectedDisagreement;
  return result(
    value,
    support,
    n,
    cases.length,
    1 - observedDisagreement,
    1 - expectedDisagreement,
    "pairable_observations",
    value === null ? "zero_expected_disagreement" : null,
  );
}

/** Cohen uses both marginals from the same overlapping scored-case cohort. */
export function cohenKappa(
  cases: readonly PanelCase[],
  left: number,
  right: number,
): Pick<PairwiseReport, "statistic" | "confusion"> {
  const matrix = new Map<string, Map<string, number>>();
  const leftCounts = new Map<string, number>(),
    rightCounts = new Map<string, number>();
  let total = 0,
    equal = 0;
  for (const item of cases) {
    const a = item.categories[left],
      b = item.categories[right];
    if (a === null || b === null) continue;
    let row = matrix.get(a);
    if (!row) {
      row = new Map();
      matrix.set(a, row);
    }
    row.set(b, (row.get(b) ?? 0) + 1);
    leftCounts.set(a, (leftCounts.get(a) ?? 0) + 1);
    rightCounts.set(b, (rightCounts.get(b) ?? 0) + 1);
    total++;
    if (a === b) equal++;
  }
  const confusion: PairwiseReport["confusion"] = [];
  for (const a of [...matrix.keys()].sort(compareNames))
    for (const b of [...matrix.get(a)!.keys()].sort(compareNames))
      confusion.push({ left: a, right: b, count: matrix.get(a)!.get(b)! });
  if (!total)
    return {
      statistic: result(
        null,
        0,
        0,
        cases.length,
        null,
        null,
        "pairwise_overlap",
        "no_overlap",
      ),
      confusion,
    };
  const observed = equal / total;
  const expected =
    sum(
      [...leftCounts].map(
        ([category, count]) => count * (rightCounts.get(category) ?? 0),
      ),
    ) /
    (total * total);
  const value = expected >= 1 ? null : (observed - expected) / (1 - expected);
  return {
    statistic: result(
      value,
      total,
      2 * total,
      cases.length,
      observed,
      expected,
      "pairwise_overlap",
      value === null ? "zero_expected_disagreement" : null,
    ),
    confusion,
  };
}
