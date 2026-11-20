import { align } from "./alignment";
import { bootstrapIntervals } from "./bootstrap";
import { compareNames, normalizeLimits, normalizePlan } from "./config";
import {
  cohenKappa,
  counts,
  fleissKappa,
  nominalAlpha,
  pairAgreement,
  rawAgreement,
  sum,
} from "./statistics";
import type {
  CaseReport,
  ConsistencyGate,
  ConsistencyLimits,
  ConsistencyPlan,
  ConsistencyReport,
  Coverage,
  GateResult,
  GroupReport,
  NormalizedPlan,
  PanelCase,
  Rating,
  RaterReport,
  ReferenceAccuracy,
  References,
  Statistic,
  StatisticName,
} from "./types";

/** A strict majority includes all declared slots in the denominator. */
export function consensus(
  item: PanelCase,
  plan: NormalizedPlan,
): [string | null, string | null] {
  const categories = counts(item.categories);
  if (sum(categories.values()) < plan.minimum_consensus_ratings)
    return [null, "insufficient_ratings"];
  const maximum = Math.max(...categories.values());
  const winners = [...categories]
    .filter(([, n]) => n === maximum)
    .map(([category]) => category);
  if (winners.length !== 1) return [null, "tie"];
  if (maximum * 2 <= plan.raters.length)
    return [null, "no_declared_panel_majority"];
  return [winners[0], null];
}

export function referenceAccuracy(
  cases: readonly PanelCase[],
  values: readonly (string | null)[],
): ReferenceAccuracy {
  let labeled = 0,
    evaluated = 0,
    correct = 0;
  for (let i = 0; i < cases.length; i++) {
    if (cases[i].reference === null) continue;
    labeled++;
    if (values[i] !== null) {
      evaluated++;
      if (values[i] === cases[i].reference) correct++;
    }
  }
  return {
    labeled_cases: labeled,
    evaluated_cases: evaluated,
    correct_cases: correct,
    unavailable_cases: labeled - evaluated,
    coverage: labeled ? evaluated / labeled : null,
    accuracy: evaluated ? correct / evaluated : null,
    reason: !labeled
      ? "no_references"
      : !evaluated
        ? "no_evaluated_references"
        : null,
  };
}

function consensusStatistic(
  cases: readonly PanelCase[],
  plan: NormalizedPlan,
): Statistic {
  const values = cases.map((item) => consensus(item, plan)[0]);
  const accuracy = referenceAccuracy(cases, values);
  let ratings = 0;
  for (let i = 0; i < cases.length; i++)
    if (cases[i].reference !== null && values[i] !== null)
      ratings += cases[i].categories.filter((value) => value !== null).length;
  return {
    value: accuracy.accuracy,
    support_cases: accuracy.evaluated_cases,
    support_ratings: ratings,
    denominator_cases: accuracy.labeled_cases,
    observed_agreement: null,
    expected_agreement: null,
    weighting: "evaluated_reference_cases",
    reason: accuracy.reason,
  };
}

export function coefficients(
  cases: readonly PanelCase[],
  plan: NormalizedPlan,
): Record<StatisticName, Statistic> {
  return {
    raw_agreement: rawAgreement(cases),
    fleiss_kappa: fleissKappa(cases, plan.raters.length),
    nominal_alpha: nominalAlpha(cases),
    consensus_accuracy: consensusStatistic(cases, plan),
  };
}

function coverage(cases: readonly PanelCase[], plan: NormalizedPlan): Coverage {
  const states = counts(cases.flatMap((item) => [...item.statuses]));
  const available = cases.map(
    (item) => item.categories.filter((value) => value !== null).length,
  );
  const expected = cases.length * plan.raters.length;
  const scored = states.get("scored") ?? 0,
    absent = states.get("absent") ?? 0;
  return {
    total_cases: cases.length,
    expected_cells: expected,
    present_cells: expected - absent,
    scored_cells: scored,
    missing_cells: states.get("missing") ?? 0,
    error_cells: states.get("error") ?? 0,
    absent_cells: absent,
    rating_coverage: expected ? scored / expected : null,
    zero_rating_cases: available.filter((n) => n === 0).length,
    singleton_cases: available.filter((n) => n === 1).length,
    pairable_cases: available.filter((n) => n >= 2).length,
    complete_cases: available.filter((n) => n === plan.raters.length).length,
  };
}

function raterReport(
  cases: readonly PanelCase[],
  index: number,
  rater: string,
): RaterReport {
  const states = counts(cases.map((item) => item.statuses[index]));
  const values = cases.map((item) => item.categories[index]);
  const reference = referenceAccuracy(cases, values);
  reference.missing_cases = cases.filter(
    (item) => item.reference !== null && item.statuses[index] === "missing",
  ).length;
  reference.error_cases = cases.filter(
    (item) => item.reference !== null && item.statuses[index] === "error",
  ).length;
  reference.absent_cases = cases.filter(
    (item) => item.reference !== null && item.statuses[index] === "absent",
  ).length;
  const scored = states.get("scored") ?? 0;
  return {
    rater_id: rater,
    scored_cases: scored,
    missing_cases: states.get("missing") ?? 0,
    error_cases: states.get("error") ?? 0,
    absent_cases: states.get("absent") ?? 0,
    coverage: cases.length ? scored / cases.length : null,
    category_counts: [...counts(values)]
      .sort(([a], [b]) => compareNames(a, b))
      .map(([category, count]) => ({ category, count })),
    reference,
  };
}

function caseReport(item: PanelCase, plan: NormalizedPlan): CaseReport {
  const categories = counts(item.categories),
    [winner, reason] = consensus(item, plan);
  return {
    case_id: item.case_id,
    scored_ratings: sum(categories.values()),
    category_counts: plan.categories.map((category) => ({
      category,
      count: categories.get(category) ?? 0,
    })),
    raw_agreement: pairAgreement(categories),
    consensus: winner,
    consensus_reason: reason,
    reference: item.reference,
    consensus_correct:
      winner !== null && item.reference !== null
        ? winner === item.reference
        : null,
    unavailable_raters: plan.raters.flatMap((rater_id, i) =>
      item.statuses[i] === "scored"
        ? []
        : [{ rater_id, status: item.statuses[i] }],
    ),
  };
}

function gateResult(
  gate: Required<ConsistencyGate>,
  statistic: Statistic,
  availability: Coverage,
): GateResult {
  const reasons: string[] = [];
  const supportCoverage = statistic.denominator_cases
    ? statistic.support_cases / statistic.denominator_cases
    : null;
  const ratingCoverage = availability.rating_coverage;
  if (statistic.value === null)
    reasons.push(statistic.reason ?? "undefined_statistic");
  if (statistic.support_cases < gate.minimum_support)
    reasons.push("insufficient_support");
  if (supportCoverage === null || supportCoverage < gate.minimum_coverage)
    reasons.push("insufficient_statistic_coverage");
  if (ratingCoverage === null || ratingCoverage < gate.minimum_rating_coverage)
    reasons.push("insufficient_rating_coverage");
  let verdict: GateResult["verdict"];
  if (reasons.length) verdict = "inconclusive";
  else if (statistic.value !== null && statistic.value < gate.minimum) {
    verdict = "fail";
    reasons.push("below_observed_threshold");
  } else {
    verdict = "pass";
    reasons.push("meets_observed_threshold");
  }
  return {
    ...gate,
    value: statistic.value,
    support_cases: statistic.support_cases,
    denominator_cases: statistic.denominator_cases,
    support_coverage: supportCoverage,
    rating_coverage: ratingCoverage,
    verdict,
    reasons,
  };
}

/**
 * Analyze one criterion without invoking models, backends or global randomness.
 * Missing observations remain in availability denominators. Each coefficient
 * labels its own supporting cohort. Consistency alone is not correctness.
 */
export function analyzeConsistency(
  ratings: Iterable<Rating>,
  options: {
    plan: ConsistencyPlan;
    references?: References | null;
    limits?: ConsistencyLimits;
  },
): ConsistencyReport {
  const limits = normalizeLimits(options.limits),
    plan = normalizePlan(options.plan, limits);
  const panel = align(ratings, plan, options.references, limits);
  const groups: GroupReport[] = [],
    gates: GateResult[] = [];
  let remainingDetails = plan.detail_limit;
  for (const [name, indices] of panel.groups) {
    const cases = indices.map((index) => panel.cases[index]);
    const stats = coefficients(cases, plan),
      availability = coverage(cases, plan);
    const pairwise: GroupReport["pairwise"] = [];
    for (let left = 0; left < plan.raters.length; left++)
      for (let right = left + 1; right < plan.raters.length; right++)
        pairwise.push({
          left_rater: plan.raters[left],
          right_rater: plan.raters[right],
          ...cohenKappa(cases, left, right),
        });
    const decisions = cases.map((item) => consensus(item, plan));
    const reference = referenceAccuracy(
      cases,
      decisions.map(([value]) => value),
    );
    reference.abstained_cases = reference.unavailable_cases;
    const reasons = counts(decisions.map(([, reason]) => reason));
    groups.push({
      name,
      coverage: availability,
      statistics: stats,
      raters: plan.raters.map((rater, index) =>
        raterReport(cases, index, rater),
      ),
      pairwise,
      consensus: {
        rule: "strict_majority_of_declared_panel",
        decided_cases: decisions.filter(([value]) => value !== null).length,
        abstained_cases: decisions.filter(([value]) => value === null).length,
        reason_counts: [...reasons]
          .sort(([a], [b]) => compareNames(a, b))
          .map(([reason, count]) => ({ reason, count })),
        reference,
      },
      intervals: bootstrapIntervals(cases, plan, name, stats),
      cases: cases
        .slice(0, remainingDetails)
        .map((item) => caseReport(item, plan)),
      omitted_cases: Math.max(0, cases.length - remainingDetails),
    });
    remainingDetails = Math.max(0, remainingDetails - cases.length);
    for (const gate of plan.gates)
      if (gate.group === name)
        gates.push(gateResult(gate, stats[gate.statistic], availability));
  }
  const verdict = !gates.length
    ? "not_evaluated"
    : gates.some((gate) => gate.verdict === "fail")
      ? "fail"
      : gates.some((gate) => gate.verdict === "inconclusive")
        ? "inconclusive"
        : "pass";
  return {
    format: "autoevals.consistency.report",
    version: 1,
    score_name: plan.score_name,
    cohort_provenance: panel.provenance,
    categories: [...plan.categories],
    raters: [...plan.raters],
    interpretation: "observed_finite_benchmark",
    interval_interpretation: "marginal_diagnostic_case_bootstrap",
    groups,
    gates,
    verdict,
  };
}
