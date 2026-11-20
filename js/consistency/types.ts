/** A single criterion's categorical result; unavailable cells carry no label. */
export interface Rating {
  case_id: string;
  rater_id: string;
  category?: string | null;
  status?: "scored" | "missing" | "error";
}

export interface ScoreMapping {
  score_name: string;
  /** Exact numeric mapping entries, not stringified JSON object keys. */
  entries: readonly (readonly [number, string])[];
}

export type StatisticName =
  | "raw_agreement"
  | "fleiss_kappa"
  | "nominal_alpha"
  | "consensus_accuracy";

export interface ConsistencyGroup {
  name: string;
  cases: readonly string[];
}

/** Observed finite-benchmark requirement. Bootstrap intervals never gate. */
export interface ConsistencyGate {
  statistic: StatisticName;
  minimum: number;
  group?: string;
  minimum_support?: number;
  minimum_coverage?: number;
  minimum_rating_coverage?: number;
}

/** Marginal diagnostic case-bootstrap intervals, disabled by default. */
export interface ConsistencyBootstrap {
  resamples?: number;
  confidence?: number;
  seed?: number;
  minimum_cases?: number;
  statistics?: readonly StatisticName[];
}

export interface ConsistencyPlan {
  score_name: string;
  raters: readonly string[];
  categories: readonly string[];
  /** Omit for observed-union provenance; an empty array declares an empty cohort. */
  cases?: readonly string[] | null;
  groups?: readonly ConsistencyGroup[];
  gates?: readonly ConsistencyGate[];
  minimum_consensus_ratings?: number;
  detail_limit?: number;
  bootstrap?: ConsistencyBootstrap;
}

/** Trusted application limits; portable input files cannot raise these. */
export interface ConsistencyLimits {
  max_ratings?: number;
  max_cases?: number;
  max_raters?: number;
  max_categories?: number;
  max_groups?: number;
  max_cells?: number;
  max_pairwise_work?: number;
  max_group_cells?: number;
  max_bootstrap_work?: number;
  max_details?: number;
  max_bytes?: number;
  max_json_nodes?: number;
  max_json_depth?: number;
}

export interface Statistic {
  value: number | null;
  support_cases: number;
  support_ratings: number;
  denominator_cases: number;
  observed_agreement: number | null;
  expected_agreement: number | null;
  weighting: string;
  reason: string | null;
}

export interface ReferenceAccuracy {
  labeled_cases: number;
  evaluated_cases: number;
  correct_cases: number;
  unavailable_cases: number;
  coverage: number | null;
  accuracy: number | null;
  reason: string | null;
  missing_cases?: number;
  error_cases?: number;
  absent_cases?: number;
  abstained_cases?: number;
}

export interface Coverage {
  total_cases: number;
  expected_cells: number;
  present_cells: number;
  scored_cells: number;
  missing_cells: number;
  error_cells: number;
  absent_cells: number;
  rating_coverage: number | null;
  zero_rating_cases: number;
  singleton_cases: number;
  pairable_cases: number;
  complete_cases: number;
}

export interface RaterReport {
  rater_id: string;
  scored_cases: number;
  missing_cases: number;
  error_cases: number;
  absent_cases: number;
  coverage: number | null;
  category_counts: { category: string; count: number }[];
  reference: ReferenceAccuracy;
}

export interface CaseReport {
  case_id: string;
  scored_ratings: number;
  category_counts: { category: string; count: number }[];
  raw_agreement: number | null;
  consensus: string | null;
  consensus_reason: string | null;
  reference: string | null;
  consensus_correct: boolean | null;
  unavailable_raters: { rater_id: string; status: string }[];
}

export interface PairwiseReport {
  left_rater: string;
  right_rater: string;
  statistic: Statistic;
  /** Sparse nonzero cells: rows are the left rater, columns the right. */
  confusion: { left: string; right: string; count: number }[];
}

export interface ConsistencyInterval {
  statistic: StatisticName;
  confidence: number;
  resamples: number;
  attempted: number;
  valid: number;
  invalid: number;
  lower: number | null;
  upper: number | null;
  reason: string | null;
  seed: number;
  algorithm: "case-xorshift32-percentile-v1";
}

export interface GroupReport {
  name: string;
  coverage: Coverage;
  statistics: Record<StatisticName, Statistic>;
  raters: RaterReport[];
  pairwise: PairwiseReport[];
  consensus: {
    rule: "strict_majority_of_declared_panel";
    decided_cases: number;
    abstained_cases: number;
    reason_counts: { reason: string; count: number }[];
    reference: ReferenceAccuracy;
  };
  intervals: ConsistencyInterval[];
  cases: CaseReport[];
  omitted_cases: number;
}

export interface GateResult extends Required<ConsistencyGate> {
  value: number | null;
  support_cases: number;
  denominator_cases: number;
  support_coverage: number | null;
  rating_coverage: number | null;
  verdict: "pass" | "fail" | "inconclusive";
  reasons: string[];
}

export interface ConsistencyReport {
  format: "autoevals.consistency.report";
  version: 1;
  score_name: string;
  cohort_provenance: "expected_manifest" | "observed_union";
  categories: string[];
  raters: string[];
  interpretation: "observed_finite_benchmark";
  interval_interpretation: "marginal_diagnostic_case_bootstrap";
  groups: GroupReport[];
  gates: GateResult[];
  verdict: "not_evaluated" | "pass" | "fail" | "inconclusive";
}

export interface NormalizedPlan extends Required<ConsistencyPlan> {
  cases: readonly string[] | null;
  gates: readonly Required<ConsistencyGate>[];
  bootstrap: Required<ConsistencyBootstrap>;
}

export interface PanelCase {
  readonly case_id: string;
  readonly categories: readonly (string | null)[];
  readonly statuses: readonly string[];
  readonly reference: string | null;
}

export type References =
  | ReadonlyMap<string, string>
  | Readonly<Record<string, string>>;
