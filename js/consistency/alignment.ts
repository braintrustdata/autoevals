import {
  compareNames,
  identifier,
  normalizeRating,
  selectedPlanBytes,
} from "./config";
import type {
  ConsistencyLimits,
  NormalizedPlan,
  PanelCase,
  Rating,
  References,
} from "./types";

export interface Panel {
  cases: readonly PanelCase[];
  groups: readonly (readonly [string, readonly number[]])[];
  provenance: "expected_manifest" | "observed_union";
}

/** Freeze observations and group memberships before eligibility filtering. */
export function align(
  ratings: Iterable<Rating>,
  plan: NormalizedPlan,
  references: References | null | undefined,
  limits: Required<ConsistencyLimits>,
): Panel {
  const knownRaters = new Set(plan.raters),
    categories = new Set(plan.categories);
  const declared = plan.cases === null ? null : new Set(plan.cases);
  const observed = new Map<string, Map<string, Required<Rating>>>();
  const encoder = new TextEncoder();
  let count = 0,
    selectedBytes = selectedPlanBytes(plan);
  for (const input of ratings) {
    if (++count > limits.max_ratings)
      throw new Error("consistency rating limit exceeded");
    const rating = normalizeRating(input);
    if (!knownRaters.has(rating.rater_id)) throw new Error("unknown rater_id");
    if (rating.category !== null && !categories.has(rating.category))
      throw new Error("unknown rating category");
    if (declared !== null && !declared.has(rating.case_id))
      throw new Error("case_id is outside the expected manifest");
    let cells = observed.get(rating.case_id);
    if (!cells) {
      cells = new Map();
      observed.set(rating.case_id, cells);
    }
    if (cells.has(rating.rater_id))
      throw new Error("duplicate case/rater cell");
    if (observed.size > limits.max_cases)
      throw new Error("consistency case limit exceeded");
    for (const value of [
      rating.case_id,
      rating.rater_id,
      rating.category ?? "",
      rating.status,
    ])
      selectedBytes += encoder.encode(value).length;
    if (selectedBytes > limits.max_bytes)
      throw new Error("consistency selected byte limit exceeded");
    cells.set(rating.rater_id, rating);
  }
  const caseIds = plan.cases ?? [...observed.keys()].sort(compareNames);
  if (caseIds.length * plan.raters.length > limits.max_cells)
    throw new Error("consistency cell limit exceeded");
  const knownCases = new Set(caseIds);
  const copiedReferences = new Map<string, string>();
  if (references !== null && references !== undefined) {
    if (typeof references !== "object" || Array.isArray(references))
      throw new Error("references must be a case/category mapping");
    const entries =
      references instanceof Map
        ? [...references.entries()]
        : Object.entries(references);
    if (entries.length > limits.max_cases)
      throw new Error("references must be a bounded case/category mapping");
    for (const [caseId, category] of entries) {
      identifier(caseId, "reference case_id");
      identifier(category, "reference category");
      if (!knownCases.has(caseId))
        throw new Error("reference case_id is outside the cohort");
      if (!categories.has(category))
        throw new Error("unknown reference category");
      selectedBytes +=
        encoder.encode(caseId).length + encoder.encode(category).length;
      if (selectedBytes > limits.max_bytes)
        throw new Error("consistency selected byte limit exceeded");
      copiedReferences.set(caseId, category);
    }
  }
  const groupCases =
    caseIds.length +
    plan.groups.reduce((total, group) => total + group.cases.length, 0);
  if (groupCases * plan.raters.length > limits.max_group_cells)
    throw new Error("consistency group cell limit exceeded");
  const pairCount = (plan.raters.length * (plan.raters.length - 1)) / 2;
  if (
    (groupCases + plan.groups.length + 1) * pairCount >
    limits.max_pairwise_work
  )
    throw new Error("consistency pairwise work limit exceeded");
  if (
    groupCases * plan.raters.length * plan.bootstrap.resamples >
    limits.max_bootstrap_work
  )
    throw new Error("consistency bootstrap work limit exceeded");
  const positions = new Map(caseIds.map((id, i) => [id, i]));
  const groups: [string, readonly number[]][] = [
    ["overall", caseIds.map((_, i) => i)],
  ];
  for (const group of plan.groups) {
    const indices = group.cases.map((id) => {
      const index = positions.get(id);
      if (index === undefined)
        throw new Error(
          "group cases are outside the cohort; declare expected cases",
        );
      return index;
    });
    groups.push([group.name, Object.freeze(indices)]);
  }
  const cases = caseIds.map((caseId) => {
    const cells = observed.get(caseId);
    return Object.freeze({
      case_id: caseId,
      categories: Object.freeze(
        plan.raters.map((rater) => cells?.get(rater)?.category ?? null),
      ),
      statuses: Object.freeze(
        plan.raters.map((rater) => cells?.get(rater)?.status ?? "absent"),
      ),
      reference: copiedReferences.get(caseId) ?? null,
    });
  });
  return {
    cases: Object.freeze(cases),
    groups: Object.freeze(groups),
    provenance: declared === null ? "observed_union" : "expected_manifest",
  };
}
