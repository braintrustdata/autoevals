import type {
  ConsistencyBootstrap,
  ConsistencyGate,
  ConsistencyLimits,
  ConsistencyPlan,
  NormalizedPlan,
  Rating,
  StatisticName,
} from "./types";

export const statistics: readonly StatisticName[] = [
  "raw_agreement",
  "fleiss_kappa",
  "nominal_alpha",
  "consensus_accuracy",
];
const encoder = new TextEncoder();

/** Python code-point and JavaScript UTF-16 ordering differ; use UTF-8 explicitly. */
export function compareNames(left: string, right: string): number {
  const a = encoder.encode(left),
    b = encoder.encode(right);
  for (let i = 0; i < Math.min(a.length, b.length); i++)
    if (a[i] !== b[i]) return a[i] - b[i];
  return a.length - b.length;
}

export function identifier(value: unknown, name: string): string {
  if (typeof value !== "string" || !value)
    throw new Error(`${name} must be a nonempty string`);
  for (let i = 0; i < value.length; i++) {
    const code = value.charCodeAt(i);
    if (code >= 0xd800 && code <= 0xdbff) {
      const next = value.charCodeAt(++i);
      if (!(next >= 0xdc00 && next <= 0xdfff))
        throw new Error(`${name} contains an invalid Unicode surrogate`);
    } else if (code >= 0xdc00 && code <= 0xdfff)
      throw new Error(`${name} contains an invalid Unicode surrogate`);
  }
  if (encoder.encode(value).length > 4096)
    throw new Error(`${name} exceeds the string limit`);
  return value;
}

export function integer(
  value: unknown,
  name: string,
  low: number,
  high: number,
): number {
  if (
    !Number.isSafeInteger(value) ||
    (value as number) < low ||
    (value as number) > high
  )
    throw new Error(`${name} must be an integer between ${low} and ${high}`);
  return value as number;
}

export function finite(
  value: unknown,
  name: string,
  low: number,
  high: number,
): number {
  if (
    typeof value !== "number" ||
    !Number.isFinite(value) ||
    value < low ||
    value > high
  )
    throw new Error(
      `${name} must be a finite number between ${low} and ${high}`,
    );
  return value;
}

export function array(
  value: unknown,
  name: string,
  maximum: number,
): unknown[] {
  if (!Array.isArray(value) || value.length > maximum)
    throw new Error(`${name} must be a bounded array`);
  return value;
}

export function names(
  value: unknown,
  name: string,
  minimum = 0,
  maximum = 100000,
): readonly string[] {
  const result = array(value, name, maximum).map((item) =>
    identifier(item, name),
  );
  if (result.length < minimum) throw new Error(`${name} has too few entries`);
  if (new Set(result).size !== result.length)
    throw new Error(`duplicate ${name}`);
  return Object.freeze(result.sort(compareNames));
}

export function object(
  value: unknown,
  allowed: readonly string[],
  required: readonly string[],
  name: string,
): Record<string, unknown> {
  if (value === null || typeof value !== "object" || Array.isArray(value))
    throw new Error(`${name} must be an object`);
  const record = value as Record<string, unknown>;
  if (Object.keys(record).some((key) => !allowed.includes(key)))
    throw new Error(`${name} contains unknown fields`);
  if (
    required.some((key) => !Object.prototype.hasOwnProperty.call(record, key))
  )
    throw new Error(`${name} is missing required fields`);
  return record;
}

function option(
  record: Record<string, unknown>,
  key: string,
  fallback: unknown,
): unknown {
  return record[key] === undefined ? fallback : record[key];
}

const defaultLimits: Required<ConsistencyLimits> = {
  max_ratings: 1000000,
  max_cases: 100000,
  max_raters: 64,
  max_categories: 100,
  max_groups: 50,
  max_cells: 1000000,
  max_pairwise_work: 10000000,
  max_group_cells: 2000000,
  max_bootstrap_work: 100000000,
  max_details: 10000,
  max_bytes: 64 * 1024 * 1024,
  max_json_nodes: 2000000,
  max_json_depth: 20,
};

export function normalizeLimits(
  input: ConsistencyLimits = {},
): Required<ConsistencyLimits> {
  const data = object(input, Object.keys(defaultLimits), [], "limits");
  const result = { ...defaultLimits };
  for (const key of Object.keys(result) as (keyof ConsistencyLimits)[])
    result[key] = integer(
      option(data, key, result[key]),
      key,
      1,
      Number.MAX_SAFE_INTEGER,
    );
  return Object.freeze(result);
}

function normalizeBootstrap(value: unknown): Required<ConsistencyBootstrap> {
  const data = object(
    value,
    ["resamples", "confidence", "seed", "minimum_cases", "statistics"],
    [],
    "bootstrap",
  );
  const confidence = finite(
    option(data, "confidence", 0.95),
    "confidence",
    0,
    1,
  );
  if (confidence === 0 || confidence === 1)
    throw new Error("confidence must be strictly between zero and one");
  const selected = names(
    option(data, "statistics", statistics.slice(0, 3)),
    "bootstrap statistics",
    1,
    4,
  );
  if (selected.some((name) => !statistics.includes(name as StatisticName)))
    throw new Error("unknown bootstrap statistic");
  return Object.freeze({
    resamples: integer(option(data, "resamples", 0), "resamples", 0, 100000),
    confidence,
    seed: integer(option(data, "seed", 0), "seed", 0, 0xffffffff),
    minimum_cases: integer(
      option(data, "minimum_cases", 20),
      "minimum_cases",
      2,
      100000,
    ),
    statistics: selected as readonly StatisticName[],
  });
}

function normalizeGate(value: unknown): Required<ConsistencyGate> {
  const data = object(
    value,
    [
      "statistic",
      "minimum",
      "group",
      "minimum_support",
      "minimum_coverage",
      "minimum_rating_coverage",
    ],
    ["statistic", "minimum"],
    "gate",
  );
  if (!statistics.includes(data.statistic as StatisticName))
    throw new Error("unknown gate statistic");
  const minimum = finite(data.minimum, "minimum", -1, 1);
  if (
    ["raw_agreement", "consensus_accuracy"].includes(
      data.statistic as string,
    ) &&
    minimum < 0
  )
    throw new Error("agreement and accuracy thresholds must be nonnegative");
  return Object.freeze({
    statistic: data.statistic as StatisticName,
    minimum,
    group: identifier(option(data, "group", "overall"), "gate group"),
    minimum_support: integer(
      option(data, "minimum_support", 2),
      "minimum_support",
      1,
      100000,
    ),
    minimum_coverage: finite(
      option(data, "minimum_coverage", 1),
      "minimum_coverage",
      0,
      1,
    ),
    minimum_rating_coverage: finite(
      option(data, "minimum_rating_coverage", 1),
      "minimum_rating_coverage",
      0,
      1,
    ),
  });
}

/** Detach caller-owned arrays and normalize every runtime field, not just TS types. */
export function normalizePlan(
  input: ConsistencyPlan,
  limits: Required<ConsistencyLimits>,
): NormalizedPlan {
  const data = object(
    input,
    [
      "score_name",
      "raters",
      "categories",
      "cases",
      "groups",
      "gates",
      "minimum_consensus_ratings",
      "detail_limit",
      "bootstrap",
    ],
    ["score_name", "raters", "categories"],
    "plan",
  );
  const raters = names(data.raters, "raters", 2, 1000);
  const categories = names(data.categories, "categories", 2, 1000);
  const cases =
    data.cases === undefined || data.cases === null
      ? null
      : names(data.cases, "cases");
  const groups = array(option(data, "groups", []), "groups", 1000)
    .map((value) => {
      const group = object(
        value,
        ["name", "cases"],
        ["name", "cases"],
        "group",
      );
      const name = identifier(group.name, "group name");
      if (name === "overall") throw new Error("overall is reserved");
      return Object.freeze({ name, cases: names(group.cases, "group cases") });
    })
    .sort((a, b) => compareNames(a.name, b.name));
  if (new Set(groups.map((group) => group.name)).size !== groups.length)
    throw new Error("duplicate group name");
  const gates = array(option(data, "gates", []), "gates", 1000)
    .map(normalizeGate)
    .sort(
      (a, b) =>
        compareNames(a.group, b.group) ||
        compareNames(a.statistic, b.statistic),
    );
  const knownGroups = new Set([
    "overall",
    ...groups.map((group) => group.name),
  ]);
  const gateKeys = new Set<string>();
  for (const gate of gates) {
    if (!knownGroups.has(gate.group))
      throw new Error("gate references an unknown group");
    const key = JSON.stringify([gate.group, gate.statistic]);
    if (gateKeys.has(key)) throw new Error("duplicate gate statistic/group");
    gateKeys.add(key);
  }
  const result = Object.freeze({
    score_name: identifier(data.score_name, "score_name"),
    raters,
    categories,
    cases,
    groups: Object.freeze(groups),
    gates: Object.freeze(gates),
    minimum_consensus_ratings: integer(
      option(data, "minimum_consensus_ratings", 2),
      "minimum_consensus_ratings",
      2,
      raters.length,
    ),
    detail_limit: integer(
      option(data, "detail_limit", 100),
      "detail_limit",
      0,
      100000,
    ),
    bootstrap: normalizeBootstrap(option(data, "bootstrap", {})),
  });
  for (const [actual, limit, label] of [
    [raters.length, limits.max_raters, "raters"],
    [categories.length, limits.max_categories, "categories"],
    [cases?.length ?? 0, limits.max_cases, "cases"],
    [groups.length, limits.max_groups, "groups"],
    [result.detail_limit, limits.max_details, "details"],
  ] as const)
    if (actual > limit) throw new Error(`consistency ${label} limit exceeded`);
  if ((cases?.length ?? 0) * raters.length > limits.max_cells)
    throw new Error("consistency cell limit exceeded");
  const declaredGroupCases =
    (cases?.length ?? 0) +
    groups.reduce((total, group) => total + group.cases.length, 0);
  if (declaredGroupCases * raters.length > limits.max_group_cells)
    throw new Error("consistency group cell limit exceeded");
  if (selectedPlanBytes(result) > limits.max_bytes)
    throw new Error("consistency selected byte limit exceeded");
  return result;
}

/** Count declared identities even if every planned observation is absent. */
export function selectedPlanBytes(plan: NormalizedPlan): number {
  let total = encoder.encode(plan.score_name).length;
  for (const values of [plan.raters, plan.categories, plan.cases ?? []])
    for (const value of values) total += encoder.encode(value).length;
  for (const group of plan.groups) {
    total += encoder.encode(group.name).length;
    for (const value of group.cases) total += encoder.encode(value).length;
  }
  return total;
}

export function normalizeRating(value: Rating): Required<Rating> {
  const data = object(
    value,
    ["case_id", "rater_id", "category", "status"],
    ["case_id", "rater_id"],
    "rating",
  );
  const status = option(data, "status", "scored");
  if (!["scored", "missing", "error"].includes(status as string))
    throw new Error("invalid rating status");
  const category = option(data, "category", null);
  if (status === "scored") identifier(category, "category");
  else if (category !== null)
    throw new Error("unavailable ratings cannot contain a category");
  return Object.freeze({
    case_id: identifier(data.case_id, "case_id"),
    rater_id: identifier(data.rater_id, "rater_id"),
    category: category as string | null,
    status: status as Required<Rating>["status"],
  });
}
