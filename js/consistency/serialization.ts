import { align } from "./alignment";
import { analyzeConsistency } from "./analysis";
import {
  array,
  normalizeLimits,
  normalizePlan,
  normalizeRating,
  object,
} from "./config";
import type {
  ConsistencyLimits,
  ConsistencyPlan,
  ConsistencyReport,
  Rating,
  References,
} from "./types";

const encoder = new TextEncoder();

function validUnicode(value: string): void {
  for (let i = 0; i < value.length; i++) {
    const code = value.charCodeAt(i);
    if (code >= 0xd800 && code <= 0xdbff) {
      const next = value.charCodeAt(++i);
      if (!(next >= 0xdc00 && next <= 0xdfff))
        throw new Error("invalid Unicode in JSON");
    } else if (code >= 0xdc00 && code <= 0xdfff)
      throw new Error("invalid Unicode in JSON");
  }
}

/** Bound parsed trees, including object keys. Never silently serialize NaN as null. */
function structure(value: unknown, limits: Required<ConsistencyLimits>): void {
  const stack: [unknown, number][] = [[value, 0]];
  let count = 0;
  while (stack.length) {
    const [item, depth] = stack.pop()!;
    if (++count > limits.max_json_nodes || depth > limits.max_json_depth)
      throw new Error("consistency JSON structure limit exceeded");
    if (typeof item === "string") {
      validUnicode(item);
      if (encoder.encode(item).length > 16384)
        throw new Error("consistency JSON string limit exceeded");
    } else if (typeof item === "number") {
      if (!Number.isFinite(item))
        throw new Error("nonfinite JSON numbers are not supported");
    } else if (Array.isArray(item)) {
      for (const child of item) stack.push([child, depth + 1]);
    } else if (item !== null && typeof item === "object") {
      for (const [key, child] of Object.entries(item))
        stack.push([key, depth + 1], [child, depth + 1]);
    } else if (item !== null && typeof item !== "boolean") {
      throw new Error("consistency data must be finite JSON");
    }
  }
}

/**
 * JSON.parse validates grammar first. This bounded lexical pass only retains
 * object key sets, detecting escaped duplicate names before evidence is used.
 * Values and numeric syntax are left to the standard parser.
 */
function uniqueKeys(
  content: string,
  limits: Required<ConsistencyLimits>,
): void {
  const stack: (Set<string> | null)[] = [];
  for (let i = 0; i < content.length; i++) {
    const char = content[i];
    if (char === "{" || char === "[") {
      stack.push(char === "{" ? new Set() : null);
      if (stack.length > limits.max_json_depth + 1)
        throw new Error("consistency JSON structure limit exceeded");
    } else if (char === "}" || char === "]") {
      stack.pop();
    } else if (char === '"') {
      const start = i;
      while (++i < content.length) {
        if (content[i] === "\\") i++;
        else if (content[i] === '"') break;
      }
      let next = i + 1;
      while (next < content.length && /\s/.test(content[next])) next++;
      if (content[next] === ":") {
        const key: string = JSON.parse(content.slice(start, i + 1));
        const keys = stack[stack.length - 1];
        if (!keys) throw new Error("invalid JSON object key");
        if (keys.has(key)) throw new Error("duplicate JSON key");
        keys.add(key);
      }
    }
  }
}

function encode(value: unknown, limits: Required<ConsistencyLimits>): string {
  structure(value, limits);
  const content = JSON.stringify(value);
  if (encoder.encode(content).length > limits.max_bytes)
    throw new Error("consistency output byte limit exceeded");
  return content;
}

/** Recompute a versioned portable evidence bundle entirely offline. */
export function analyzeConsistencyJSON(
  content: string,
  limitsInput: ConsistencyLimits = {},
): ConsistencyReport {
  const limits = normalizeLimits(limitsInput);
  if (typeof content !== "string")
    throw new Error("input JSON must be a string");
  validUnicode(content);
  if (encoder.encode(content).length > limits.max_bytes)
    throw new Error("consistency input byte limit exceeded");
  const value: unknown = JSON.parse(content);
  uniqueKeys(content, limits);
  structure(value, limits);
  const data = object(
    value,
    ["format", "version", "plan", "ratings", "references"],
    ["format", "version", "plan", "ratings"],
    "input bundle",
  );
  if (data.format !== "autoevals.consistency.input" || data.version !== 1)
    throw new Error("unsupported consistency input format/version");
  const plan = normalizePlan(data.plan as ConsistencyPlan, limits);
  const ratings = array(data.ratings, "ratings", limits.max_ratings).map(
    (row) => normalizeRating(row as Rating),
  );
  return analyzeConsistency(ratings, {
    plan,
    references: data.references as References | undefined,
    limits,
  });
}

/** Snapshot only selected observations; provider metadata and error messages are excluded. */
export function inputToJSON(
  ratings: Iterable<Rating>,
  options: {
    plan: ConsistencyPlan;
    references?: References | null;
    limits?: ConsistencyLimits;
  },
): string {
  const limits = normalizeLimits(options.limits);
  const plan = normalizePlan(options.plan, limits);
  const collected: Required<Rating>[] = [];
  for (const rating of ratings) {
    if (collected.length >= limits.max_ratings)
      throw new Error("consistency rating limit exceeded");
    collected.push(normalizeRating(rating));
  }
  const panel = align(collected, plan, options.references, limits);
  const references = Object.fromEntries(
    panel.cases
      .filter((row) => row.reference !== null)
      .map((row) => [row.case_id, row.reference]),
  );
  return encode(
    {
      format: "autoevals.consistency.input",
      version: 1,
      plan,
      ratings: collected,
      references,
    },
    limits,
  );
}

/** Serialize finite evidence; the format is not a proof of authenticity. */
export function reportToJSON(
  report: ConsistencyReport,
  limits: ConsistencyLimits = {},
): string {
  if (report.format !== "autoevals.consistency.report" || report.version !== 1)
    throw new Error("unsupported consistency report");
  return encode(report, normalizeLimits(limits));
}
