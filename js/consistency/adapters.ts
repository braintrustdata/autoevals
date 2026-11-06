import type { Score } from "../score";
import { array, finite, identifier, normalizeRating, object } from "./config";
import type { Rating, ScoreMapping } from "./types";

/** Explicitly skipped evaluation, distinct from an absent cell. */
export function missingRating(
  caseId: string,
  raterId: string,
): Required<Rating> {
  return normalizeRating({
    case_id: caseId,
    rater_id: raterId,
    status: "missing",
  });
}

/** Record a caught scorer error without retaining exception text or metadata. */
export function errorRating(caseId: string, raterId: string): Required<Rating> {
  return normalizeRating({
    case_id: caseId,
    rater_id: raterId,
    status: "error",
  });
}

/**
 * Convert a native Score using exact caller-declared labels. Scores that raise
 * must be handled explicitly by the caller; no evaluation or retry is started.
 * The deprecated error field takes precedence over a simultaneous numeric value.
 */
export function ratingFromScore(
  caseId: string,
  raterId: string,
  score: Score,
  mapping: ScoreMapping,
): Required<Rating> {
  const config = object(
    mapping,
    ["score_name", "entries"],
    ["score_name", "entries"],
    "mapping",
  );
  const name = identifier(config.score_name, "score_name");
  const entries = array(config.entries, "mapping entries", 1000);
  if (!entries.length) throw new Error("mapping entries must be nonempty");
  const labels = new Map<number, string>();
  for (const entry of entries) {
    const pair = array(entry, "mapping entry", 2);
    if (pair.length !== 2)
      throw new Error("mapping entries require score/category pairs");
    const numeric = finite(pair[0], "mapped score", 0, 1);
    if (labels.has(numeric)) throw new Error("duplicate mapped numeric score");
    labels.set(numeric, identifier(pair[1], "category"));
  }
  if (
    score === null ||
    typeof score !== "object" ||
    !Object.prototype.hasOwnProperty.call(score, "score")
  )
    throw new Error("expected a native Score with name and score fields");
  if (score.name !== name)
    throw new Error("score name does not match the selected criterion");
  if (score.error !== undefined && score.error !== null)
    return errorRating(caseId, raterId);
  if (score.score === null) return missingRating(caseId, raterId);
  const numeric = finite(score.score, "score", 0, 1);
  const category = labels.get(numeric);
  if (category === undefined)
    throw new Error("numeric score is not mapped to a category");
  return normalizeRating({ case_id: caseId, rater_id: raterId, category });
}
