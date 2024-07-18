import { makePartial, ScorerWithPartial } from "./partial";

/**
 * A simple scorer that tests whether two values are equal. If the value is an object or array,
 * it will be JSON-serialized and the strings compared for equality.
 */
export const ExactMatch: ScorerWithPartial<unknown, {}> = makePartial(
  (args) => {
    const maybeObject = needsJSON(args.output) || needsJSON(args.expected);
    const [output, expected] = [
      normalizeValue(args.output ?? null, maybeObject),
      normalizeValue(args.expected ?? null, maybeObject),
    ];

    const score = output === expected ? 1 : 0;

    return {
      name: "ExactMatch",
      score,
    };
  },
  "ExactMatch",
);

function needsJSON(value: unknown): boolean {
  return typeof value === "object" || Array.isArray(value);
}

export function normalizeValue(value: unknown, maybeObject: boolean): string {
  if (needsJSON(value)) {
    return JSON.stringify(value);
  }
  try {
    if (typeof value === "string" && maybeObject) {
      return JSON.stringify(JSON.parse(value));
    }
  } catch (e) {
    // That's ok, just return the string representation
  }
  return `${value}`;
}
