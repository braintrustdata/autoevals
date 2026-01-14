/**
 * JSON evaluation scorers for comparing and validating JSON data.
 *
 * This module provides scorers for working with JSON data:
 *
 * - **JSONDiff**: Compare JSON objects for structural and content similarity
 * - **ValidJSON**: Validate if a value is valid JSON and matches an optional schema
 *
 * ## Creating Custom JSON Scorers
 *
 * You can create custom JSON scorers by composing existing scorers or building new ones:
 *
 * @example
 * ```typescript
 * import { Scorer } from "autoevals";
 * import { JSONDiff, ValidJSON } from "autoevals/json";
 * import { EmbeddingSimilarity } from "autoevals/string";
 *
 * // Custom scorer that validates JSON schema then compares semantically
 * const myJSONScorer: Scorer<any, { schema?: any }> = async ({ output, expected, schema }) => {
 *   // First, validate both outputs against schema
 *   const outputValid = await ValidJSON({ output, schema });
 *   const expectedValid = await ValidJSON({ output: expected, schema });
 *
 *   if (outputValid.score === 0 || expectedValid.score === 0) {
 *     return {
 *       name: "CustomJSONScorer",
 *       score: 0,
 *       error: "Invalid JSON format"
 *     };
 *   }
 *
 *   // Then compare using semantic similarity for strings
 *   return JSONDiff({
 *     output,
 *     expected,
 *     stringScorer: EmbeddingSimilarity
 *   });
 * };
 *
 * // Custom scorer for specific JSON structure validation
 * const apiResponseScorer: Scorer<any, object> = async ({ output }) => {
 *   const parsed = typeof output === "string" ? JSON.parse(output) : output;
 *
 *   let score = 0;
 *   const errors: string[] = [];
 *
 *   // Check required fields
 *   if (parsed.status) score += 0.3;
 *   else errors.push("Missing status field");
 *
 *   if (parsed.data) score += 0.3;
 *   else errors.push("Missing data field");
 *
 *   // Check data structure
 *   if (parsed.data?.items && Array.isArray(parsed.data.items)) {
 *     score += 0.4;
 *   } else {
 *     errors.push("data.items must be an array");
 *   }
 *
 *   return {
 *     name: "APIResponseScorer",
 *     score: Math.min(score, 1),
 *     metadata: { errors }
 *   };
 * };
 * ```
 */

import { Scorer } from "./score";
import { NumericDiff } from "./number";
import { LevenshteinScorer } from "./string";
import Ajv, { JSONSchemaType, Schema } from "ajv";
import { makePartial, ScorerWithPartial } from "./partial";

/**
 * Compare JSON objects for structural and content similarity.
 *
 * This scorer recursively compares JSON objects, handling:
 * - Nested dictionaries and arrays
 * - String similarity using Levenshtein distance (or custom scorer)
 * - Numeric value comparison (or custom scorer)
 * - Automatic parsing of JSON strings
 *
 * @example
 * ```typescript
 * import { JSONDiff } from "autoevals";
 * import { EmbeddingSimilarity } from "autoevals/string";
 *
 * // Basic comparison
 * const result = await JSONDiff({
 *   output: {
 *     name: "John Smith",
 *     age: 30,
 *     skills: ["python", "javascript"]
 *   },
 *   expected: {
 *     name: "John A. Smith",
 *     age: 31,
 *     skills: ["python", "typescript"]
 *   }
 * });
 * console.log(result.score); // Similarity score between 0-1
 *
 * // With custom string scorer using embeddings
 * const semanticResult = await JSONDiff({
 *   output: { description: "A fast car" },
 *   expected: { description: "A quick automobile" },
 *   stringScorer: EmbeddingSimilarity
 * });
 * ```
 *
 * @param output - The JSON object or string to evaluate
 * @param expected - The expected JSON object or string to compare against
 * @param stringScorer - Optional custom scorer for string comparisons (default: LevenshteinScorer)
 * @param numberScorer - Optional custom scorer for number comparisons (default: NumericDiff)
 * @param preserveStrings - Don't attempt to parse strings as JSON (default: false)
 * @returns Score object with similarity score between 0-1
 */
export const JSONDiff: ScorerWithPartial<
  any,
  {
    stringScorer?: Scorer<string, object>;
    numberScorer?: Scorer<number, object>;
    preserveStrings?: boolean;
  }
> = makePartial(
  async ({
    output,
    expected,
    stringScorer = LevenshteinScorer,
    numberScorer = NumericDiff,
    preserveStrings = false,
  }) => {
    return {
      name: "JSONDiff",
      score: await jsonDiff(
        output,
        expected,
        stringScorer,
        numberScorer,
        preserveStrings,
      ),
    };
  },
  "JSONDiff",
);

/**
 * Validate if a value is valid JSON and optionally matches a JSON Schema.
 *
 * This scorer checks if:
 * - The input can be parsed as valid JSON (if it's a string)
 * - The parsed JSON matches an optional JSON Schema
 * - Handles both string inputs and pre-parsed JSON objects
 *
 * @example
 * ```typescript
 * import { ValidJSON } from "autoevals";
 *
 * // Basic JSON validation
 * const result1 = await ValidJSON({
 *   output: '{"name": "John", "age": 30}'
 * });
 * console.log(result1.score); // 1 (valid JSON)
 *
 * const result2 = await ValidJSON({
 *   output: '{invalid json}'
 * });
 * console.log(result2.score); // 0 (invalid JSON)
 *
 * // With schema validation
 * const schema = {
 *   type: "object",
 *   properties: {
 *     name: { type: "string" },
 *     age: { type: "number" }
 *   },
 *   required: ["name", "age"]
 * };
 *
 * const result3 = await ValidJSON({
 *   output: { name: "John", age: 30 },
 *   schema
 * });
 * console.log(result3.score); // 1 (matches schema)
 *
 * const result4 = await ValidJSON({
 *   output: { name: "John" }, // missing required "age"
 *   schema
 * });
 * console.log(result4.score); // 0 (doesn't match schema)
 * ```
 *
 * @param output - The value to validate (string or object)
 * @param schema - Optional JSON Schema to validate against (see https://json-schema.org)
 * @returns Score object with score of 1 if valid, 0 otherwise
 */
export const ValidJSON: ScorerWithPartial<any, { schema?: any }> = makePartial(
  async ({ output, schema }) => {
    return {
      name: "ValidJSON",
      score: validJSON(output, schema),
      metadata: { schema },
    };
  },
  "ValidJSON",
);

async function jsonDiff(
  o1: any,
  o2: any,
  stringScorer: Scorer<string, object>,
  numberScorer: Scorer<number, object>,
  preserveStrings: boolean,
): Promise<number | null> {
  if (!preserveStrings) {
    if (typeof o1 === "string" && validJSON(o1) === 1) {
      o1 = JSON.parse(o1);
    }
    if (typeof o2 === "string" && validJSON(o2) === 1) {
      o2 = JSON.parse(o2);
    }
  }

  if (isObject(o1) && isObject(o2)) {
    if (Object.keys(o1).length == 0 && Object.keys(o2).length == 0) {
      return 1;
    }

    const allKeys = Object.keys(
      Object.fromEntries(
        Object.keys(o1)
          .concat(Object.keys(o2))
          .map((k) => [k, true]),
      ),
    );

    // eslint-disable-next-line @typescript-eslint/consistent-type-assertions
    const baseScores = (
      await Promise.all(
        allKeys.map((k) =>
          jsonDiff(o1[k], o2[k], stringScorer, numberScorer, preserveStrings),
        ),
      )
    ).filter((s) => s !== null) as number[];
    return baseScores.reduce((acc, s) => acc + s, 0) / baseScores.length;
  } else if (isArray(o1) && isArray(o2)) {
    if (o1.length === 0 && o2.length === 0) {
      return 1;
    }

    // eslint-disable-next-line @typescript-eslint/consistent-type-assertions
    const baseScores = (
      await Promise.all(
        Array.from({
          length: Math.min(o1.length, o2.length),
        }).map((_, i) =>
          jsonDiff(o1[i], o2[i], stringScorer, numberScorer, preserveStrings),
        ),
      )
    ).filter((s) => s !== null) as number[];
    return (
      baseScores.reduce((acc, s) => acc + s, 0) / Math.max(o1.length, o2.length)
    );
  } else if (typeof o1 === "string" && typeof o2 === "string") {
    return (await stringScorer({ output: o1, expected: o2 })).score;
  } else if (typeof o1 === "number" && typeof o2 === "number") {
    return (await numberScorer({ output: o1, expected: o2 })).score;
  } else if (
    (o1 === null || o1 === undefined) &&
    (o2 === null || o2 === undefined)
  ) {
    return 1;
  } else if (
    o1 === null ||
    o1 === undefined ||
    o2 === null ||
    o2 === undefined
  ) {
    return 0;
  } else {
    return (
      await stringScorer({
        output: JSON.stringify(o1, replacer),
        expected: JSON.stringify(o2, replacer),
      })
    ).score;
  }
}

function isObject(value: any): value is { [key: string]: any } {
  return value instanceof Object && !(value instanceof Array);
}

function isArray(value: any): value is Array<unknown> {
  return value instanceof Array;
}

// https://gist.github.com/davidfurlong/463a83a33b70a3b6618e97ec9679e490
const replacer = (key: string, value: any) =>
  isObject(value)
    ? Object.keys(value)
        .sort()
        .reduce((sorted: { [key: string]: any }, key) => {
          sorted[key] = value[key];
          return sorted;
        }, {})
    : value;

function validJSON<T>(output: any, schema?: Schema | JSONSchemaType<T>) {
  try {
    const parsed = typeof output === "string" ? JSON.parse(output) : output;

    if (schema) {
      return validateSchema(parsed, schema);
    }
    if (isObject(parsed) || isArray(parsed)) {
      return 1;
    }
  } catch {
    // Ignore errors
  }

  return 0;
}

function validateSchema(data: any, schema: any) {
  const ajv = new Ajv();
  const validate = ajv.compile(schema);
  const valid = validate(data);
  return valid ? 1 : 0;
}
