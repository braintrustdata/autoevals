import { Scorer } from "@braintrust/core";
import { NumericDiff } from "./number";
import { LevenshteinScorer } from "./string";
import Ajv, { JSONSchemaType, Schema } from "ajv";
import { makePartial, ScorerWithPartial } from "./partial";

/**
 * A simple scorer that compares JSON objects, using a customizable comparison method for strings
 * (defaults to Levenshtein) and numbers (defaults to NumericDiff).
 */
export const JSONDiff: ScorerWithPartial<
  any,
  { stringScorer?: Scorer<string, {}>; numberScorer?: Scorer<number, {}> }
> = makePartial(
  async ({
    output,
    expected,
    stringScorer = LevenshteinScorer,
    numberScorer = NumericDiff,
  }) => {
    return {
      name: "JSONDiff",
      score: await jsonDiff(output, expected, stringScorer, numberScorer),
    };
  },
  "JSONDiff",
);

/**
 * A binary scorer that evaluates the validity of JSON output, optionally validating against a
 * JSON Schema definition (see https://json-schema.org/learn/getting-started-step-by-step#create).
 */
export const ValidJSON: ScorerWithPartial<string, { schema?: any }> =
  makePartial(async ({ output, schema }) => {
    return {
      name: "ValidJSON",
      score: validJSON(output, schema),
      metadata: { schema },
    };
  }, "ValidJSON");

async function jsonDiff(
  o1: any,
  o2: any,
  stringScorer: Scorer<string, {}>,
  numberScorer: Scorer<number, {}>,
): Promise<number | null> {
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

    const baseScores = (
      await Promise.all(
        allKeys.map((k) => jsonDiff(o1[k], o2[k], stringScorer, numberScorer)),
      )
    ).filter((s) => s !== null) as number[];
    return baseScores.reduce((acc, s) => acc + s, 0) / baseScores.length;
  } else if (isArray(o1) && isArray(o2)) {
    if (o1.length === 0 && o2.length === 0) {
      return 1;
    }

    const baseScores = (
      await Promise.all(
        Array.from({
          length: Math.min(o1.length, o2.length),
        }).map((_, i) => jsonDiff(o1[i], o2[i], stringScorer, numberScorer)),
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

function validJSON<T>(output: string, schema?: Schema | JSONSchemaType<T>) {
  try {
    const parsed = JSON.parse(output);

    if (schema) {
      return validateSchema(parsed, schema);
    }
    if (isObject(parsed) || isArray(parsed)) {
      return 1;
    }
  } catch (err) {}

  return 0;
}

function validateSchema(data: any, schema: any) {
  const ajv = new Ajv();
  const validate = ajv.compile(schema);
  const valid = validate(data);
  return valid ? 1 : 0;
}
