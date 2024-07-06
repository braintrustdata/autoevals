import { Eval, EvalCase, wrapTraced } from "braintrust";
import path from "path";
import fs from "fs";
import { coqaCaseSchema, dataDir } from "./datasets";
import { z } from "zod";
import { DEFAULT_MODEL, Factuality, NumericDiff } from "autoevals";

const experimentNamePrefix = process.env.EXPERIMENT_NAME;

const datasets = [
  {
    name: "Factuality",
    path: path.join(dataDir, "coqa.json"),
    parser: coqaCaseSchema,
  },
];

const runScorer = wrapTraced(async function runScorer(
  scorer: string,
  input: any
) {
  switch (scorer) {
    case "Factuality":
      return Factuality(input);
    default:
      throw new Error(`Unknown scorer: ${scorer}`);
  }
});

Eval("Autoevals", {
  data: () =>
    datasets.flatMap(({ name, path, parser }) => {
      const data = fs.readFileSync(path, "utf-8");
      return z
        .array(parser)
        .parse(JSON.parse(data))
        .map((d: EvalCase<any, any, object>) => ({
          ...d,
          input: { ...d.input, scorer: name },
          metadata: { ...d.metadata, scorer: name },
          tags: [name],
        }));
    }),
  task: async (input) => {
    const { scorer, ...rest } = input;
    const result = await runScorer(scorer, rest);
    return result.score ?? -1;
  },
  scores: [NumericDiff],
  experimentName: experimentNamePrefix ?? undefined,
  metadata: {
    model: DEFAULT_MODEL,
  },
});
