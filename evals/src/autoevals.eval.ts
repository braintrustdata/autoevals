import { Eval, EvalCase, wrapTraced } from "braintrust";
import path from "path";
import fs from "fs";
import {
  closedQACaseSchema,
  contextRelevancyCaseSchema,
  coqaCaseSchema,
  dataDir,
} from "./datasets";
import { z } from "zod";
import {
  AnswerCorrectness,
  ClosedQA,
  ContextRelevancy,
  DEFAULT_MODEL,
  Factuality,
  NumericDiff,
  Score,
} from "autoevals";

const experimentNamePrefix = process.env.EXPERIMENT_NAME;

const datasets = [
  {
    name: "Factuality",
    path: path.join(dataDir, "coqa-factuality.json"),
    parser: coqaCaseSchema,
  },
  {
    name: "ClosedQA",
    path: path.join(dataDir, "coqa-closed-qa.json"),
    parser: closedQACaseSchema,
  },
  {
    name: "AnswerCorrectness",
    path: path.join(dataDir, "coqa-factuality.json"),
    parser: coqaCaseSchema,
    tags: ["ragas"],
  },
  {
    name: "ContextRelevancy",
    path: path.join(dataDir, "coqa-context-relevancy.json"),
    parser: contextRelevancyCaseSchema,
    tags: ["ragas"],
  },
];

const runScorerT = wrapTraced(async function runScorer(
  scorer: string,
  input: any,
) {
  switch (scorer) {
    case "Factuality":
      return Factuality(input);
    case "ClosedQA":
      return ClosedQA(input);
    case "AnswerCorrectness":
      return AnswerCorrectness(input);
    case "ContextRelevancy":
      return ContextRelevancy(input);
    default:
      throw new Error(`Unknown scorer: ${scorer}`);
  }
});

Eval("Autoevals", {
  data: () =>
    datasets.flatMap(({ name, path, parser, tags }) => {
      const data = fs.readFileSync(path, "utf-8");
      return z
        .array(parser)
        .parse(JSON.parse(data))
        .map((d: EvalCase<any, any, object>) => ({
          ...d,
          input: { ...d.input, scorer: name },
          metadata: { ...d.metadata, scorer: name },
          tags: [...(tags ?? []), name],
        }));
    }),
  task: async (input, hooks) => {
    const { scorer, ...rest } = input;
    let result: Score | null = null;
    try {
      result = await runScorerT(scorer, rest);
    } catch (e) {
      hooks.meta({ error: `${e}` });
    }
    return result?.score ?? -1;
  },
  scores: [NumericDiff],
  experimentName: experimentNamePrefix ?? undefined,
  metadata: {
    model: DEFAULT_MODEL,
  },
});
