// Braintrust experiment that judges answers with Drex.
//
//   npm install autoevals braintrust
//   export NACE_API_KEY=nace_sk_...      # https://console.nace.ai
//   export BRAINTRUST_API_KEY=...
//   npx braintrust eval examples/drex/drex_eval.eval.ts
//
// Drex returns probabilities. The scorer keeps them in each score's metadata, so you can pick a
// threshold in Braintrust instead of guessing one up front.
import { Eval } from "braintrust";
import { DrexClassifier, drexQuestion } from "autoevals";

const cases = [
  {
    question: "What is the capital of France?",
    context: "Paris is the capital of France.",
    output: "Paris.",
  },
  {
    question: "What is the capital of France?",
    context: "Paris is the capital of France.",
    output: "Lyon, since 1960.",
  },
];

// The default model is drex-max. For large text-only rubrics Drex may require model: "drex-v2.0".
const grounded = DrexClassifier.partial({
  name: "Grounded",
  question: drexQuestion.noul(
    "Is every claim in the output supported by the context?",
  ),
});

Eval("Drex judge example", {
  data: () => cases.map((c) => ({ input: c })),
  task: (input) => input.output,
  scores: [
    ({ input, output }) =>
      grounded({ output, input: input.question, context: input.context }),
  ],
});
