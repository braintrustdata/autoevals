import {
  AnswerCorrectness,
  AnswerRelevancy,
  AnswerSimilarity,
  ContextEntityRecall,
  ContextPrecision,
  ContextRecall,
  ContextRelevancy,
  Faithfulness,
} from "./ragas";

const data = {
  input: "Can starred docs from different workspaces be accessed in one place?",
  output:
    "Yes, all starred docs, even from multiple different workspaces, will live in the My Shortcuts section.",
  expected:
    "Yes, all starred docs, even from multiple different workspaces, will live in the My Shortcuts section.",
  context: [
    "Not all Coda docs are used in the same way. You'll inevitably have a few that you use every week, and some that you'll only use once. This is where starred docs can help you stay organized.\n\n\n\nStarring docs is a great way to mark docs of personal importance. After you star a doc, it will live in a section on your doc list called **[My Shortcuts](https://coda.io/shortcuts)**. All starred docs, even from multiple different workspaces, will live in this section.\n\n\n\nStarring docs only saves them to your personal My Shortcuts. It doesn\u2019t affect the view for others in your workspace. If you\u2019re wanting to shortcut docs not just for yourself but also for others in your team or workspace, you\u2019ll [use pinning](https://help.coda.io/en/articles/2865511-starred-pinned-docs) instead.",
  ],
};

const retrievalMetrics = [
  { scorer: ContextEntityRecall, score: 0.69525 },
  { scorer: ContextRelevancy, score: 0.7423 },
  { scorer: ContextRecall, score: 1 },
  { scorer: ContextPrecision, score: 1 },
];

test("Ragas retrieval test", async () => {
  for (const { scorer, score } of retrievalMetrics) {
    const actualScore = await scorer({
      output: data.output,
      input: data.input,
      expected: data.expected,
      context: data.context,
    });

    if (score === 1) {
      expect(actualScore.score).toBeCloseTo(score, 4);
    }
  }
}, 600000);

const generationMetrics = [
  { scorer: AnswerRelevancy, score: 0.59 },
  { scorer: Faithfulness, score: 1 },
];

test("Ragas generation test", async () => {
  for (const { scorer, score } of generationMetrics) {
    const actualScore = await scorer({
      input: data.input,
      output: data.output,
      expected: data.expected,
      context: data.context,
      temperature: 0,
    });

    if (score === 1) {
      expect(actualScore.score).toBeCloseTo(score, 4);
    }
  }
}, 600000);

const endToEndMetrics = [
  { scorer: AnswerSimilarity, score: 1 },
  { scorer: AnswerCorrectness, score: 1 },
];

test("Ragas end-to-end test", async () => {
  for (const { scorer, score } of endToEndMetrics) {
    const actualScore = await scorer({
      input: data.input,
      output: data.output,
      expected: data.expected,
      context: data.context,
    });

    if (score === 1) {
      expect(actualScore.score).toBeCloseTo(score, 4);
    }
  }
}, 600000);
