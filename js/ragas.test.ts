import { http, HttpResponse } from "msw";
import { setupServer } from "msw/node";
import { OpenAI } from "openai";
import { afterAll, afterEach, beforeAll, describe, expect, test } from "vitest";
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
import { init } from "./oai";

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
      expect(actualScore.score).toBeLessThanOrEqual(1);
    }
  }
}, 600000);

// Tests for ContextRelevancy score clamping (#80)
describe("ContextRelevancy score clamping", () => {
  const server = setupServer();

  beforeAll(() => {
    server.listen({
      onUnhandledRequest: (req) => {
        throw new Error(`Unhandled request ${req.method}, ${req.url}`);
      },
    });
  });

  afterEach(() => {
    server.resetHandlers();
    init();
  });

  afterAll(() => {
    server.close();
  });

  test("clamps score to 1.0 when LLM returns sentences longer than context", async () => {
    // Mock response where extracted sentences are LONGER than the context
    // This would produce a raw score > 1.0 without clamping
    server.use(
      http.post("https://api.openai.com/v1/chat/completions", () => {
        return HttpResponse.json({
          id: "chatcmpl-test",
          object: "chat.completion",
          created: Date.now(),
          model: "gpt-5-mini",
          choices: [
            {
              index: 0,
              message: {
                role: "assistant",
                content: null,
                tool_calls: [
                  {
                    id: "call_test",
                    type: "function",
                    function: {
                      name: "extract_sentences",
                      arguments: JSON.stringify({
                        sentences: [
                          {
                            sentence:
                              "Hello world, this is a much longer sentence than the original context that was provided",
                            reasons: ["This is a test reason"],
                          },
                        ],
                      }),
                    },
                  },
                ],
              },
              finish_reason: "tool_calls",
            },
          ],
          usage: { prompt_tokens: 10, completion_tokens: 20, total_tokens: 30 },
        });
      }),
    );

    init({
      client: new OpenAI({
        apiKey: "test-api-key",
        baseURL: "https://api.openai.com/v1",
      }),
    });

    // Short context that would cause score > 1.0 without clamping
    const result = await ContextRelevancy({
      input: "What is hello?",
      output: "Hello world",
      context: "Hello world", // 11 chars, but mock returns 88 chars
    });

    // Score should be clamped to 1.0, not exceed it
    expect(result.score).toBe(1);
    expect(result.score).toBeLessThanOrEqual(1);
    expect(result.score).toBeGreaterThanOrEqual(0);
  });

  test("returns expected score for normal case", async () => {
    const context =
      "Hello world, this is a test context with some content that is reasonably long.";

    // Mock response where extracted sentences are shorter than the context
    server.use(
      http.post("https://api.openai.com/v1/chat/completions", () => {
        return HttpResponse.json({
          id: "chatcmpl-test",
          object: "chat.completion",
          created: Date.now(),
          model: "gpt-5-mini",
          choices: [
            {
              index: 0,
              message: {
                role: "assistant",
                content: null,
                tool_calls: [
                  {
                    id: "call_test",
                    type: "function",
                    function: {
                      name: "extract_sentences",
                      arguments: JSON.stringify({
                        sentences: [
                          { sentence: "Hello world", reasons: ["Test reason"] },
                        ],
                      }),
                    },
                  },
                ],
              },
              finish_reason: "tool_calls",
            },
          ],
          usage: { prompt_tokens: 10, completion_tokens: 20, total_tokens: 30 },
        });
      }),
    );

    init({
      client: new OpenAI({
        apiKey: "test-api-key",
        baseURL: "https://api.openai.com/v1",
      }),
    });

    const result = await ContextRelevancy({
      input: "What is hello?",
      output: "Hello world",
      context,
    });

    // Score should be len("Hello world") / len(context) = 11 / 79 ≈ 0.139
    const expectedScore = "Hello world".length / context.length;
    expect(result.score).toBeCloseTo(expectedScore, 2);
    expect(result.score).toBeLessThanOrEqual(1);
    expect(result.score).toBeGreaterThanOrEqual(0);
  });
});

describe("AnswerCorrectness custom embedding model", () => {
  const server = setupServer();

  beforeAll(() => {
    server.listen({
      onUnhandledRequest: (req) => {
        throw new Error(`Unhandled request ${req.method}, ${req.url}`);
      },
    });
  });

  afterEach(() => {
    server.resetHandlers();
    init();
  });

  afterAll(() => {
    server.close();
  });

  test("AnswerCorrectness uses custom embedding model", async () => {
    let capturedEmbeddingModel: string | undefined;

    server.use(
      http.post("https://api.openai.com/v1/chat/completions", async () => {
        return HttpResponse.json({
          id: "test-id",
          object: "chat.completion",
          created: Date.now(),
          model: "gpt-5-mini",
          choices: [
            {
              index: 0,
              message: {
                role: "assistant",
                tool_calls: [
                  {
                    id: "call_test",
                    type: "function",
                    function: {
                      name: "classify_statements",
                      arguments: JSON.stringify({
                        TP: ["Paris is the capital"],
                        FP: [],
                        FN: [],
                      }),
                    },
                  },
                ],
              },
              finish_reason: "tool_calls",
            },
          ],
        });
      }),
      http.post("https://api.openai.com/v1/embeddings", async ({ request }) => {
        const body = (await request.json()) as { model: string; input: string };
        capturedEmbeddingModel = body.model;
        return HttpResponse.json({
          object: "list",
          data: [
            {
              object: "embedding",
              embedding: new Array(1536).fill(0.1),
              index: 0,
            },
          ],
          model: body.model,
          usage: {
            prompt_tokens: 5,
            total_tokens: 5,
          },
        });
      }),
    );

    init({
      client: new OpenAI({
        apiKey: "test-api-key",
        baseURL: "https://api.openai.com/v1",
      }),
    });

    await AnswerCorrectness({
      input: "What is the capital of France?",
      output: "Paris",
      expected: "Paris is the capital of France",
      embeddingModel: "text-embedding-3-large",
    });

    expect(capturedEmbeddingModel).toBe("text-embedding-3-large");
  });
});
