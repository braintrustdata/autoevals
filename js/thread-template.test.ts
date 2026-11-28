import { http, HttpResponse } from "msw";
import { setupServer } from "msw/node";
import { OpenAI } from "openai";
import { afterAll, afterEach, beforeAll, expect, test } from "vitest";
import { LLMClassifierFromTemplate } from "./llm";

const server = setupServer();
beforeAll(() => server.listen({ onUnhandledRequest: "error" }));
afterEach(() => server.resetHandlers());
afterAll(() => server.close());

test.each([
  [
    "{{#user_messages}}{{content}}{{/user_messages}}",
    [{ role: "user", content: "Hello" }],
    "Hello",
    1,
  ],
  [
    "{{# user_messages }}{{content}}{{/ user_messages }}",
    [{ role: "user", content: "Hello" }],
    "Hello",
    1,
  ],
  [
    "{{^user_messages}}No users{{/user_messages}}",
    [{ role: "user", content: "Hello" }],
    "",
    1,
  ],
  ["{{^user_messages}}No users{{/user_messages}}", [], "No users", 1],
  [
    "{{&first_message.content}}",
    [{ role: "user", content: "Hello" }],
    "Hello",
    1,
  ],
  [
    "{{first_message.content}}",
    [{ role: "user", content: "Hello" }],
    "Hello",
    1,
  ],
  [
    "{{{first_message.content}}}",
    [{ role: "user", content: "Hello" }],
    "Hello",
    1,
  ],
  ["Output: {{output}}", [], "Output: answer", 0],
  ["{{!user_messages}}No conversation", [], "No conversation", 0],
  [
    "{{#user_messages}}{{content}}{{/user_messages}}",
    [{ role: "user", content: "Hello" }],
    "Override",
    null,
    "Override",
  ],
] as const)(
  "fetches and renders trace for %s",
  async (template, thread, rendered, expectedCalls, override?: string) => {
    let prompt: unknown;
    let calls = 0;
    server.use(
      http.post("https://api.openai.com/v1/responses", async ({ request }) => {
        const body = (await request.json()) as {
          input: { content: unknown }[];
        };
        prompt = body.input[0].content;
        return HttpResponse.json({
          id: "resp-test",
          object: "response",
          created: 1234567890,
          model: "gpt-5-mini",
          output: [
            {
              type: "function_call",
              call_id: "call-test",
              name: "select_choice",
              arguments: JSON.stringify({ choice: "A" }),
            },
          ],
        });
      }),
    );
    const classifier = LLMClassifierFromTemplate<{
      user_messages?: { role: string; content: string }[];
    }>({
      name: "thread-tags",
      promptTemplate: "BEGIN" + template + "END",
      choiceScores: { A: 1 },
      model: "gpt-5-mini",
      useCoT: false,
    });
    const score = await classifier({
      output: "answer",
      client: new OpenAI({
        apiKey: "test-api-key",
        baseURL: "https://api.openai.com/v1",
      }),
      ...(override === undefined
        ? {}
        : { user_messages: [{ role: "user", content: override }] }),
      trace: {
        async getThread() {
          calls += 1;
          return [...thread];
        },
      },
    });
    expect(score.score).toBe(1);
    if (expectedCalls !== null) {
      expect(calls).toBe(expectedCalls);
    }
    expect(prompt).toBe(
      "BEGIN" +
        rendered +
        "END\n" +
        'Answer the question by calling `select_choice` with a single choice from ["A"].',
    );
  },
);
