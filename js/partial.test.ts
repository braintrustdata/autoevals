import { http, HttpResponse } from "msw";
import { setupServer } from "msw/node";
import { OpenAI } from "openai";
import { afterAll, afterEach, beforeAll, expect, test } from "vitest";
import { ClosedQA } from "./llm";
import { init } from "./oai";
import { Levenshtein } from "./string";

const server = setupServer();

beforeAll(() => {
  server.listen({
    onUnhandledRequest: (req) => {
      throw new Error(`Unhandled request ${req.method}, ${req.url}`);
    },
  });

  server.use(
    http.post("https://api.openai.com/v1/responses", async () => {
      return HttpResponse.json({
        id: "resp-test",
        object: "response",
        created: Math.floor(Date.now() / 1000),
        model: "gpt-5-mini",
        output: [
          {
            type: "function_call",
            call_id: "call_test",
            name: "select_choice",
            arguments: JSON.stringify({ choice: "Y" }),
          },
        ],
      });
    }),
  );

  init({
    client: new OpenAI({
      apiKey: "test-api-key",
      baseURL: "https://api.openai.com/v1",
    }),
  });
});

afterEach(() => {
  server.resetHandlers();
});

afterAll(() => {
  server.close();
  init();
});

test("Partial Test", async () => {
  const levenshteinBasic = await Levenshtein({
    output: "abc",
    expected: "abcd",
  });
  const levenshteinPartial = await Levenshtein.partial({ expected: "abcd" })({
    output: "abc",
  });
  expect(levenshteinBasic.score).toBeDefined();
  expect(levenshteinPartial.score).toBeDefined();
  expect(levenshteinPartial.score).toEqual(levenshteinBasic.score);
  expect(levenshteinBasic.name).toEqual(levenshteinPartial.name);
  expect(levenshteinBasic.name).toEqual("Levenshtein");

  // Now do the same with ClosedQA which is an "LLM" scorer
  const closedQABasic = await ClosedQA({
    criteria: "Is the answer correct?",
    input: "What is 1+1?",
    output: "2",
  });
  const closedQAPartial = await ClosedQA.partial({
    criteria: "Is the answer correct?",
  })({
    input: "What is 1+1?",
    output: "2",
  });
  expect(closedQABasic.score).toBeDefined();
  expect(closedQAPartial.score).toBeDefined();
  expect(closedQAPartial.score).toEqual(closedQABasic.score);
  expect(closedQABasic.name).toEqual(closedQAPartial.name);
  expect(closedQABasic.name).toEqual("ClosedQA");
});
