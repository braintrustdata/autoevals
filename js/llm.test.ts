import { bypass, http, HttpResponse } from "msw";
import { setupServer } from "msw/node";
import { OpenAI } from "openai";
import { ChatCompletionMessageParam } from "openai/resources";
import { afterAll, afterEach, beforeAll, describe, expect, test } from "vitest";
import {
  Battle,
  buildClassificationTools,
  LLMClassifierFromTemplate,
  OpenAIClassifier,
} from "../js/llm";
import {
  openaiClassifierShouldEvaluateArithmeticExpressions,
  openaiClassifierShouldEvaluateTitles,
  openaiClassifierShouldEvaluateTitlesWithCoT,
} from "./llm.fixtures";
import { init } from "./oai";

export const server = setupServer();

beforeAll(() => {
  server.listen({
    onUnhandledRequest: (req) => {
      throw new Error(`Unhandled request ${req.method}, ${req.url}`);
    },
  });

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

describe("LLM Tests", () => {
  test("openai classifier should evaluate titles", async () => {
    let callCount = -1;
    server.use(
      http.post("https://api.openai.com/v1/chat/completions", async () => {
        const response = openaiClassifierShouldEvaluateTitles[++callCount];
        return response
          ? HttpResponse.json(response)
          : HttpResponse.json({}, { status: 500 });
      }),
    );

    const messages: ChatCompletionMessageParam[] = [
      {
        role: "system",
        content: `You are a technical project manager who helps software engineers generate better titles for their GitHub issues.
You will look at the issue description, and pick which of two titles better describes it.`,
      },
      {
        role: "user",
        content: `I'm going to provide you with the issue description, and two possible titles.

Issue Description: {{page_content}}

1: {{output}}
2: {{expected}}

Please discuss each title briefly (one line for pros, one for cons), and then answer the question by calling
the select_choice function with "1" or "2".`,
      },
    ];

    const page_content = `As suggested by Nicolo, we should standardize the error responses coming from GoTrue, postgres, and realtime (and any other/future APIs) so that it's better DX when writing a client,

We can make this change on the servers themselves, but since postgrest and gotrue are fully/partially external may be harder to change, it might be an option to transform the errors within the client libraries/supabase-js, could be messy?

Nicolo also dropped this as a reference: http://spec.openapis.org/oas/v3.0.3#openapi-specification`;

    const output = `Standardize error responses from GoTrue, Postgres, and Realtime APIs for better DX`;
    const expected = `Good title`;

    const score = await OpenAIClassifier({
      name: "titles",
      output,
      expected,
      messages,
      model: "gpt-3.5-turbo",
      parseScoreFn: (grade: string) => grade.match(/Winner: (\d+)/)![1],
      choiceScores: { "1": 1, "2": 0 },
      classificationTools: buildClassificationTools(true, ["1", "2"]),
      page_content,
      maxTokens: 500,
      openAiApiKey: "test-api-key",
    });

    expect(score.error).toBeUndefined();
  });

  test("llm classifier should evaluate with and without chain of thought", async () => {
    let callCount = -1;
    server.use(
      http.post(
        "https://api.openai.com/v1/chat/completions",
        async ({ request }) => {
          const response =
            openaiClassifierShouldEvaluateTitlesWithCoT[++callCount];

          if (!response) {
            const res = await fetch(bypass(request));
            const body = await res.json();
            return HttpResponse.json(body, {
              status: res.status,
              headers: res.headers,
            });
          }

          return response
            ? HttpResponse.json(response)
            : HttpResponse.json({}, { status: 500 });
        },
      ),
    );

    const pageContent = `As suggested by Nicolo, we should standardize the error responses coming from GoTrue, postgres, and realtime (and any other/future APIs) so that it's better DX when writing a client,

We can make this change on the servers themselves, but since postgrest and gotrue are fully/partially external may be harder to change, it might be an option to transform the errors within the client libraries/supabase-js, could be messy?

Nicolo also dropped this as a reference: http://spec.openapis.org/oas/v3.0.3#openapi-specification`;
    const genTitle = `Standardize error responses from GoTrue, Postgres, and Realtime APIs for better DX`;
    const originalTitle = `Good title`;

    for (const useCoT of [true, false]) {
      const classifier = LLMClassifierFromTemplate<{ page_content: string }>({
        name: "titles",
        promptTemplate: `You are a technical project manager who helps software engineers generate better titles for their GitHub issues.
You will look at the issue description, and pick which of two titles better describes it.

I'm going to provide you with the issue description, and two possible titles.

Issue Description: {{page_content}}

1: {{output}}
2: {{expected}}`,
        choiceScores: { "1": 1, "2": 0 },
        useCoT,
      });

      let response = await classifier({
        output: genTitle,
        expected: originalTitle,
        page_content: pageContent,
        openAiApiKey: "test-api-key",
      });

      expect(response.error).toBeUndefined();

      response = await classifier({
        output: originalTitle,
        expected: genTitle,
        page_content: pageContent,
        openAiApiKey: "test-api-key",
      });

      expect(response.error).toBeUndefined();
    }
  });

  test("battle should evaluate arithmetic expressions", async () => {
    let callCount = -1;
    server.use(
      http.post("https://api.openai.com/v1/chat/completions", async () => {
        const response =
          openaiClassifierShouldEvaluateArithmeticExpressions[++callCount];

        return response
          ? HttpResponse.json(response)
          : HttpResponse.json({}, { status: 500 });
      }),
    );

    // reset the client to test direct client usage
    init();

    const client = new OpenAI({
      apiKey: "test-api-key",
      baseURL: "https://api.openai.com/v1",
    });

    for (const useCoT of [true, false]) {
      let response = await Battle({
        useCoT,
        instructions: "Add the following numbers: 1, 2, 3",
        output: "600",
        expected: "6",
        client,
      });

      expect(response.error).toBeUndefined();

      response = await Battle({
        useCoT,
        instructions: "Add the following numbers: 1, 2, 3",
        output: "6",
        expected: "600",
        client,
      });

      expect(response.error).toBeUndefined();

      response = await Battle({
        useCoT,
        instructions: "Add the following numbers: 1, 2, 3",
        output: "6",
        expected: "6",
        client,
      });

      expect(response.error).toBeUndefined();
    }
  });
});
