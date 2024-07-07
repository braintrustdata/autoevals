import { ChatCompletionMessageParam } from "openai/resources";
import {
  Battle,
  LLMClassifierFromTemplate,
  OpenAIClassifier,
  buildClassificationTools,
} from "../js/llm";
import { ChatCache } from "../js/oai";

let cache: ChatCache | undefined;

beforeAll(() => {
  cache = undefined;
});

test("openai", async () => {
  const parseBestTitle = (grade: string) => {
    return grade.match(/Winner: (\d+)/)![1];
  };

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
    parseScoreFn: parseBestTitle,
    choiceScores: { "1": 1, "2": 0 },
    classificationTools: buildClassificationTools(true, ["1", "2"]),
    page_content,
    maxTokens: 500,
    cache,
    openAiApiKey: process.env.OPENAI_API_KEY!,
  });

  expect(score.score).toBe(1);
  expect(score.error).toBeUndefined();
}, 600000);

test("llm_classifier", async () => {
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
      openAiApiKey: process.env.OPENAI_API_KEY!,
    });

    expect(response.score).toBe(1);
    expect(response.error).toBeUndefined();

    response = await classifier({
      output: originalTitle,
      expected: genTitle,
      page_content: pageContent,
      openAiApiKey: process.env.OPENAI_API_KEY!,
    });

    expect(response.score).toBe(0);
    expect(response.error).toBeUndefined();
  }
}, 600000);

test("battle", async () => {
  for (const useCoT of [true, false]) {
    console.log("useCoT", useCoT);
    let response = await Battle({
      useCoT,
      instructions: "Add the following numbers: 1, 2, 3",
      output: "600",
      expected: "6",
      openAiApiKey: process.env.OPENAI_API_KEY!,
    });

    expect(response.score).toBe(0);
    expect(response.error).toBeUndefined();

    response = await Battle({
      useCoT,
      instructions: "Add the following numbers: 1, 2, 3",
      output: "6",
      expected: "600",
      openAiApiKey: process.env.OPENAI_API_KEY!,
    });

    expect(response.score).toBe(useCoT ? 1 : 0);
    expect(response.error).toBeUndefined();

    response = await Battle({
      useCoT,
      instructions: "Add the following numbers: 1, 2, 3",
      output: "6",
      expected: "6",
      openAiApiKey: process.env.OPENAI_API_KEY!,
    });

    expect(response.score).toBe(0);
    expect(response.error).toBeUndefined();
  }
}, 600000);
