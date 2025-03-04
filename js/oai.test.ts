import { buildOpenAIClient } from "./oai";
import { http, HttpResponse } from "msw";
import { server } from "./test/setup";

const mockOpenAIResponse = {
  choices: [
    {
      message: {
        content: "Hello, I am a mock response!",
        role: "assistant",
      },
      finish_reason: "stop",
      index: 0,
    },
  ],
  created: Date.now(),
  id: "mock-id",
  model: "mock-model",
  object: "chat.completion",
  usage: {
    completion_tokens: 9,
    prompt_tokens: 5,
    total_tokens: 14,
  },
};

describe("OAI", () => {
  test("should use Azure OpenAI", async () => {
    server.use(
      http.post(
        "https://*.openai.azure.com/openai/deployments/*/chat/completions*",
        () => {
          return HttpResponse.json(mockOpenAIResponse);
        },
      ),
    );

    const client = buildOpenAIClient({
      azureOpenAi: {
        apiKey: "test-api-key",
        endpoint: "https://test-resource.openai.azure.com",
        apiVersion: "2024-02-15-preview",
      },
    });

    const response = await client.chat.completions.create({
      model: "test-model",
      messages: [{ role: "system", content: "Hello" }],
    });

    expect(response.choices[0].message.content).toBe(
      "Hello, I am a mock response!",
    );
    expect(response.choices).toHaveLength(1);
  });

  test("should handle Azure OpenAI error responses", async () => {
    server.use(
      http.post(
        "https://*.openai.azure.com/openai/deployments/*/chat/completions*",
        () => {
          return new HttpResponse(null, {
            status: 401,
            statusText: "Unauthorized",
          });
        },
      ),
    );

    const client = buildOpenAIClient({
      azureOpenAi: {
        apiKey: "invalid-api-key",
        endpoint: "https://test-resource.openai.azure.com",
        apiVersion: "2024-02-15-preview",
      },
    });

    await expect(
      client.chat.completions.create({
        model: "test-model",
        messages: [{ role: "system", content: "Hello" }],
      }),
    ).rejects.toThrow();
  });
});
