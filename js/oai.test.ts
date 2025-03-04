import { buildOpenAIClient } from "./oai";
import { http, HttpResponse } from "msw";
import { server } from "./test/setup";
import OpenAI from "openai";

const MOCK_OPENAI_COMPLETION_RESPONSE = {
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
          return HttpResponse.json(MOCK_OPENAI_COMPLETION_RESPONSE);
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

  test("should use regular OpenAI", async () => {
    server.use(
      http.post("https://api.openai.com/v1/chat/completions", () => {
        return HttpResponse.json(MOCK_OPENAI_COMPLETION_RESPONSE);
      }),
    );

    const client = buildOpenAIClient({
      openAiApiKey: "test-api-key",
      openAiBaseUrl: "https://api.openai.com/v1",
    });

    const response = await client.chat.completions.create({
      model: "gpt-4",
      messages: [{ role: "user", content: "Hello" }],
    });

    expect(response.choices[0].message.content).toBe(
      "Hello, I am a mock response!",
    );
  });

  test("calls proxy if everything unset", async () => {
    server.use(
      http.post("https://api.braintrust.dev/v1/proxy/chat/completions", () => {
        return HttpResponse.json(MOCK_OPENAI_COMPLETION_RESPONSE);
      }),
    );

    const client = buildOpenAIClient({});
    const response = await client.chat.completions.create({
      model: "gpt-4",
      messages: [{ role: "user", content: "Hello" }],
    });

    expect(response.choices[0].message.content).toBe(
      "Hello, I am a mock response!",
    );
  });

  test("default wraps", async () => {
    server.use(
      http.post("https://api.braintrust.dev/v1/proxy/chat/completions", () => {
        return HttpResponse.json(MOCK_OPENAI_COMPLETION_RESPONSE);
      }),
    );

    await withMockWrapper(async ({ createSpy }) => {
      const client = buildOpenAIClient({});

      await client.chat.completions.create({
        model: "gpt-4",
        messages: [{ role: "user", content: "Hello" }],
      });

      expect(createSpy).toHaveBeenCalledTimes(1);
      expect(createSpy).toHaveBeenCalledWith({
        model: "gpt-4",
        messages: [{ role: "user", content: "Hello" }],
      });
    });
  });

  test("wraps once", async () => {
    server.use(
      http.post("https://api.braintrust.dev/v1/proxy/chat/completions", () => {
        return HttpResponse.json(MOCK_OPENAI_COMPLETION_RESPONSE);
      }),
    );

    await withMockWrapper(async ({ wrapperMock, createSpy }) => {
      const client = wrapperMock(
        new OpenAI({
          apiKey: "test-api-key",
        }),
      );
      const builtClient = buildOpenAIClient({ client });

      expect(builtClient).toBe(client);

      await builtClient.chat.completions.create({
        model: "gpt-4",
        messages: [{ role: "user", content: "Hello" }],
      });

      expect(createSpy).toHaveBeenCalledTimes(1);
      expect(createSpy).toHaveBeenCalledWith({
        model: "gpt-4",
        messages: [{ role: "user", content: "Hello" }],
      });
    });
  });

  test("wraps client, if possible", async () => {
    server.use(
      http.post("https://api.openai.com/v1/chat/completions", () => {
        return HttpResponse.json(MOCK_OPENAI_COMPLETION_RESPONSE);
      }),
    );

    await withMockWrapper(async ({ wrapperMock, createSpy }) => {
      const client = new OpenAI({ apiKey: "test-api-key" });
      const builtClient = buildOpenAIClient({ client });

      await builtClient.chat.completions.create({
        model: "gpt-4",
        messages: [{ role: "user", content: "Hello" }],
      });

      expect(createSpy).toHaveBeenCalledTimes(1);
      expect(createSpy).toHaveBeenCalledWith({
        model: "gpt-4",
        messages: [{ role: "user", content: "Hello" }],
      });
    });
  });
});

const withMockWrapper = async (
  fn: (args: {
    wrapperMock: (client: any) => any;
    createSpy: jest.Mock;
  }) => Promise<void>,
) => {
  const createSpy = jest.fn();
  const wrapperMock = (client: any) => {
    return new Proxy(client, {
      get(target, prop) {
        if (prop === "chat") {
          return new Proxy(
            {},
            {
              get(target, prop) {
                if (prop === "completions") {
                  return new Proxy(
                    {},
                    {
                      get(target, prop) {
                        if (prop === "create") {
                          return createSpy;
                        }
                        return Reflect.get(target, prop);
                      },
                    },
                  );
                }
                return Reflect.get(target, prop);
              },
            },
          );
        }
        return Reflect.get(target, prop);
      },
    });
  };

  const originalWrapper = globalThis.__inherited_braintrust_wrap_openai;
  try {
    globalThis.__inherited_braintrust_wrap_openai = wrapperMock;
    await fn({ wrapperMock, createSpy });
  } finally {
    globalThis.__inherited_braintrust_wrap_openai = originalWrapper;
  }
};
