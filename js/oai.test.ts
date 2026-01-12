import { http, HttpResponse } from "msw";
import OpenAI from "openai";
import {
  afterAll,
  afterEach,
  beforeAll,
  beforeEach,
  describe,
  expect,
  test,
  vi,
} from "vitest";
import { buildOpenAIClient, init, getDefaultModel } from "./oai";

import { setupServer } from "msw/node";

export const server = setupServer();

beforeAll(() => {
  server.listen({
    onUnhandledRequest: (req) => {
      throw new Error(`Unhandled request ${req.method}, ${req.url}`);
    },
  });
});

let OPENAI_API_KEY: string | undefined;
let OPENAI_BASE_URL: string | undefined;

beforeEach(() => {
  OPENAI_API_KEY = process.env.OPENAI_API_KEY;
  OPENAI_BASE_URL = process.env.OPENAI_BASE_URL;
});

afterEach(() => {
  server.resetHandlers();

  process.env.OPENAI_API_KEY = OPENAI_API_KEY;
  process.env.OPENAI_BASE_URL = OPENAI_BASE_URL;

  // Reset init state
  init({ client: undefined, defaultModel: undefined });
});

afterAll(() => {
  server.close();
});

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
    delete process.env.OPENAI_API_KEY;
    delete process.env.OPENAI_BASE_URL;

    server.use(
      http.post("https://api.braintrust.dev/v1/proxy/chat/completions", () => {
        debugger;
        return HttpResponse.json(MOCK_OPENAI_COMPLETION_RESPONSE);
      }),
    );

    const client = buildOpenAIClient({});
    const response = await client.chat.completions.create({
      model: "gpt-4",
      messages: [{ role: "user", content: "Hello" }],
    });

    debugger;

    expect(response.choices[0].message.content).toBe(
      "Hello, I am a mock response!",
    );
  });

  test("default wraps", async () => {
    delete process.env.OPENAI_API_KEY;
    delete process.env.OPENAI_BASE_URL;

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
    delete process.env.OPENAI_API_KEY;
    delete process.env.OPENAI_BASE_URL;

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

  test("init sets client", async () => {
    server.use(
      http.post("https://api.openai.com/v1/chat/completions", () => {
        return HttpResponse.json(MOCK_OPENAI_COMPLETION_RESPONSE);
      }),
    );

    const client = new OpenAI({ apiKey: "test-api-key" });

    init({ client });

    const builtClient = buildOpenAIClient({});

    expect(Object.is(builtClient, client)).toBe(true);
  });

  test("client wins against init", async () => {
    server.use(
      http.post("https://api.openai.com/v1/chat/completions", () => {
        return HttpResponse.json(MOCK_OPENAI_COMPLETION_RESPONSE);
      }),
    );

    const client = new OpenAI({ apiKey: "test-api-key" });

    init({ client });

    const otherClient = new OpenAI({ apiKey: "other-api-key" });

    const builtClient = buildOpenAIClient({ client: otherClient });

    expect(Object.is(builtClient, otherClient)).toBe(true);
  });

  test("getDefaultModel returns gpt-4o by default", () => {
    expect(getDefaultModel()).toBe("gpt-4o");
  });

  test("init sets default model", () => {
    init({ defaultModel: "claude-3-5-sonnet-20241022" });
    expect(getDefaultModel()).toBe("claude-3-5-sonnet-20241022");
  });

  test("init can reset default model", () => {
    init({ defaultModel: "claude-3-5-sonnet-20241022" });
    expect(getDefaultModel()).toBe("claude-3-5-sonnet-20241022");

    init({ defaultModel: undefined });
    expect(getDefaultModel()).toBe("gpt-4o");
  });

  test("init can set both client and default model", () => {
    const client = new OpenAI({ apiKey: "test-api-key" });
    init({ client, defaultModel: "gpt-4-turbo" });

    const builtClient = buildOpenAIClient({});
    expect(Object.is(builtClient, client)).toBe(true);
    expect(getDefaultModel()).toBe("gpt-4-turbo");
  });
});

const withMockWrapper = async (
  fn: (args: {
    wrapperMock: (client: any) => any;
    createSpy: ReturnType<typeof vi.fn>;
  }) => Promise<void>,
) => {
  const createSpy = vi.fn();
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
