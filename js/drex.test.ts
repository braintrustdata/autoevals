// Offline tests run against a fake Drex (msw). Live tests call the real API and run only when
// NACE_API_KEY is set: `NACE_API_KEY=nace_sk_... pnpm run test -- js/drex.test.ts -t live`.
import { deflateSync } from "node:zlib";
import { delay, http, HttpResponse } from "msw";
import { setupServer } from "msw/node";
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
import {
  buildState,
  DREX_CLIENT_NAME,
  DrexClassifier,
  DrexClassifierMany,
  DrexError,
  drexQuestion,
  isTextOnly,
} from "./drex";

const BASE = "https://drex.test";
const URL_ = `${BASE}/v1/systemone`;

function crc32(buf: Uint8Array): number {
  let c = ~0;
  for (const byte of buf) {
    c ^= byte;
    for (let k = 0; k < 8; k++) c = c & 1 ? (c >>> 1) ^ 0xedb88320 : c >>> 1;
  }
  return ~c >>> 0;
}

function png(rgb: [number, number, number]): Uint8Array {
  const chunk = (kind: string, data: Uint8Array) => {
    const body = Buffer.concat([Buffer.from(kind, "latin1"), data]);
    const out = Buffer.alloc(12 + data.length);
    out.writeUInt32BE(data.length, 0);
    body.copy(out, 4);
    out.writeUInt32BE(crc32(body), 8 + data.length);
    return out;
  };
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(2, 0);
  ihdr.writeUInt32BE(2, 4);
  ihdr[8] = 8;
  ihdr[9] = 2;
  const row = Buffer.concat([Buffer.from([0]), Buffer.from([...rgb, ...rgb])]);
  return new Uint8Array(
    Buffer.concat([
      Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
      chunk("IHDR", ihdr),
      chunk("IDAT", deflateSync(Buffer.concat([row, row]))),
      chunk("IEND", Buffer.alloc(0)),
    ]),
  );
}

const RED = png([255, 0, 0]);
const BLUE = png([0, 0, 255]);

function body(
  answers: Record<string, unknown>,
  extra: { model?: string; input_tokens?: number } = {},
) {
  return {
    model: extra.model ?? "drex-max",
    answers,
    usage: {
      input_tokens: extra.input_tokens ?? 120,
      image_tokens: 0,
      output_tokens: 0,
    },
    evaluation_time_ms: 41.5,
    request_id: "req_test123",
  };
}

const NOUL = { type: "noul", noul: 0.9 };
const CHOICE = {
  type: "choice",
  choice: "good",
  confidence: 0.8,
  probabilities: { good: 0.7, ok: 0.2, bad: 0.1 },
};
const SCORE = {
  type: "score",
  score: 3,
  confidence: 0.7,
  legend: ["awful", "poor", "fine", "great", "perfect"],
  probabilities: { "2": 0.2, "3": 0.6, "4": 0.2 },
};

const YES_NO = drexQuestion.noul("Is the output correct?");
const CHOICES = drexQuestion.choice(
  { good: "Fully right", ok: null, bad: "Wrong" },
  { scores: { good: 1, ok: 0.5, bad: 0 } },
);
const LEVELS: [string, ...string[]] = [
  "awful",
  "poor",
  "fine",
  "great",
  "perfect",
];
const RUBRIC = drexQuestion.score(LEVELS, { instructions: "Rate the answer." });

const server = setupServer();
let requests: { body: any; headers: Headers }[] = [];

beforeAll(() =>
  server.listen({
    onUnhandledRequest: (req) => {
      throw new Error(`Unhandled request ${req.method}, ${req.url}`);
    },
  }),
);
afterEach(() => {
  server.resetHandlers();
  requests = [];
});
afterAll(() => server.close());

function respond(
  json: unknown,
  status = 200,
  headers: Record<string, string> = {},
  url = URL_,
) {
  server.use(
    http.post(url, async ({ request }) => {
      requests.push({ body: await request.json(), headers: request.headers });
      return HttpResponse.json(json as any, { status, headers });
    }),
  );
}

function respondSequence(
  responses: {
    status: number;
    json: unknown;
    headers?: Record<string, string>;
  }[],
) {
  let calls = 0;
  server.use(
    http.post(URL_, async ({ request }) => {
      requests.push({ body: await request.json(), headers: request.headers });
      const next = responses[Math.min(calls++, responses.length - 1)];
      return HttpResponse.json(next.json as any, {
        status: next.status,
        headers: next.headers,
      });
    }),
  );
}

const auth = { apiKey: "nace_sk_test", baseUrl: BASE, maxRetries: 2 };
const score = async (args: Record<string, unknown>) =>
  DrexClassifier({ ...auth, ...args } as any);

describe("DrexClassifier", () => {
  for (const model of ["drex-max", "drex-v2.0"]) {
    test(`noul, choice and score on ${model}`, async () => {
      respond(body({ DrexClassifier: NOUL }, { model }));
      let result = await score({ question: YES_NO, output: "Paris", model });
      expect(result.score).toBeCloseTo(0.9);
      expect(requests[0].body.model).toBe(model);
      expect((result.metadata as any).probabilities.yes).toBe(0.9);
      expect((result.metadata as any).request_id).toBe("req_test123");
      expect((result.metadata as any).model).toBe(model);

      respond(body({ DrexClassifier: CHOICE }, { model }));
      result = await score({ question: CHOICES, output: "Paris", model });
      expect(result.score).toBeCloseTo(0.8); // 0.7 * 1 + 0.2 * 0.5 + 0.1 * 0
      expect((result.metadata as any).choice).toBe("good");
      expect((result.metadata as any).probabilities).toEqual({
        good: 0.7,
        ok: 0.2,
        bad: 0.1,
      });

      respond(body({ DrexClassifier: SCORE }, { model }));
      result = await score({ question: RUBRIC, output: "Paris", model });
      expect(result.score).toBeCloseTo(0.75); // 3 / 4
      expect((result.metadata as any).raw_score).toBe(3);
      expect((result.metadata as any).scale).toEqual([0, 4]);
      expect((result.metadata as any).legend[3]).toBe("great");
      expect((result.metadata as any).probabilities["3"]).toBe(0.6);
    });
  }

  test("defaults to drex-max and passes unknown model ids through", async () => {
    respond(body({ DrexClassifier: NOUL }));
    await score({ question: YES_NO, output: "x" });
    expect(requests[0].body.model).toBe("drex-max");
    await score({ question: YES_NO, output: "x", model: "drex-v9" });
    expect(requests[1].body.model).toBe("drex-v9");
  });

  test("a mixed batch is one request and counts usage once", async () => {
    respond(
      body(
        { correct: NOUL, quality: CHOICE, depth: SCORE },
        { input_tokens: 300 },
      ),
    );
    const scores = await DrexClassifierMany({
      ...auth,
      output: "Paris",
      input: "Capital of France?",
      questions: { correct: YES_NO, quality: CHOICES, depth: RUBRIC },
    });
    expect(requests).toHaveLength(1);
    expect(scores.map((s) => s.name)).toEqual(["correct", "quality", "depth"]);
    expect(Object.keys(requests[0].body.questions)).toEqual([
      "correct",
      "quality",
      "depth",
    ]);
    expect((scores[0].metadata as any).usage.input_tokens).toBe(300);
    expect((scores[1].metadata as any).usage).toBeNull();
    expect((scores[2].metadata as any).usage).toBeNull();
    expect((scores[1].metadata as any).usage_counted_on).toBe("correct");
  });

  test("a criteria-only noul sends empty instructions", async () => {
    respond(body({ DrexClassifier: NOUL }));
    await score({
      question: drexQuestion.noul(undefined, {
        true: "grounded",
        false: "invented",
      }),
      output: "x",
    });
    const question = requests[0].body.questions.DrexClassifier;
    expect(question.instructions).toBe("");
    expect(question.criteria).toEqual({ true: "grounded", false: "invented" });
  });

  test("a choice option with a null description is kept", async () => {
    respond(body({ DrexClassifier: CHOICE }));
    await score({ question: CHOICES, output: "x" });
    expect(requests[0].body.questions.DrexClassifier.criteria.ok).toBeNull();
  });

  test("state is built from the case fields", async () => {
    respond(body({ DrexClassifier: NOUL }));
    await score({
      question: YES_NO,
      output: "answer",
      expected: "reference",
      input: "question",
      context: "docs",
    });
    expect(requests[0].body.state).toEqual({
      input: "question",
      output: "answer",
      expected: "reference",
      context: "docs",
    });
    await score({
      question: YES_NO,
      output: "answer",
      expected: "reference",
      context: "docs",
      stateFields: ["output", "context"],
    });
    expect(requests[1].body.state).toEqual({
      output: "answer",
      context: "docs",
    });
    await score({ question: YES_NO, output: "answer" });
    expect(requests[2].body.state).toBe("answer");
    expect(buildState({ output: { a: 1 } })).toEqual({ output: { a: 1 } });
  });

  test("direction, reverse and threshold", async () => {
    respond(body({ DrexClassifier: NOUL }));
    const defect = drexQuestion.noul("Does the output contain a defect?", {
      yesIs: "fail",
    });
    let result = await score({ question: defect, output: "x" });
    expect(result.score).toBeCloseTo(0.1);
    expect((result.metadata as any).yes_is).toBe("fail");
    expect((result.metadata as any).probabilities.yes).toBe(0.9); // the answer itself is not flipped

    respond(body({ DrexClassifier: SCORE }));
    result = await score({
      question: drexQuestion.score(LEVELS, { reverse: true }),
      output: "x",
    });
    expect(result.score).toBeCloseTo(0.25);

    respond(body({ DrexClassifier: NOUL }));
    result = await score({ question: YES_NO, output: "x", threshold: 0.95 });
    expect(result.score).toBe(0);
    expect((result.metadata as any).raw_value).toBeCloseTo(0.9);
    expect(
      (await score({ question: YES_NO, output: "x", threshold: 0.5 })).score,
    ).toBe(1);
  });

  test("attribution headers", async () => {
    respond(body({ DrexClassifier: NOUL }));
    await score({ question: YES_NO, output: "x" });
    const { headers } = requests[0];
    expect(headers.get("authorization")).toBe("Bearer nace_sk_test");
    expect(headers.get("x-nace-client")).toBe(DREX_CLIENT_NAME);
    expect(headers.get("user-agent")).toBe(DREX_CLIENT_NAME);
  });

  test("one image, two images and reversed order", async () => {
    respond(body({ DrexClassifier: NOUL }));
    const q = drexQuestion.noul("Is Image 1 red?");
    await score({ question: q, output: "Look.", images: [RED] });
    expect(requests[0].body.images).toHaveLength(1);
    await score({ question: q, output: "Look.", images: [RED, BLUE] });
    const forward = requests[1].body.images;
    await score({ question: q, output: "Look.", images: [BLUE, RED] });
    const reverse = requests[2].body.images;
    expect(forward).toHaveLength(2);
    expect(forward).toEqual([...reverse].reverse());
    expect(forward[0].startsWith("data:image/png;base64,")).toBe(true);
    expect(
      Buffer.from(forward[0].split(",")[1], "base64").equals(Buffer.from(RED)),
    ).toBe(true);
  });

  test("images on a text-only model are refused before sending", async () => {
    respond(body({ DrexClassifier: NOUL }));
    for (const model of [
      "drex-v2.0",
      "drex-v1.5",
      "drex-latest",
      "nace-ai/drex-v2.0",
    ]) {
      await expect(
        score({ question: YES_NO, output: "x", images: [RED], model }),
      ).rejects.toThrow(/reads text only/);
    }
    expect(requests).toHaveLength(0);
    expect(isTextOnly("drex-v2.0")).toBe(true);
    expect(isTextOnly("drex-max")).toBe(false);
  });

  test("remote image URLs are refused", async () => {
    await expect(
      score({
        question: YES_NO,
        output: "x",
        images: ["https://example.com/a.png"],
      }),
    ).rejects.toThrow(/doesn't fetch image URLs/);
  });

  test("a workspace_file reference passes through", async () => {
    respond(body({ DrexClassifier: NOUL }));
    const ref = {
      type: "workspace_file",
      workspace_id: "ws_1",
      file_id: "file_1",
    };
    await score({ question: YES_NO, output: "x", images: [ref] });
    expect(requests[0].body.images).toEqual([ref]);
  });

  test("an uploaded file becomes a workspace_file reference", async () => {
    respond(body({ DrexClassifier: NOUL }));
    await score({
      question: YES_NO,
      output: "x",
      images: [{ workspace_id: "ws_1", file_id: "file_1" }],
    });
    expect(requests[0].body.images).toEqual([
      expect.objectContaining({
        type: "workspace_file",
        workspace_id: "ws_1",
        file_id: "file_1",
      }),
    ]);
  });

  test("more than 16 questions on Max surfaces Drex's 422", async () => {
    respond(
      {
        error: {
          type: "invalid_request_error",
          message: "invalid request",
          issues: [
            {
              path: "questions",
              message: "at most 16 questions allowed for drex-max",
            },
          ],
        },
      },
      422,
    );
    const questions = Object.fromEntries(
      Array.from({ length: 17 }, (_, i) => [
        `q${i}`,
        drexQuestion.noul(`Question ${i}?`),
      ]),
    );
    const error = await DrexClassifierMany({
      ...auth,
      output: "x",
      questions,
    }).catch((e) => e);
    expect(error).toBeInstanceOf(DrexError);
    expect(error.status).toBe(422);
    expect(error.message).toContain(
      "questions: at most 16 questions allowed for drex-max",
    );
  });

  test("401 and 402 messages", async () => {
    respond(
      { error: { type: "authentication_error", message: "Invalid API key" } },
      401,
    );
    let error = await score({ question: YES_NO, output: "x" }).catch((e) => e);
    expect(error).toBeInstanceOf(DrexError);
    expect(error.status).toBe(401);

    respond(
      { error: { type: "insufficient_credit", message: "No available funds" } },
      402,
    );
    error = await score({ question: YES_NO, output: "x" }).catch((e) => e);
    expect(error.message).toContain(`${BASE}/dashboard/billing`);
  });

  for (const status of [429, 529]) {
    test(`${status} is retried after retry-after-ms`, async () => {
      respondSequence([
        {
          status,
          headers: { "retry-after-ms": "5" },
          json: { error: { type: "rate_limit_error", message: "slow down" } },
        },
        { status: 200, json: body({ DrexClassifier: NOUL }) },
      ]);
      const started = Date.now();
      const result = await score({ question: YES_NO, output: "x" });
      expect(requests).toHaveLength(2);
      expect(result.score).toBeCloseTo(0.9);
      // The header's 5 ms, not the SDK's own backoff (at least 375 ms).
      expect(Date.now() - started).toBeLessThan(300);
    });
  }

  test("a huge retry-after-ms is not slept: the error is raised at once", async () => {
    respondSequence([
      {
        status: 529,
        headers: { "retry-after-ms": "86400000" },
        json: { error: { type: "overloaded_error", message: "busy" } },
      },
    ]);
    const started = Date.now();
    const error = await score({
      question: YES_NO,
      output: "x",
      maxRetries: 1,
    }).catch((e) => e);
    expect(error).toBeInstanceOf(DrexError);
    expect(error.status).toBe(529);
    expect(requests).toHaveLength(1);
    expect(Date.now() - started).toBeLessThan(1000);
  });

  test("exhausted retries are an error, not a score", async () => {
    respondSequence([
      {
        status: 529,
        headers: { "retry-after-ms": "1" },
        json: { error: { type: "overloaded_error", message: "busy" } },
      },
    ]);
    const error = await score({
      question: YES_NO,
      output: "x",
      maxRetries: 1,
    }).catch((e) => e);
    expect(error).toBeInstanceOf(DrexError);
    expect(error.status).toBe(529);
    expect(requests).toHaveLength(2);
  });

  test("OpenRouter payload and headers", async () => {
    respond(
      body({ DrexClassifier: NOUL }),
      200,
      {},
      "https://openrouter.ai/api/v1/systemone",
    );
    await score({
      question: drexQuestion.noul(undefined, { true: "red" }),
      output: "Is Image 1 red?",
      images: [RED],
      model: "nace-ai/drex-max",
      baseUrl: "https://openrouter.ai/api/v1",
    });
    const { body: payload, headers } = requests[0];
    expect(payload.images).toBeUndefined();
    expect(payload.state[0]).toEqual({ type: "text", text: "Is Image 1 red?" });
    expect(payload.state[1].type).toBe("image_url");
    expect(
      payload.state[1].image_url.url.startsWith("data:image/png;base64,"),
    ).toBe(true);
    expect(payload.questions.DrexClassifier.instructions).toBe("");
    expect(headers.get("http-referer")).toBe("https://nace.ai");
    expect(headers.get("x-openrouter-title")).toBe("Drex for Autoevals");
    expect(headers.get("x-nace-client")).toBe(DREX_CLIENT_NAME);
  });

  test("OpenRouter refuses workspace files and links its own credits", async () => {
    const baseUrl = "https://openrouter.ai/api/v1";
    const model = "nace-ai/drex-max";
    await expect(
      score({
        question: YES_NO,
        output: "x",
        baseUrl,
        model,
        images: [
          { type: "workspace_file", workspace_id: "ws_1", file_id: "f" },
        ],
      }),
    ).rejects.toThrow(/inline images only/);
    respond(
      { error: { type: "insufficient_credit", message: "No credits" } },
      402,
      {},
      "https://openrouter.ai/api/v1/systemone",
    );
    const error = await score({
      question: YES_NO,
      output: "x",
      baseUrl,
      model,
    }).catch((e) => e);
    expect(error.message).toContain("openrouter.ai/settings/credits");
    expect(error.message).not.toContain("/dashboard/billing");
  });

  test("inline base64 images get their MIME type on OpenRouter", async () => {
    respond(
      body({ DrexClassifier: NOUL }),
      200,
      {},
      "https://openrouter.ai/api/v1/systemone",
    );
    const b64 = (bytes: number[]) =>
      Buffer.from([...bytes, ...new Array(20).fill(0)]).toString("base64");
    await score({
      question: YES_NO,
      output: "x",
      baseUrl: "https://openrouter.ai/api/v1",
      model: "nace-ai/drex-max",
      images: [
        b64([0xff, 0xd8, 0xff, 0xe0]),
        b64([0x52, 0x49, 0x46, 0x46, 0, 0, 0, 0, 0x57, 0x45, 0x42, 0x50]),
        b64([0, 0, 0, 0]),
      ],
    });
    const urls = requests[0].body.state
      .slice(1)
      .map((part: any) => part.image_url.url.split(";")[0]);
    expect(urls).toEqual([
      "data:image/jpeg",
      "data:image/webp",
      "data:application/octet-stream",
    ]);
  });

  test("a timeout is an error, not a score", async () => {
    server.use(
      http.post(URL_, async () => {
        await delay(300);
        return HttpResponse.json(body({ DrexClassifier: NOUL }));
      }),
    );
    const error = await score({
      question: YES_NO,
      output: "x",
      timeoutMs: 50,
      maxRetries: 0,
    }).catch((e) => e);
    expect(error).toBeInstanceOf(DrexError);
    expect(error.status).toBeUndefined();
  });

  test("OpenRouter is detected from NACE_BASE_URL", async () => {
    respond(
      body({ DrexClassifier: NOUL }),
      200,
      {},
      "https://openrouter.ai/api/v1/systemone",
    );
    const saved = process.env.NACE_BASE_URL;
    process.env.NACE_BASE_URL = "https://openrouter.ai/api/v1";
    try {
      await score({
        question: YES_NO,
        output: "x",
        baseUrl: undefined,
        apiKey: "nace_sk_test",
      });
    } finally {
      if (saved === undefined) delete process.env.NACE_BASE_URL;
      else process.env.NACE_BASE_URL = saved;
    }
    expect(requests[0].headers.get("x-openrouter-title")).toBe(
      "Drex for Autoevals",
    );
  });

  test("an empty output string is scored, not rejected", async () => {
    respond(body({ DrexClassifier: NOUL }));
    await score({ question: YES_NO, output: "" });
    expect(requests[0].body.state).toBe("");
    await expect(
      score({ question: YES_NO, output: "x", stateFields: ["input"] }),
    ).rejects.toThrow(/Nothing to evaluate/);
  });

  describe("API key and secrets follow the host", () => {
    const OR = "https://openrouter.ai/api/v1";
    const OR_URL = `${OR}/systemone`;
    const names = [
      "NACE_API_KEY",
      "DREX_API_KEY",
      "OPENROUTER_API_KEY",
      "NACE_BASE_URL",
      "DREX_BASE_URL",
      "NACE_VERCEL_PROTECTION_BYPASS",
    ];
    const saved: Record<string, string | undefined> = {};
    beforeEach(() => {
      for (const n of names) {
        saved[n] = process.env[n];
        delete process.env[n];
      }
    });
    afterEach(() => {
      for (const n of names) {
        if (saved[n] === undefined) delete process.env[n];
        else process.env[n] = saved[n];
      }
    });
    const bare = async (args: Record<string, unknown>) =>
      DrexClassifier({ question: YES_NO, output: "x", ...args } as any);

    test("OpenRouter uses apiKey or OPENROUTER_API_KEY, never NACE_API_KEY", async () => {
      respond(body({ DrexClassifier: NOUL }), 200, {}, OR_URL);
      process.env.NACE_API_KEY = "nace_sk_must_not_leave";
      await expect(bare({ baseUrl: OR })).rejects.toThrow(/OPENROUTER_API_KEY/);
      expect(requests).toHaveLength(0);
      process.env.OPENROUTER_API_KEY = "or_env_key";
      await bare({ baseUrl: OR });
      expect(requests[0].headers.get("authorization")).toBe(
        "Bearer or_env_key",
      );
      await bare({ baseUrl: OR, apiKey: "or_explicit" });
      expect(requests[1].headers.get("authorization")).toBe(
        "Bearer or_explicit",
      );
    });

    test("a Nace host never uses OPENROUTER_API_KEY", async () => {
      respond(body({ DrexClassifier: NOUL }));
      process.env.OPENROUTER_API_KEY = "or_must_not_leave";
      await expect(bare({ baseUrl: BASE })).rejects.toThrow(/API key/);
      expect(requests).toHaveLength(0);
      process.env.NACE_API_KEY = "nace_sk_env";
      await bare({ baseUrl: BASE });
      expect(requests[0].headers.get("authorization")).toBe(
        "Bearer nace_sk_env",
      );
    });

    test("the Vercel bypass secret is never forwarded", async () => {
      respond(body({ DrexClassifier: NOUL }), 200, {}, OR_URL);
      respond(body({ DrexClassifier: NOUL }));
      process.env.NACE_VERCEL_PROTECTION_BYPASS = "bypass-secret";
      await bare({ baseUrl: OR, apiKey: "or_key" });
      await bare({ baseUrl: BASE, apiKey: "nace_sk_test" });
      expect(requests).toHaveLength(2);
      for (const request of requests) {
        expect(request.headers.has("x-vercel-protection-bypass")).toBe(false);
      }
    });

    test("an on-prem host uses the Nace key and plain Nace shaping", async () => {
      const onPrem = "https://drex.customer.example";
      process.env.NACE_API_KEY = "nace_sk_onprem";
      process.env.OPENROUTER_API_KEY = "or_must_not_leave";
      respond(
        body({ DrexClassifier: NOUL }),
        200,
        {},
        `${onPrem}/v1/systemone`,
      );
      await bare({ baseUrl: `${onPrem}/`, images: [RED] });
      const { body: payload, headers } = requests[0];
      expect(headers.get("authorization")).toBe("Bearer nace_sk_onprem");
      expect(payload.model).toBe("drex-max");
      expect(payload.images).toHaveLength(1);
      expect(headers.has("x-openrouter-title")).toBe(false);
      expect(headers.has("http-referer")).toBe(false);
      expect(headers.get("x-nace-client")).toBe(DREX_CLIENT_NAME);
      // NACE_BASE_URL alone reaches the same host, and an explicit apiKey wins over the environment.
      process.env.NACE_BASE_URL = onPrem;
      await bare({ apiKey: "nace_sk_explicit" });
      expect(requests[1].headers.get("authorization")).toBe(
        "Bearer nace_sk_explicit",
      );
    });

    test("the billing link follows the configured host", async () => {
      const onPrem = "https://drex.customer.example";
      const insufficient = {
        error: { type: "insufficient_credit", message: "No available funds" },
      };
      respond(insufficient, 402, {}, `${onPrem}/v1/systemone`);
      let error = await bare({
        baseUrl: onPrem,
        apiKey: "nace_sk_onprem",
      }).catch((e) => e);
      expect(error.message).toContain(`${onPrem}/dashboard/billing`);
      // With no base URL the link is Nace's console.
      respond(insufficient, 402, {}, "https://console.nace.ai/v1/systemone");
      error = await bare({ apiKey: "nace_sk_test" }).catch((e) => e);
      expect(error.message).toContain(
        "https://console.nace.ai/dashboard/billing",
      );
    });

    test("the OpenRouter host match is exact", async () => {
      respond(
        body({ DrexClassifier: NOUL }),
        200,
        {},
        "https://notopenrouter.ai/v1/systemone",
      );
      await bare({ baseUrl: "https://notopenrouter.ai", apiKey: "k" });
      expect(requests[0].headers.has("x-openrouter-title")).toBe(false);
      respond(
        body({ DrexClassifier: NOUL }),
        200,
        {},
        "https://api.openrouter.ai/v1/systemone",
      );
      await bare({ baseUrl: "https://api.openrouter.ai/v1", apiKey: "k" });
      respond(
        body({ DrexClassifier: NOUL }),
        200,
        {},
        "https://openrouter.ai./api/v1/systemone",
      );
      await bare({ baseUrl: "https://openrouter.ai./api/v1", apiKey: "k" });
      expect(requests[2].headers.get("x-openrouter-title")).toBe(
        "Drex for Autoevals",
      );
      expect(requests[1].headers.get("x-openrouter-title")).toBe(
        "Drex for Autoevals",
      );
    });
  });

  test("OpenRouter's default model is namespaced and bare ids are refused", async () => {
    const baseUrl = "https://openrouter.ai/api/v1";
    respond(
      body({ DrexClassifier: NOUL }),
      200,
      {},
      "https://openrouter.ai/api/v1/systemone",
    );
    await score({ question: YES_NO, output: "x", baseUrl, apiKey: "or_key" });
    expect(requests[0].body.model).toBe("nace-ai/drex-max");
    await score({
      question: YES_NO,
      output: "x",
      baseUrl,
      apiKey: "or_key",
      model: "nace-ai/drex-v2.0",
    });
    expect(requests[1].body.model).toBe("nace-ai/drex-v2.0");
    for (const model of ["drex-max", "drex-v2.0", "drex-latest"]) {
      await expect(
        score({
          question: YES_NO,
          output: "x",
          baseUrl,
          apiKey: "or_key",
          model,
        }),
      ).rejects.toThrow(/namespaced/);
    }
    expect(requests).toHaveLength(2);
    // Nace direct keeps the bare default.
    respond(body({ DrexClassifier: NOUL }));
    await score({ question: YES_NO, output: "x" });
    expect(requests[2].body.model).toBe("drex-max");
  });

  test("drex-latest stays text-only on OpenRouter", async () => {
    await expect(
      score({
        question: YES_NO,
        output: "x",
        baseUrl: "https://openrouter.ai/api/v1",
        apiKey: "or_key",
        model: "nace-ai/drex-latest",
        images: [RED],
      }),
    ).rejects.toThrow(/reads text only/);
  });

  test("a response without a request id is still scored", async () => {
    const { request_id: _omitted, ...payload } = body({
      DrexClassifier: NOUL,
    });
    respond(payload);
    const result = await score({ question: YES_NO, output: "x" });
    expect(result.score).toBeCloseTo(0.9);
    expect((result.metadata as any).request_id).toBeUndefined();
  });

  test("the request body and headers follow the Drex contract", async () => {
    respond(body({ DrexClassifier: CHOICE }));
    await score({ question: CHOICES, output: "Paris", input: "Capital?" });
    const { body: payload, headers } = requests[0];
    expect(payload).toEqual({
      state: { input: "Capital?", output: "Paris" },
      questions: {
        DrexClassifier: {
          type: "choice",
          instructions: "",
          criteria: { good: "Fully right", ok: null, bad: "Wrong" },
        },
      },
      model: "drex-max",
    });
    expect(headers.get("content-type")).toBe("application/json");
    expect(headers.get("accept")).toBe("application/json");
    expect(headers.get("authorization")).toBe("Bearer nace_sk_test");
    expect(headers.get("x-nace-client")).toBe(DREX_CLIENT_NAME);
    expect(headers.get("user-agent")).toBe(DREX_CLIENT_NAME);
  });

  test("the request id comes from the header, else the body, and reaches errors", async () => {
    respond(body({ DrexClassifier: NOUL }), 200, { "x-request-id": "req_h" });
    let result = await score({ question: YES_NO, output: "x" });
    expect((result.metadata as any).request_id).toBe("req_h");
    respond(body({ DrexClassifier: NOUL }));
    result = await score({ question: YES_NO, output: "x" });
    expect((result.metadata as any).request_id).toBe("req_test123");
    respond({ error: { type: "authentication_error", message: "no" } }, 401, {
      "x-request-id": "req_401",
    });
    const error = await score({ question: YES_NO, output: "x" }).catch(
      (e) => e,
    );
    expect(error.requestId).toBe("req_401");
    expect(error.message).toContain("req_401");
  });

  test("authentication errors and a server veto are not retried", async () => {
    respondSequence([
      {
        status: 401,
        json: { error: { type: "authentication_error", message: "no" } },
      },
    ]);
    await expect(score({ question: YES_NO, output: "x" })).rejects.toThrow(
      /401/,
    );
    expect(requests).toHaveLength(1);
    respondSequence([
      {
        status: 503,
        json: { error: { message: "stop", retryable: false } },
      },
    ]);
    await expect(score({ question: YES_NO, output: "x" })).rejects.toThrow(
      /503/,
    );
    expect(requests).toHaveLength(2);
  });

  test("retry-after seconds and dates are honoured, retry-after-ms wins, backoff is the fallback", async () => {
    const busy = { error: { type: "overloaded_error", message: "busy" } };
    const ok = { status: 200, json: body({ DrexClassifier: NOUL }) };
    const elapsed = async (first: Record<string, string>, status = 429) => {
      requests = [];
      respondSequence([{ status, headers: first, json: busy }, ok]);
      const started = Date.now();
      await score({ question: YES_NO, output: "x" });
      expect(requests).toHaveLength(2);
      return Date.now() - started;
    };
    expect(await elapsed({ "retry-after": "0" })).toBeLessThan(300);
    expect(
      await elapsed({
        "retry-after": new Date(Date.now() - 5000).toUTCString(),
      }),
    ).toBeLessThan(300);
    expect(
      await elapsed({ "retry-after-ms": "5", "retry-after": "30" }),
    ).toBeLessThan(300);
    // No usable header: 500 ms backoff, less up to 25% jitter.
    expect(await elapsed({}, 500)).toBeGreaterThanOrEqual(300);
  });

  test("connection errors are retried, then reported", async () => {
    let calls = 0;
    server.use(
      http.post(URL_, async ({ request }) => {
        requests.push({ body: await request.json(), headers: request.headers });
        if (calls++ === 0) return HttpResponse.error();
        return HttpResponse.json(body({ DrexClassifier: NOUL }));
      }),
    );
    const result = await score({ question: YES_NO, output: "x" });
    expect(result.score).toBeCloseTo(0.9);
    expect(requests).toHaveLength(2);
    server.use(http.post(URL_, () => HttpResponse.error()));
    const error = await score({
      question: YES_NO,
      output: "x",
      maxRetries: 0,
    }).catch((e) => e);
    expect(error).toBeInstanceOf(DrexError);
    expect(error.message).toMatch(/Connection error/);
    expect(error.status).toBeUndefined();
  });

  test("a timeout is retried, and the default waits at least 60 seconds", async () => {
    let aborts = 0;
    server.use(
      http.post(URL_, async () => {
        aborts++;
        await delay(200);
        return HttpResponse.json(body({ DrexClassifier: NOUL }));
      }),
    );
    const error = await score({
      question: YES_NO,
      output: "x",
      timeoutMs: 30,
      maxRetries: 1,
    }).catch((e) => e);
    expect(error.message).toMatch(/timed out/);
    expect(aborts).toBe(2);
    const spy = vi.spyOn(globalThis, "setTimeout");
    respond(body({ DrexClassifier: NOUL }));
    await score({ question: YES_NO, output: "x" });
    expect(spy.mock.calls.some(([, ms]) => ms === 60_000)).toBe(true);
    spy.mockRestore();
  });

  test("invalid timeout and retry settings are refused", async () => {
    for (const timeoutMs of [0, -1, Infinity, NaN]) {
      await expect(
        score({ question: YES_NO, output: "x", timeoutMs }),
      ).rejects.toThrow(/timeoutMs/);
    }
    await expect(
      score({ question: YES_NO, output: "x", maxRetries: -1 }),
    ).rejects.toThrow(/maxRetries/);
  });

  test("a request over 4 MiB is refused before sending", async () => {
    respond(body({ DrexClassifier: NOUL }));
    await expect(
      score({
        question: YES_NO,
        output: "x".repeat(4 * 1024 * 1024 + 1),
      }),
    ).rejects.toThrow(/4 MiB/);
    expect(requests).toHaveLength(0);
    await score({ question: YES_NO, output: "y".repeat(3 * 1024 * 1024) });
    expect(requests).toHaveLength(1);
  });

  test("422 issues are listed as path and message", async () => {
    respond(
      {
        error: {
          type: "invalid_request_error",
          message: "invalid request",
          issues: [
            { path: "questions.q1.criteria", message: "must not be empty" },
            { path: "images", message: "at most 8 images" },
          ],
        },
      },
      422,
    );
    const error = await score({ question: YES_NO, output: "x" }).catch(
      (e) => e,
    );
    expect(error.message).toContain("422 invalid request");
    expect(error.message).toContain(
      "\n  questions.q1.criteria: must not be empty",
    );
    expect(error.message).toContain("\n  images: at most 8 images");
  });

  test("error bodies that are not Drex JSON still give a message", async () => {
    server.use(
      http.post(
        URL_,
        () => new HttpResponse("<html>bad gateway</html>", { status: 502 }),
      ),
    );
    let error = await score({
      question: YES_NO,
      output: "x",
      maxRetries: 0,
    }).catch((e) => e);
    expect(error.message).toContain("502 <html>bad gateway</html>");
    server.use(http.post(URL_, () => new HttpResponse(null, { status: 500 })));
    error = await score({ question: YES_NO, output: "x", maxRetries: 0 }).catch(
      (e) => e,
    );
    expect(error.message).toContain("500 status code (no body)");
  });

  test("malformed and missing answers are errors", async () => {
    respond({ model: "drex-max", usage: {} });
    await expect(score({ question: YES_NO, output: "x" })).rejects.toThrow(
      /unexpected response/,
    );
    respond(body({ DrexClassifier: { type: "noul" } }));
    await expect(score({ question: YES_NO, output: "x" })).rejects.toThrow(
      /malformed answer/,
    );
    respond(body({ DrexClassifier: { type: "noul", noul: "high" } }));
    await expect(score({ question: YES_NO, output: "x" })).rejects.toThrow(
      /malformed answer/,
    );
    // An answer of another type, or one this client does not know, is no answer to the question asked.
    respond(body({ DrexClassifier: CHOICE }));
    await expect(score({ question: YES_NO, output: "x" })).rejects.toThrow(
      /no answer/,
    );
    respond(body({ DrexClassifier: { type: "future", x: 1 } }));
    await expect(score({ question: YES_NO, output: "x" })).rejects.toThrow(
      /no answer/,
    );
  });

  test("usage fields are optional", async () => {
    const { evaluation_time_ms: _t, ...payload } = body({
      DrexClassifier: NOUL,
    });
    respond({ ...payload, usage: { input_tokens: 5 } });
    const result = await score({ question: YES_NO, output: "x" });
    const metadata = result.metadata as any;
    expect(metadata.usage).toEqual({
      input_tokens: 5,
      image_tokens: null,
      output_tokens: null,
    });
    expect(metadata.evaluation_time_ms).toBeNull();
  });

  test("images must be inline and are never dropped", async () => {
    respond(body({ DrexClassifier: NOUL }));
    const run = (images: unknown[]) =>
      score({ question: YES_NO, output: "x", images });
    await expect(
      run([new TextEncoder().encode("%PDF-1.7 no")]),
    ).rejects.toThrow(/PNG, JPEG and WebP/);
    await expect(run(["/tmp/not-base64.png"])).rejects.toThrow(
      /neither base64 nor a data URL/,
    );
    await expect(run([42])).rejects.toThrow(/images\[0\]/);
    await expect(
      run([RED, { type: "url", url: "https://x.test/a.png" }]),
    ).rejects.toThrow(/images\[1\]/);
    expect(requests).toHaveLength(0);
    const b64 = Buffer.from(RED).toString("base64");
    await run([
      b64,
      `data:image/png;base64,${b64}`,
      BLUE,
      { workspace_id: "ws_1", file_id: "file_1" },
    ]);
    const images = requests[0].body.images;
    expect(images[0]).toBe(b64);
    expect(images[1]).toBe(`data:image/png;base64,${b64}`);
    expect(Buffer.from(images[2].split(",")[1], "base64")).toEqual(
      Buffer.from(BLUE),
    );
    expect(images[3]).toEqual({
      type: "workspace_file",
      workspace_id: "ws_1",
      file_id: "file_1",
    });
  });

  test("a missing answer is an error", async () => {
    respond(body({}));
    await expect(score({ question: YES_NO, output: "x" })).rejects.toThrow(
      /no answer/,
    );
  });

  test("noul criteria must be non-empty strings or null", async () => {
    await expect(
      score({
        question: drexQuestion.noul(undefined, { true: "" }),
        output: "x",
      }),
    ).rejects.toThrow(/instructions or criteria/);
    await expect(
      score({
        question: {
          type: "noul",
          instructions: "Right?",
          criteria: { true: 1 },
        },
        output: "x",
      }),
    ).rejects.toThrow(/strings or null/);
    respond(body({ DrexClassifier: NOUL }));
    await score({
      question: {
        type: "noul",
        instructions: "Right?",
        criteria: { true: "yes", false: null },
      },
      output: "x",
    });
    expect(requests).toHaveLength(1);
  });

  test("invalid questions fail before any request", async () => {
    await expect(
      score({
        question: drexQuestion.choice(
          { a: null, b: null },
          { scores: { a: 1 } },
        ),
        output: "x",
      }),
    ).rejects.toThrow(/cover exactly/);
    await expect(
      score({ question: drexQuestion.score(["only"]), output: "x" }),
    ).rejects.toThrow(/at least two levels/);
    await expect(
      score({ question: drexQuestion.noul(), output: "x" }),
    ).rejects.toThrow(/instructions or criteria/);
    await expect(
      score({ question: YES_NO, output: "x", threshold: 2 }),
    ).rejects.toThrow(/threshold/);
    expect(requests).toHaveLength(0);
  });

  test("works through .partial", async () => {
    respond(body({ Grounded: NOUL }));
    const grounded = DrexClassifier.partial({
      ...auth,
      question: YES_NO,
      name: "Grounded",
    } as any);
    const result = await grounded({ output: "x" } as any);
    expect(result.name).toBe("Grounded");
  });
});

// Live checks: need a real key.
describe.skipIf(!process.env.NACE_API_KEY)("live", () => {
  // The offline tests have run by now. Stop msw so these calls reach the real Drex instead of failing
  // as unhandled requests (closing restores the global fetch).
  beforeAll(() => server.close());
  const live = { baseUrl: process.env.NACE_BASE_URL, apiKey: undefined };
  for (const model of ["drex-max", "drex-v2.0"]) {
    test(`live text on ${model}`, async () => {
      const question = drexQuestion.noul(
        "Does the output state that Paris is the capital of France?",
      );
      const good = await DrexClassifier({
        ...live,
        question,
        model,
        output: "The capital of France is Paris.",
      });
      const bad = await DrexClassifier({
        ...live,
        question,
        model,
        output: "The capital of France is Berlin.",
      });
      expect(good.score!).toBeGreaterThan(0.8);
      expect(bad.score!).toBeLessThan(0.2);
    });
  }

  test("live images on drex-max", async () => {
    const question = drexQuestion.noul("Is Image 1 solid red?");
    const forward = await DrexClassifier({
      ...live,
      question,
      model: "drex-max",
      output: "Look at the images.",
      images: [RED, BLUE],
    });
    const reverse = await DrexClassifier({
      ...live,
      question,
      model: "drex-max",
      output: "Look at the images.",
      images: [BLUE, RED],
    });
    expect(forward.score!).toBeGreaterThan(0.8);
    expect(reverse.score!).toBeLessThan(0.2);
  });
});

// Live OpenRouter checks: need OPENROUTER_API_KEY and never NACE_API_KEY.
describe.skipIf(!process.env.OPENROUTER_API_KEY)("live OpenRouter", () => {
  const saved: Record<string, string | undefined> = {};
  const names = [
    "NACE_API_KEY",
    "DREX_API_KEY",
    "NACE_BASE_URL",
    "DREX_BASE_URL",
    "NACE_VERCEL_PROTECTION_BYPASS",
  ];
  beforeAll(() => {
    server.close(); // reach the real OpenRouter, not msw
    for (const n of names) {
      saved[n] = process.env[n];
      delete process.env[n]; // prove the Nace key is neither needed nor sent
    }
  });
  afterAll(() => {
    for (const n of names) {
      if (saved[n] !== undefined) process.env[n] = saved[n];
    }
  });
  const base = { baseUrl: "https://openrouter.ai/api/v1" };

  for (const model of [undefined, "nace-ai/drex-v2.0", "nace-ai/drex-max"]) {
    test(`live OpenRouter text on ${model ?? "the default model"}`, async () => {
      const question = drexQuestion.noul(
        "Does the output state that Paris is the capital of France?",
      );
      const good = await DrexClassifier({
        ...base,
        question,
        model,
        output: "The capital of France is Paris.",
      });
      const bad = await DrexClassifier({
        ...base,
        question,
        model,
        output: "The capital of France is Berlin.",
      });
      expect(good.score!).toBeGreaterThan(0.8);
      expect(bad.score!).toBeLessThan(0.2);
    });
  }

  test("live OpenRouter inline image on nace-ai/drex-max", async () => {
    const question = drexQuestion.noul("Is Image 1 solid red?");
    const args = { ...base, question, model: "nace-ai/drex-max" };
    const forward = await DrexClassifier({
      ...args,
      output: "Look at the images.",
      images: [RED, BLUE],
    });
    const reverse = await DrexClassifier({
      ...args,
      output: "Look at the images.",
      images: [BLUE, RED],
    });
    expect(forward.score!).toBeGreaterThan(0.8);
    expect(reverse.score!).toBeLessThan(0.2);
  });
});
