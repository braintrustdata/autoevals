/**
 * Drex decision-model scorer.
 *
 * `DrexClassifier` asks a Drex decision model (https://console.nace.ai/docs) one typed question about an
 * evaluation case and turns the answer's probabilities into a `Score`. Drex answers with probabilities,
 * not a verdict, so the score is the probability mass on the outcomes you call a pass, and the original
 * probabilities stay in `metadata`.
 *
 * It calls `POST {baseUrl}/v1/systemone` with the global `fetch` and needs no extra package. Credentials
 * come from `NACE_API_KEY` (and `NACE_BASE_URL`, which may name your own Drex deployment) unless passed
 * explicitly.
 *
 * @module drex
 */

import { makePartial, ScorerWithPartial } from "./partial";
import { Score } from "./score";
import { VERSION } from "./version";

export const DREX_CLIENT_NAME = `autoevals/${VERSION}`;
export const DREX_DEFAULT_MODEL = "drex-max";
// OpenRouter namespaces model ids: `drex-max` there is `nace-ai/drex-max`, and a bare id does not exist.
const OPENROUTER_MODEL_PREFIX = "nace-ai/";
// Released versions that read text only; Drex enforces every other limit itself.
const TEXT_ONLY_MODELS = new Set(["drex-v1.5", "drex-v2.0", "drex-latest"]);
// Drex holds a decision for up to 55 s, so the client must wait longer.
const DEFAULT_TIMEOUT_MS = 60_000;
const DEFAULT_MAX_RETRIES = 2;
const DEFAULT_BASE_URL = "https://console.nace.ai";
const SYSTEM_ONE_PATH = "/v1/systemone";
// A server-requested wait longer than this is not slept: the error is raised instead.
const MAX_RETRY_WAIT_MS = 60_000;
const BACKOFF_INITIAL_MS = 500;
const BACKOFF_MAX_MS = 5_000;
const BACKOFF_JITTER = 0.25;
const RETRY_STATUSES = (status: number) =>
  status === 408 || status === 429 || (status >= 500 && status <= 599);
// Drex refuses a larger request body; images go in inline, so the client says so before sending.
const MAX_BODY_BYTES = 4 * 1024 * 1024;
const MAX_IMAGE_BYTES = 16 * 1024 * 1024;
const STATE_FIELDS = ["input", "output", "expected", "context"] as const;
type StateField = (typeof STATE_FIELDS)[number];

/** A request to Drex failed. This is an error in the evaluation, never a low score. */
export class DrexError extends Error {
  readonly status?: number;
  readonly requestId?: string;
  constructor(message: string, status?: number, requestId?: string) {
    super(message);
    this.name = "DrexError";
    this.status = status;
    this.requestId = requestId;
  }
}

/** Questions for Drex, with how each answer becomes a 0-1 score (higher is better). */
export type DrexQuestion =
  | {
      type: "noul";
      instructions?: string | null;
      /** Optional descriptions of the yes and no outcomes. */
      criteria?: { true?: string | null; false?: string | null } | null;
      /** Whether a yes answer is good. With `"fail"` the score is the probability of no. Default `"pass"`. */
      yesIs?: "pass" | "fail";
    }
  | {
      type: "choice";
      instructions?: string | null;
      /** Labels mapped to descriptions, or null for undescribed labels. */
      criteria: Record<string, string | null>;
      /** The score of each label, from 0 to 1. The score is the probability-weighted sum. */
      scores: Record<string, number>;
    }
  | {
      type: "score";
      instructions?: string | null;
      /** An ordered rubric, lowest level first. */
      criteria: [string, ...string[]];
      /** Treat the lowest level as best: the score is one minus the normalized level. */
      reverse?: boolean;
    };

/** Builders for `DrexQuestion`. */
export const drexQuestion = {
  noul(
    instructions?: string,
    options: { true?: string; false?: string; yesIs?: "pass" | "fail" } = {},
  ): DrexQuestion {
    const criteria: { true?: string; false?: string } = {};
    if (options.true !== undefined) criteria.true = options.true;
    if (options.false !== undefined) criteria.false = options.false;
    return {
      type: "noul",
      instructions,
      criteria: Object.keys(criteria).length ? criteria : undefined,
      yesIs: options.yesIs ?? "pass",
    };
  },
  choice(
    choices: Record<string, string | null>,
    options: { scores: Record<string, number>; instructions?: string },
  ): DrexQuestion {
    return {
      type: "choice",
      instructions: options.instructions,
      criteria: choices,
      scores: options.scores,
    };
  },
  score(
    levels: [string, ...string[]],
    options: { instructions?: string; reverse?: boolean } = {},
  ): DrexQuestion {
    return {
      type: "score",
      instructions: options.instructions,
      criteria: levels,
      reverse: options.reverse ?? false,
    };
  },
};

/** An image for Drex Max: bytes, a base64 string or data URL, or an uploaded `workspace_file` reference. */
export type DrexImage = Uint8Array | string | Record<string, unknown>;

export interface DrexAuth {
  /**
   * Defaults to `NACE_API_KEY`. For OpenRouter it defaults to `OPENROUTER_API_KEY` instead and never to
   * `NACE_API_KEY`.
   */
  apiKey?: string;
  /**
   * Defaults to `NACE_BASE_URL`, then Nace direct. Use `https://openrouter.ai/api/v1` for OpenRouter
   * (model ids `nace-ai/drex-max`).
   */
  baseUrl?: string;
  /** Milliseconds per attempt. At least 60000 is advised. */
  timeoutMs?: number;
  /**
   * Retries on 408, 429 and 5xx (529 included) and on connection errors, honouring `retry-after-ms`, then
   * `retry-after`. A requested wait over 60 s is not slept: the error is raised.
   */
  maxRetries?: number;
}

export type DrexArgs = DrexAuth & {
  /** The question and how to score its answer. */
  question: DrexQuestion;
  /** The score's name. Defaults to `DrexClassifier`. */
  name?: string;
  /** A Drex model id, passed through unchanged. Defaults to `drex-max` (`nace-ai/drex-max` on OpenRouter, where a bare id is refused). */
  model?: string;
  /** Images every question sees (`drex-max` only). */
  images?: DrexImage[];
  /** Which of `input`, `output`, `expected` and `context` Drex reads. */
  stateFields?: StateField[];
  /** When set, the score is 1 if the 0-1 score reaches it and 0 otherwise; the raw value stays in metadata. */
  threshold?: number;
  input?: unknown;
  context?: unknown;
};

export type DrexManyArgs = Omit<DrexArgs, "question" | "name"> & {
  /** Questions to answer in one request, keyed by name. */
  questions: Record<string, DrexQuestion>;
  output: unknown;
  expected?: unknown;
};

export function isTextOnly(model: string): boolean {
  const id = model.split("/").pop()!.replace(/^~+/, "");
  return TEXT_ONLY_MODELS.has(id);
}

function isOpenRouterHost(host: string): boolean {
  const name = host.toLowerCase().replace(/\.+$/, "");
  return name === "openrouter.ai" || name.endsWith(".openrouter.ai");
}

function isOpenRouter(baseUrl?: string): boolean {
  if (!baseUrl) return false;
  try {
    return isOpenRouterHost(new URL(baseUrl).hostname);
  } catch {
    return false;
  }
}

/** An environment value: blank counts as unset, then the legacy `DREX_*` name is tried. */
function env(name: string, legacy: string): string | undefined {
  return process.env[name]?.trim() || process.env[legacy]?.trim() || undefined;
}

/** What Drex reads: `output` alone as text, otherwise an object of the case fields that are set. */
export function buildState(
  values: Partial<Record<StateField, unknown>>,
  fields: readonly StateField[] = STATE_FIELDS,
): unknown {
  const chosen: Record<string, unknown> = {};
  for (const field of fields) {
    const value = values[field];
    if (value !== undefined && value !== null) chosen[field] = value;
  }
  const keys = Object.keys(chosen);
  if (
    keys.length === 1 &&
    keys[0] === "output" &&
    typeof chosen.output === "string"
  ) {
    return chosen.output;
  }
  return chosen;
}

function validate(name: string, q: DrexQuestion): void {
  if (q.type === "noul") {
    if (q.yesIs !== undefined && q.yesIs !== "pass" && q.yesIs !== "fail") {
      throw new Error(`${name}: yesIs must be 'pass' or 'fail'`);
    }
    const outcomes = Object.values(q.criteria ?? {});
    if (
      outcomes.some(
        (v) => v !== null && v !== undefined && typeof v !== "string",
      )
    ) {
      throw new Error(`${name}: noul criteria must be strings or null`);
    }
    // Drex refuses a yes/no question with neither instructions nor a described outcome.
    if (!q.instructions && !outcomes.some((v) => v)) {
      throw new Error(
        `${name}: a noul question needs instructions or criteria`,
      );
    }
  } else if (q.type === "choice") {
    const labels = Object.keys(q.criteria ?? {});
    if (!labels.length)
      throw new Error(`${name}: a choice question needs at least one label`);
    const scored = Object.keys(q.scores ?? {});
    if (
      scored.length !== labels.length ||
      !labels.every((l) => scored.includes(l))
    ) {
      throw new Error(
        `${name}: choice scores must cover exactly the labels ${JSON.stringify([...labels].sort())}`,
      );
    }
    if (scored.some((l) => !(q.scores[l] >= 0 && q.scores[l] <= 1))) {
      throw new Error(`${name}: choice scores must be between 0 and 1`);
    }
  } else if (q.type === "score") {
    if (!q.criteria || q.criteria.length < 2) {
      throw new Error(`${name}: a score question needs at least two levels`);
    }
  } else {
    throw new Error(
      `${name}: unknown question type ${JSON.stringify((q as any).type)}`,
    );
  }
}

/** The request form. `instructions` is always a string: OpenRouter requires it, Drex rejects null. */
function wire(q: DrexQuestion): Record<string, unknown> {
  const question: Record<string, unknown> = {
    type: q.type,
    instructions: q.instructions ?? "",
  };
  if (q.criteria !== undefined && q.criteria !== null)
    question.criteria = q.criteria;
  return question;
}

function dataUrl(image: string): string {
  if (image.startsWith("data:")) return image;
  const head = Buffer.from(image.slice(0, 24), "base64");
  let mime = "application/octet-stream";
  if (
    head
      .subarray(0, 8)
      .equals(Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]))
  ) {
    mime = "image/png";
  } else if (head[0] === 0xff && head[1] === 0xd8 && head[2] === 0xff) {
    mime = "image/jpeg";
  } else if (
    head.subarray(0, 4).toString("latin1") === "RIFF" &&
    head.subarray(8, 12).toString("latin1") === "WEBP"
  ) {
    mime = "image/webp";
  }
  return `data:${mime};base64,${image}`;
}

function mediaType(bytes: Uint8Array): string | undefined {
  const starts = (...head: number[]) => head.every((b, i) => bytes[i] === b);
  if (starts(0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a))
    return "image/png";
  if (starts(0xff, 0xd8, 0xff)) return "image/jpeg";
  if (
    starts(0x52, 0x49, 0x46, 0x46) &&
    bytes[8] === 0x57 &&
    bytes[9] === 0x45 &&
    bytes[10] === 0x42 &&
    bytes[11] === 0x50
  ) {
    return "image/webp";
  }
  return undefined;
}

const BASE64 = /^[A-Za-z0-9+/]+={0,2}$/;

/**
 * One `images` entry as the request sends it: base64, a data URL, or a `workspace_file` reference. Drex
 * does not fetch URLs or read paths, and images are never dropped: anything else is refused.
 */
function prepareImage(image: DrexImage, index: number): unknown {
  if (typeof image === "string") {
    if (/^https?:\/\//.test(image)) {
      throw new Error(
        "Drex doesn't fetch image URLs. Pass the image bytes, a base64 string or data URL, or a workspace_file reference.",
      );
    }
    if (
      !image.startsWith("data:") &&
      (image.length % 4 || !BASE64.test(image))
    ) {
      throw new Error(
        `images[${index}] is neither base64 nor a data URL; pass the image bytes instead.`,
      );
    }
    return image;
  }
  if (image instanceof Uint8Array) {
    const type = mediaType(image);
    if (!type) {
      throw new Error(
        "Drex Max reads PNG, JPEG and WebP images only; these bytes are none of them.",
      );
    }
    if (image.length > MAX_IMAGE_BYTES) {
      throw new Error(
        `The image is ${image.length} bytes, over Drex's 16 MiB per image.`,
      );
    }
    return `data:${type};base64,${Buffer.from(image).toString("base64")}`;
  }
  if (image && typeof image === "object" && !Array.isArray(image)) {
    const { type, ...rest } = image as Record<string, unknown>;
    if (
      type === "workspace_file" ||
      (type === undefined && "file_id" in rest && "workspace_id" in rest)
    ) {
      return { type: "workspace_file", ...rest };
    }
  }
  throw new Error(
    `images[${index}] must be bytes, a base64 string, a data URL or a workspace_file reference.`,
  );
}

/** OpenRouter takes images as `image_url` parts of a `state` content array, not as `images`. */
function openrouterState(state: unknown, images: unknown[]): unknown[] {
  const parts: unknown[] = [
    {
      type: "text",
      text: typeof state === "string" ? state : JSON.stringify(state),
    },
  ];
  for (const image of images) {
    if (typeof image !== "string") {
      throw new Error(
        "OpenRouter accepts inline images only, not uploaded workspace files.",
      );
    }
    parts.push({ type: "image_url", image_url: { url: dataUrl(image) } });
  }
  return parts;
}

type Answer = any;

function num(value: unknown): number {
  if (typeof value !== "number") throw new TypeError("not a number");
  return value;
}

/** Return the 0-1 score (higher is better) and the answer's own figures. */
function normalize(
  q: DrexQuestion,
  answer: Answer,
): [number, Record<string, unknown>] {
  if (q.type === "noul") {
    const pYes = num(answer.noul);
    const yesIs = q.yesIs ?? "pass";
    return [
      yesIs === "pass" ? pYes : 1 - pYes,
      {
        probabilities: { yes: pYes, no: 1 - pYes },
        yes_is: yesIs,
        // Computed here: older Drex deployments return no confidence on noul answers.
        confidence: Math.max(pYes, 1 - pYes),
      },
    ];
  }
  if (q.type === "choice") {
    const probabilities: Record<string, number> = {};
    for (const [label, p] of Object.entries(
      answer.probabilities as Record<string, number>,
    )) {
      probabilities[label] = num(p);
    }
    let score = 0;
    for (const [label, p] of Object.entries(probabilities))
      score += p * (q.scores[label] ?? 0);
    return [
      score,
      {
        choice: answer.choice,
        confidence: num(answer.confidence),
        probabilities,
        choice_scores: { ...q.scores },
      },
    ];
  }
  const levels = q.criteria.length;
  const probabilities: Record<string, number> = {};
  for (const [level, p] of Object.entries(
    answer.probabilities as Record<string, number>,
  )) {
    probabilities[String(level)] = num(p);
  }
  const normalized = num(answer.score) / (levels - 1);
  return [
    q.reverse ? 1 - normalized : normalized,
    {
      raw_score: num(answer.score),
      scale: [0, levels - 1],
      legend: [...answer.legend],
      confidence: num(answer.confidence),
      probabilities,
      reverse: q.reverse ?? false,
    },
  ];
}

const clamp = (value: number) => Math.min(1, Math.max(0, value));

class TransportError extends Error {
  constructor(
    message: string,
    readonly timeout = false,
  ) {
    super(message);
  }
}

interface Reply {
  status: number;
  headers: Headers;
  text: string;
}

async function post(
  url: string,
  headers: Record<string, string>,
  body: string,
  timeoutMs: number,
): Promise<Reply> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(url, {
      method: "POST",
      headers,
      body,
      signal: controller.signal,
    });
    return {
      status: response.status,
      headers: response.headers,
      text: await response.text(),
    };
  } catch (error) {
    if (controller.signal.aborted) {
      throw new TransportError(
        `Request timed out after ${timeoutMs / 1000}s`,
        true,
      );
    }
    const cause = (error as { cause?: { code?: string } }).cause?.code;
    throw new TransportError(
      `Connection error: ${(error as Error).message}${cause ? ` (${cause})` : ""}`,
    );
  } finally {
    clearTimeout(timer);
  }
}

/** The server's requested wait in ms: `retry-after-ms`, else `retry-after` (seconds or an HTTP date). */
function retryAfterMs(headers: Headers): number | undefined {
  for (const [name, scale] of [
    ["retry-after-ms", 1],
    ["retry-after", 1000],
  ] as const) {
    const raw = headers.get(name);
    if (raw === null) continue;
    const value = Number(raw.trim() || "0");
    if (Number.isFinite(value) && value >= 0) return value * scale;
    if (name === "retry-after") {
      const date = Date.parse(raw);
      if (!Number.isNaN(date)) return Math.max(0, date - Date.now());
    }
  }
  return undefined;
}

function parseJson(text: string): unknown {
  if (!text) return undefined;
  try {
    return JSON.parse(text);
  } catch {
    return text;
  }
}

const isRecord = (value: unknown): value is Record<string, any> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

/** Milliseconds to wait before retry number `attempt`, or undefined to stop and raise. */
function retryDelayMs(
  attempt: number,
  maxRetries: number,
  reply?: Reply,
): number | undefined {
  if (attempt > maxRetries) return undefined;
  if (reply) {
    if (!RETRY_STATUSES(reply.status)) return undefined;
    const body = parseJson(reply.text);
    if (
      isRecord(body) &&
      isRecord(body.error) &&
      body.error.retryable === false
    ) {
      return undefined;
    }
    const wait = retryAfterMs(reply.headers);
    if (wait !== undefined) return wait <= MAX_RETRY_WAIT_MS ? wait : undefined;
  }
  const backoff = Math.min(
    BACKOFF_INITIAL_MS * 2 ** (attempt - 1),
    BACKOFF_MAX_MS,
  );
  return Math.round(backoff * (1 - Math.random() * BACKOFF_JITTER));
}

function errorMessage(body: unknown): string | undefined {
  if (typeof body === "string") return body.slice(0, 200) || undefined;
  if (!isRecord(body)) return undefined;
  const { error, message, detail } = body;
  if (typeof error === "string") return error;
  if (isRecord(error) && typeof error.message === "string")
    return error.message;
  if (typeof message === "string") return message;
  if (typeof detail === "string") return detail;
  if (isRecord(detail) && typeof detail.message === "string")
    return detail.message;
  return undefined;
}

function apiError(
  reply: Reply,
  baseUrl: string | undefined,
  openrouter: boolean,
): DrexError {
  const body = parseJson(reply.text);
  let message = `${reply.status} ${errorMessage(body) ?? "status code (no body)"}`;
  const issues =
    isRecord(body) && isRecord(body.error) ? body.error.issues : undefined;
  if (Array.isArray(issues)) {
    for (const issue of issues.filter(isRecord))
      message += `\n  ${issue.path ?? ""}: ${issue.message ?? ""}`;
  }
  if (reply.status === 402) {
    if (openrouter) {
      message += "\nAdd credits at https://openrouter.ai/settings/credits.";
    } else {
      const host = (baseUrl ?? DEFAULT_BASE_URL).replace(/\/+$/, "");
      message += `\nAdd funds or a card at ${host}/dashboard/billing.`;
    }
  }
  const bodyId =
    isRecord(body) && typeof body.request_id === "string"
      ? body.request_id
      : undefined;
  const requestId = reply.headers.get("x-request-id") ?? bodyId;
  if (requestId) message += ` (request_id=${requestId})`;
  return new DrexError(message, reply.status, requestId);
}

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

async function run(
  args: DrexAuth & {
    model?: string;
    images?: DrexImage[];
    stateFields?: StateField[];
    threshold?: number;
    input?: unknown;
    context?: unknown;
    output: unknown;
    expected?: unknown;
  },
  questions: Record<string, DrexQuestion>,
  singleName?: string,
): Promise<Score[]> {
  const names = Object.keys(questions);
  if (!names.length) throw new Error("At least one question is required");
  for (const name of names) validate(name, questions[name]);
  const { threshold } = args;
  if (threshold !== undefined && !(threshold >= 0 && threshold <= 1)) {
    throw new Error("threshold must be between 0 and 1");
  }
  const timeoutMs = args.timeoutMs ?? DEFAULT_TIMEOUT_MS;
  if (!Number.isFinite(timeoutMs) || timeoutMs <= 0) {
    throw new Error(
      "timeoutMs must be a positive, finite number of milliseconds",
    );
  }
  const maxRetries = args.maxRetries ?? DEFAULT_MAX_RETRIES;
  if (!(maxRetries >= 0)) throw new Error("maxRetries must not be negative");
  const fields = args.stateFields ?? STATE_FIELDS;
  const unknownFields = fields.filter((f) => !STATE_FIELDS.includes(f));
  if (unknownFields.length) {
    throw new Error(
      `Unknown state fields ${JSON.stringify(unknownFields)}; choose from ${JSON.stringify(STATE_FIELDS)}`,
    );
  }
  const configuredBaseUrl =
    args.baseUrl || env("NACE_BASE_URL", "DREX_BASE_URL");
  const openrouter = isOpenRouter(configuredBaseUrl);
  if (args.model !== undefined && !args.model.trim()) {
    throw new Error("model must not be empty");
  }
  if (openrouter && args.model && !args.model.includes("/")) {
    throw new Error(
      `OpenRouter model ids are namespaced: use '${OPENROUTER_MODEL_PREFIX}${args.model}' (for example ${OPENROUTER_MODEL_PREFIX}${DREX_DEFAULT_MODEL}), not '${args.model}'.`,
    );
  }
  const model =
    args.model ||
    (openrouter
      ? OPENROUTER_MODEL_PREFIX + DREX_DEFAULT_MODEL
      : DREX_DEFAULT_MODEL);
  let apiKey: string | undefined;
  if (openrouter) {
    // The key follows the host: never fall back to NACE_API_KEY for OpenRouter.
    apiKey = args.apiKey?.trim() || process.env.OPENROUTER_API_KEY?.trim();
    if (!apiKey) {
      throw new DrexError(
        "No OpenRouter API key was provided. Pass apiKey or set OPENROUTER_API_KEY.",
      );
    }
  } else {
    apiKey = args.apiKey?.trim() || env("NACE_API_KEY", "DREX_API_KEY");
    if (!apiKey) {
      throw new DrexError(
        "No API key was provided. Pass apiKey or set the NACE_API_KEY environment variable.",
      );
    }
  }
  if (!/^[\x21-\x7e]+$/.test(apiKey)) {
    throw new DrexError(
      "The API key must contain only printable ASCII characters without whitespace.",
    );
  }

  let state = buildState(
    {
      input: args.input,
      output: args.output,
      expected: args.expected,
      context: args.context,
    },
    fields,
  );
  if (
    state === undefined ||
    (typeof state === "object" && state !== null && !Object.keys(state).length)
  ) {
    throw new Error(
      "Nothing to evaluate: pass at least one of output, expected, input or context",
    );
  }

  let images = (args.images ?? []).map((image, index) =>
    prepareImage(image, index),
  );
  if (images.length && isTextOnly(model)) {
    throw new Error(
      `${model} reads text only, so it can't take images. Use an image model such as model='drex-max'.`,
    );
  }
  if (images.length && openrouter) {
    state = openrouterState(state, images);
    images = [];
  }

  let base = (configuredBaseUrl ?? DEFAULT_BASE_URL).replace(/\/+$/, "");
  // OpenRouter documents `.../api/v1`; the route is `/v1/systemone` under `.../api`.
  if (openrouter) base = base.replace(/\/v1$/, "");
  const url = base + SYSTEM_ONE_PATH;
  const headers: Record<string, string> = {
    Authorization: `Bearer ${apiKey}`,
    "Content-Type": "application/json",
    Accept: "application/json",
    "X-Nace-Client": DREX_CLIENT_NAME,
    "User-Agent": DREX_CLIENT_NAME,
    ...(openrouter
      ? {
          "HTTP-Referer": "https://nace.ai",
          "X-OpenRouter-Title": "Drex for Autoevals",
        }
      : {}),
  };
  let content: string;
  try {
    content = JSON.stringify({
      state,
      questions: Object.fromEntries(names.map((n) => [n, wire(questions[n])])),
      model,
      ...(images.length ? { images } : {}),
    });
  } catch (error) {
    throw new Error(
      `The case can't be sent as JSON: ${(error as Error).message}`,
    );
  }
  // OpenRouter's own limit is not Drex's, so only Drex deployments are checked here.
  const bytes = Buffer.byteLength(content);
  if (!openrouter && bytes > MAX_BODY_BYTES) {
    throw new Error(
      `The request is ${(bytes / 1048576).toFixed(1)} MiB, over Drex's 4 MiB limit. Send fewer or smaller images or less text; the scorer does not upload files.`,
    );
  }

  let reply: Reply;
  for (let attempt = 1; ; attempt++) {
    let delay: number | undefined;
    try {
      reply = await post(url, headers, content, timeoutMs);
      if (reply.status >= 200 && reply.status < 300) break;
      delay = retryDelayMs(attempt, maxRetries, reply);
      if (delay === undefined) {
        throw apiError(reply, configuredBaseUrl, openrouter);
      }
    } catch (error) {
      if (!(error instanceof TransportError)) throw error;
      delay = retryDelayMs(attempt, maxRetries);
      if (delay === undefined) throw new DrexError(error.message);
    }
    await sleep(delay);
  }

  const parsed = parseJson(reply.text);
  const bodyId =
    isRecord(parsed) && typeof parsed.request_id === "string"
      ? parsed.request_id
      : undefined;
  const requestId = reply.headers.get("x-request-id") ?? bodyId;
  if (!isRecord(parsed) || !isRecord(parsed.answers)) {
    throw new DrexError(
      "Drex returned an unexpected response (no answers).",
      reply.status,
      requestId,
    );
  }
  const answers = parsed.answers;
  const rawUsage = isRecord(parsed.usage) ? parsed.usage : {};
  const usage = {
    input_tokens: rawUsage.input_tokens ?? null,
    image_tokens: rawUsage.image_tokens ?? null,
    output_tokens: rawUsage.output_tokens ?? null,
  };
  return names.map((questionName, index) => {
    const question = questions[questionName];
    const answer = answers[questionName];
    if (!isRecord(answer) || answer.type !== question.type) {
      throw new DrexError(
        `Drex returned no answer for question '${questionName}'`,
        undefined,
        requestId,
      );
    }
    let raw: number;
    let figures: Record<string, unknown>;
    try {
      [raw, figures] = normalize(question, answer);
    } catch {
      throw new DrexError(
        `Drex returned a malformed answer for question '${questionName}'`,
        undefined,
        requestId,
      );
    }
    const value = clamp(raw);
    return {
      name: singleName ?? questionName,
      score: threshold === undefined ? value : value >= threshold ? 1 : 0,
      metadata: {
        question: questionName,
        type: question.type,
        direction: "higher is better (probability of a pass)",
        ...figures,
        raw_value: value,
        threshold: threshold ?? null,
        model: parsed.model ?? null,
        requested_model: model,
        request_id: requestId,
        evaluation_time_ms: parsed.evaluation_time_ms ?? null,
        // One request serves every question; count its tokens once, on the first score.
        usage: index === 0 ? usage : null,
        usage_counted_on: names[0],
      },
    };
  });
}

/**
 * Score an evaluation case with a Drex decision model.
 *
 * @example
 * ```ts
 * const grounded = DrexClassifier.partial({
 *   question: drexQuestion.noul("Is every claim in the output supported by the context?"),
 *   name: "Grounded",
 * });
 * const result = await grounded({ output: "Paris is the capital of France.", context: "..." });
 * ```
 */
export const DrexClassifier: ScorerWithPartial<unknown, DrexArgs> = makePartial(
  async (args) => {
    const name = args.name ?? "DrexClassifier";
    const [score] = await run(args, { [name]: args.question }, name);
    return score;
  },
  "DrexClassifier",
);

/** Answer several questions about one case in a single request; one `Score` per question, in order. */
export async function DrexClassifierMany(args: DrexManyArgs): Promise<Score[]> {
  return run(args, args.questions);
}
