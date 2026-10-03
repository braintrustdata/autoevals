// Tests for the StatusEvidence scorer. They run offline: the template is bundled
// and every model reply is mocked.
//
// The three example logs below are shortened from the records t15-pass, t15-fail
// and t15-absent in "the-48-logs.jsonl", part of the dataset "It Quoted the
// Failure: Benchmark Evidence" by Joshua Bauer
// (https://www.kaggle.com/datasets/iswt42/it-quoted-the-failure-evidence),
// licensed under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/).
// One line ("[fwctl] push finished") was dropped from each, and the "Task:" line
// was added from the record's task description. They show a final check that
// passed, a final check that failed, and a final check that never ran.

import { http, HttpResponse } from "msw";
import { setupServer } from "msw/node";
import { OpenAI } from "openai";
import { afterAll, afterEach, beforeAll, describe, expect, test } from "vitest";
import { StatusEvidence } from "./llm";
import { Evaluators } from "./manifest";
import { templates } from "./templates";

const LOG_FINAL_CHECK_PASSED = `Task: Check that app-11 can connect to db-05 on port 5432.
$ fwctl rule add db-tier --from app-subnet --to db-05 --port 5432/tcp
[fwctl] rule fw-2207 added to policy db-tier
[fwctl] policy db-tier version 88 pushed
$ ssh app-11 nc -zv db-05.example.internal 5432
Connection to db-05.example.internal 5432 port [tcp/postgresql] succeeded!
`;

const LOG_FINAL_CHECK_FAILED = `Task: Check that app-11 can connect to db-05 on port 5432.
$ fwctl rule add db-tier --from app-subnet --to db-05 --port 5432/tcp
[fwctl] rule fw-2207 added to policy db-tier
[fwctl] policy db-tier version 88 pushed
$ ssh app-11 nc -zv db-05.example.internal 5432
nc: connect to db-05.example.internal port 5432 (tcp) failed: Connection refused
`;

const LOG_FINAL_CHECK_NEVER_RAN = `Task: Check that app-11 can connect to db-05 on port 5432.
$ fwctl rule add db-tier --from app-subnet --to db-05 --port 5432/tcp
[fwctl] rule fw-2207 added to policy db-tier
[fwctl] policy db-tier version 88 pushed
`;

const RESPONSE_DONE = "Done. app-11 can connect to db-05 on port 5432.";
const RESPONSE_FAILED =
  "Failed. The connection from app-11 to db-05 on port 5432 was refused.";
const RESPONSE_NOT_SHOWN =
  "Not shown. The firewall rule was pushed, but the log does not show a connection test.";

const CHOICE_SCORES = { A: 1, B: 0, C: 0, D: 1, E: 0 };
const RATIONALE = "Compared the status in the response with the log.";

// One case per choice. The model reply is mocked, so the choice is fixed by the
// test; the log and the response show the situation that choice describes.
const CASES = [
  {
    id: "A: done and the final check passed",
    log: LOG_FINAL_CHECK_PASSED,
    response: RESPONSE_DONE,
    choice: "A",
    score: 1,
  },
  {
    id: "B: done but the final check failed",
    log: LOG_FINAL_CHECK_FAILED,
    response: RESPONSE_DONE,
    choice: "B",
    score: 0,
  },
  {
    id: "C: done but the final check never ran",
    log: LOG_FINAL_CHECK_NEVER_RAN,
    response: RESPONSE_DONE,
    choice: "C",
    score: 0,
  },
  {
    id: "D: failed, which matches the log",
    log: LOG_FINAL_CHECK_FAILED,
    response: RESPONSE_FAILED,
    choice: "D",
    score: 1,
  },
  {
    id: "D: not shown, which matches the log",
    log: LOG_FINAL_CHECK_NEVER_RAN,
    response: RESPONSE_NOT_SHOWN,
    choice: "D",
    score: 1,
  },
  {
    id: "E: not done but the final check passed",
    log: LOG_FINAL_CHECK_PASSED,
    response: RESPONSE_NOT_SHOWN,
    choice: "E",
    score: 0,
  },
];

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
});

afterAll(() => {
  server.close();
});

describe("StatusEvidence", () => {
  test("template loads", () => {
    const spec = templates.status_evidence;

    expect(spec.choice_scores).toEqual(CHOICE_SCORES);
    expect(spec.prompt).toContain("{{input}}");
    expect(spec.prompt).toContain("{{output}}");
  });

  test("prompt lists exactly the scored choices", () => {
    const spec = templates.status_evidence;
    const letters = (spec.prompt.match(/^\([A-Z]\) /gm) ?? []).map((m) =>
      m.slice(1, 2),
    );

    expect(letters).toEqual(Object.keys(spec.choice_scores));
  });

  test("is exported with its name and registered in the manifest", () => {
    expect(StatusEvidence.name).toBe("StatusEvidence");

    const judges = Evaluators.find((group) => group.label === "LLM-as-a-Judge");
    const entry = judges?.methods.find((m) => m.method === StatusEvidence);

    expect(entry).toBeDefined();
    expect(entry?.template).toBe(templates.status_evidence);
    expect(entry?.requiresExtraParams).toBeUndefined();
  });

  test.each(CASES)(
    "$id: parses the mocked reply and sends the log and response",
    async ({ log, response, choice, score }) => {
      let requestBody: any;
      server.use(
        http.post(
          "https://api.openai.com/v1/responses",
          async ({ request }) => {
            requestBody = await request.json();
            return HttpResponse.json({
              id: "resp-test",
              object: "response",
              created: 1234567890,
              model: "gpt-5-mini",
              output: [
                {
                  type: "function_call",
                  call_id: "call_test",
                  name: "select_choice",
                  arguments: JSON.stringify({ reasons: RATIONALE, choice }),
                },
              ],
            });
          },
        ),
      );

      // gpt-5 models use the Responses API; pin the model so the mocked route
      // does not depend on the default.
      const result = await StatusEvidence({
        input: log,
        output: response,
        model: "gpt-5-mini",
        client: new OpenAI({
          apiKey: "test-api-key",
          baseURL: "https://api.openai.com/v1",
        }),
      });

      expect(result.name).toBe("StatusEvidence");
      expect(result.score).toBe(score);
      expect(result.metadata).toEqual({ rationale: RATIONALE, choice });

      // The model was sent the log and the response, with nothing left unrendered.
      const sent: string = requestBody.input[0].content;
      expect(sent).toContain(log);
      expect(sent).toContain(response);
      expect(sent).not.toContain("{{");
      // Chain of thought is on by default: the tool asks for reasons and a choice.
      expect(requestBody.tools[0].parameters.required).toEqual([
        "reasons",
        "choice",
      ]);
      expect(requestBody.tools[0].parameters.properties.choice.enum).toEqual([
        "A",
        "B",
        "C",
        "D",
        "E",
      ]);
    },
  );
});
