import { promises as fs } from "node:fs";
import os from "node:os";
import path from "node:path";

import { http, HttpResponse } from "msw";
import { setupServer } from "msw/node";
import { OpenAI } from "openai";
import { afterAll, afterEach, beforeAll, describe, expect, test } from "vitest";

import { Behavior, discoverAgentBehaviors } from "./llm";

const server = setupServer();
const behaviorProjects = new Set<string>();

beforeAll(() => server.listen({ onUnhandledRequest: "error" }));
afterEach(async () => {
  server.resetHandlers();
  await Promise.all(
    [...behaviorProjects].map((root) =>
      fs.rm(root, { recursive: true, force: true }),
    ),
  );
  behaviorProjects.clear();
});
afterAll(() => server.close());

async function createBehaviorProject(name = "verify-work") {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), "autoevals-behavior-"));
  behaviorProjects.add(root);
  const directory = path.join(root, ".agents", "behaviors", name);
  await fs.mkdir(directory, { recursive: true });
  await fs.writeFile(
    path.join(directory, "BEHAVIOR.md"),
    `---\nname: ${name}\ndescription: Verify work before answering.\n---\n# Verify work\n\nThe agent MUST show its calculation.\n`,
  );
  return root;
}

describe("Behavior", () => {
  test("discovers a valid BEHAVIOR.md from a project root", async () => {
    const root = await createBehaviorProject();
    const behaviors = await discoverAgentBehaviors(root);

    expect(behaviors).toHaveLength(1);
    expect(behaviors[0]).toMatchObject({
      name: "verify-work",
      description: "Verify work before answering.",
    });
    expect(behaviors[0]?.location).toBe(
      path.join(root, ".agents", "behaviors", "verify-work", "BEHAVIOR.md"),
    );
  });

  test("judges structured output against an explicitly provided behavior", async () => {
    let prompt = "";
    let choice: "true" | "na" = "true";
    server.use(
      http.post(
        "https://api.openai.com/v1/chat/completions",
        async ({ request }) => {
          const body = (await request.json()) as {
            messages: Array<{ content: string }>;
          };
          prompt = body.messages[0]!.content;
          return HttpResponse.json({
            id: "chatcmpl-behavior",
            object: "chat.completion",
            created: 0,
            model: "gpt-4o-mini",
            choices: [
              {
                index: 0,
                finish_reason: "tool_calls",
                message: {
                  role: "assistant",
                  content: null,
                  tool_calls: [
                    {
                      id: "call-behavior",
                      type: "function",
                      function: {
                        name: "select_choice",
                        arguments: JSON.stringify({
                          choice,
                          reasons: "The calculation is visible.",
                        }),
                      },
                    },
                  ],
                },
              },
            ],
            usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
          });
        },
      ),
    );

    const result = await Behavior({
      behavior: {
        name: "verify-work",
        description: "Verify work before answering.",
        body: "# Verify work\n\nThe agent MUST show its calculation.",
      },
      input: { question: "What is 2 + 2?" },
      output: { events: [{ type: "answer", content: "2 + 2 = 4" }] },
      trace: {
        getThread: async () => [
          { role: "system", content: "Use the calculator." },
          { role: "user", content: "What is 2 + 2?" },
          { role: "assistant", content: "2 + 2 = 4" },
        ],
      },
      model: "gpt-4o-mini",
      client: new OpenAI({
        apiKey: "test",
        baseURL: "https://api.openai.com/v1",
      }),
    });

    expect(result.score).toBe(1);
    expect(result.metadata?.choice).toBe("true");
    expect(result.metadata?.behavior).toMatchObject({ name: "verify-work" });
    expect(prompt).toContain("The agent MUST show its calculation.");
    expect(prompt).toContain('"question":"What is 2 + 2?"');
    expect(prompt).toContain("System:\n  Use the calculator.");

    choice = "na";
    const notApplicable = await Behavior({
      behavior: {
        name: "verify-work",
        description: "Verify work before answering.",
        body: "# Verify work\n\nThe agent MUST show its calculation.",
      },
      output: "No calculation was requested.",
      model: "gpt-4o-mini",
      client: new OpenAI({
        apiKey: "test",
        baseURL: "https://api.openai.com/v1",
      }),
    });
    expect(notApplicable.score).toBeNull();

    const root = await createBehaviorProject();
    await fs.writeFile(
      path.join(root, "verify-work"),
      "unrelated project file",
    );
    choice = "true";
    const namedBehavior = await Behavior({
      behavior: "verify-work",
      behaviorRoot: root,
      output: "2 + 2 = 4",
      model: "gpt-4o-mini",
      client: new OpenAI({
        apiKey: "test",
        baseURL: "https://api.openai.com/v1",
      }),
    });
    expect(namedBehavior.metadata?.behavior).toMatchObject({
      name: "verify-work",
    });
  });

  test("includes invalid spec diagnostics when discovery finds no valid behavior", async () => {
    const root = await createBehaviorProject();
    const behaviorFile = path.join(
      root,
      ".agents",
      "behaviors",
      "verify-work",
      "BEHAVIOR.md",
    );
    await fs.writeFile(
      behaviorFile,
      "---\nname: INVALID\ndescription: Invalid behavior.\n---\n# Invalid\n",
    );

    await expect(
      Behavior({ output: "done", behaviorRoot: root, model: "gpt-4o-mini" }),
    ).rejects.toThrow("Diagnostics: Agent Behavior name");
  });

  test("requires an explicit selection when discovery finds multiple behaviors", async () => {
    const root = await createBehaviorProject("first-behavior");
    const second = path.join(root, ".agents", "behaviors", "second-behavior");
    await fs.mkdir(second, { recursive: true });
    await fs.writeFile(
      path.join(second, "BEHAVIOR.md"),
      "---\nname: second-behavior\ndescription: A second behavior.\n---\n# Second\n",
    );

    await expect(
      Behavior({ output: "done", behaviorRoot: root, model: "gpt-4o-mini" }),
    ).rejects.toThrow("Multiple Agent Behavior specs were discovered");
  });
});
