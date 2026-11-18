import { expect, test } from "vitest";
import { LLMClassifierFromTemplate } from "./llm";
import { renderMessages } from "./render-messages";

const recording = { id: "r", mime_type: "audio/ogg", data: "AAAA" };
const classifier = LLMClassifierFromTemplate({
  name: "recording",
  promptTemplate: "Listen: {{recording}}",
  choiceScores: { Yes: 1, No: 0 },
});

test("renders {{recording}} as a file part", () => {
  const [message] = renderMessages(
    [{ role: "user", content: "Listen: {{recording}} Done" }],
    { recording },
  );
  expect(message.content).toEqual([
    { type: "text", text: "Listen: " },
    {
      type: "file",
      file: { filename: "recording", file_data: "data:audio/ogg;base64,AAAA" },
    },
    { type: "text", text: " Done" },
  ]);
});

test("throws without trace.getRecording()", async () => {
  await expect(classifier({ output: "" })).rejects.toThrow(
    "{{recording}} requires trace.getRecording()",
  );
});

test("skips without calling the model when there is no recording", async () => {
  const client = {
    chat: {
      completions: {
        create: () => {
          throw new Error("called the model");
        },
      },
    },
  };
  const trace = { getThread: async () => [], getRecording: async () => null };
  expect(await classifier({ output: "", trace, client } as any)).toEqual({
    name: "recording",
    score: null,
    metadata: { recording_empty: true },
  });
});
