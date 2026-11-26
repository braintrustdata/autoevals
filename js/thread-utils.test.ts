import { describe, expect, it } from "vitest";
import { readFileSync } from "fs";
import { join } from "path";
import { renderMessages } from "./render-messages";
import { formatMessageArrayAsText } from "./thread-utils";

// Shared with py/autoevals/test_thread_utils.py so both languages render threads identically.
const FIXTURES: Record<
  string,
  { messages: any[]; text?: string; message_text?: string }
> = JSON.parse(
  readFileSync(
    join(__dirname, "..", "test-data", "thread_formatting.json"),
    "utf-8",
  ),
);

describe("thread formatting shared fixtures", () => {
  for (const [name, c] of Object.entries(FIXTURES)) {
    if (c.text !== undefined) {
      it(`formats a message array: ${name}`, () => {
        expect(formatMessageArrayAsText(c.messages)).toBe(c.text);
      });
    }
    if (c.message_text !== undefined) {
      it(`renders a single message: ${name}`, () => {
        const rendered = renderMessages(
          [{ role: "user", content: "{{message}}" }],
          { message: c.messages[0] },
        );
        expect(rendered[0].content).toBe(c.message_text);
      });
    }
  }
});
