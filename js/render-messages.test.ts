import { describe, expect, it } from "vitest";
import { renderMessages } from "./render-messages";
import { ChatCompletionMessageParam } from "openai/resources";

describe("renderMessages", () => {
  it("should never HTML-escape values, regardless of mustache syntax", () => {
    const messages: ChatCompletionMessageParam[] = [
      { role: "user", content: "{{value}} and {{{value}}}" },
    ];
    const rendered = renderMessages(messages, { value: "<b>bold</b>" });
    expect(rendered[0].content).toBe("<b>bold</b> and <b>bold</b>");
  });

  it("should stringify objects when using {{...}}", () => {
    const messages: ChatCompletionMessageParam[] = [
      { role: "user", content: "Data: {{data}}" },
    ];
    const data = { foo: "bar", num: 42 };
    const rendered = renderMessages(messages, { data });
    expect(rendered[0].content).toBe('Data: {"foo":"bar","num":42}');
  });

  it("should output [object Object] when using {{{...}}} with objects", () => {
    const messages: ChatCompletionMessageParam[] = [
      { role: "user", content: "Data: {{{data}}}" },
    ];
    const data = { foo: "bar", num: 42 };
    const rendered = renderMessages(messages, { data });
    expect(rendered[0].content).toBe("Data: [object Object]");
  });

  it("should handle empty content", () => {
    const messages: ChatCompletionMessageParam[] = [
      { role: "user", content: "" },
    ];
    const rendered = renderMessages(messages, {});
    expect(rendered[0].content).toBe("");
  });
});
