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

  it("should stringify objects when using either {{...}} or {{{...}}}", () => {
    const messages: ChatCompletionMessageParam[] = [
      {
        role: "user",
        content: "Double braces: {{data}}, Triple braces: {{{data}}}",
      },
    ];
    const data = { foo: "bar", num: 42 };
    const rendered = renderMessages(messages, { data });
    const stringified = JSON.stringify(data);
    expect(rendered[0].content).toBe(
      `Double braces: ${stringified}, Triple braces: ${stringified}`,
    );
  });

  it("should handle empty content", () => {
    const messages: ChatCompletionMessageParam[] = [
      { role: "user", content: "" },
    ];
    const rendered = renderMessages(messages, {});
    expect(rendered[0].content).toBe("");
  });
});
