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

describe("renderMessages with thread variables", () => {
  const sampleThread = [
    { role: "user", content: "Hello, how are you?" },
    { role: "assistant", content: "I am doing well, thank you!" },
    { role: "user", content: "What is the weather like?" },
    { role: "assistant", content: "It is sunny and warm today." },
  ];

  it("{{thread}} renders full conversation as human-readable text", () => {
    const messages: ChatCompletionMessageParam[] = [
      { role: "user", content: "{{thread}}" },
    ];
    const rendered = renderMessages(messages, { thread: sampleThread });

    expect(rendered[0].content).toContain("User:");
    expect(rendered[0].content).toContain("Hello, how are you?");
    expect(rendered[0].content).toContain("Assistant:");
    expect(rendered[0].content).toContain("I am doing well, thank you!");
    expect(rendered[0].content).toContain("What is the weather like?");
    expect(rendered[0].content).toContain("It is sunny and warm today.");
  });

  it("{{thread.0}} renders first message as formatted text", () => {
    const messages: ChatCompletionMessageParam[] = [
      { role: "user", content: "First message: {{thread.0}}" },
    ];
    const rendered = renderMessages(messages, { thread: sampleThread });

    expect(rendered[0].content).toBe(
      "First message: user: Hello, how are you?",
    );
  });

  it("{{thread.1}} renders second message as formatted text", () => {
    const messages: ChatCompletionMessageParam[] = [
      { role: "user", content: "Second message: {{thread.1}}" },
    ];
    const rendered = renderMessages(messages, { thread: sampleThread });

    expect(rendered[0].content).toBe(
      "Second message: assistant: I am doing well, thank you!",
    );
  });

  it("{{first_message}} renders single message formatted", () => {
    const messages: ChatCompletionMessageParam[] = [
      { role: "user", content: "First: {{first_message}}" },
    ];
    const rendered = renderMessages(messages, {
      first_message: sampleThread[0],
    });

    expect(rendered[0].content).toBe("First: user: Hello, how are you?");
  });

  it("{{thread_count}} renders as a number", () => {
    const messages: ChatCompletionMessageParam[] = [
      { role: "user", content: "Count: {{thread_count}}" },
    ];
    const rendered = renderMessages(messages, { thread_count: 4 });

    expect(rendered[0].content).toBe("Count: 4");
  });

  it("{{user_messages}} renders array of user messages", () => {
    const messages: ChatCompletionMessageParam[] = [
      { role: "user", content: "Users said: {{user_messages}}" },
    ];
    const userMessages = sampleThread.filter((m) => m.role === "user");
    const rendered = renderMessages(messages, { user_messages: userMessages });

    expect(rendered[0].content).toContain("User:");
    expect(rendered[0].content).toContain("Hello, how are you?");
    expect(rendered[0].content).toContain("What is the weather like?");
    expect(rendered[0].content).not.toContain("Assistant:");
  });

  it("{{user_messages.0}} renders first user message", () => {
    const messages: ChatCompletionMessageParam[] = [
      { role: "user", content: "First user: {{user_messages.0}}" },
    ];
    const userMessages = sampleThread.filter((m) => m.role === "user");
    const rendered = renderMessages(messages, { user_messages: userMessages });

    expect(rendered[0].content).toBe("First user: user: Hello, how are you?");
  });

  it("{{human_ai_pairs}} renders array of paired turns", () => {
    const messages: ChatCompletionMessageParam[] = [
      { role: "user", content: "Pairs: {{human_ai_pairs}}" },
    ];
    const pairs = [
      { human: sampleThread[0], assistant: sampleThread[1] },
      { human: sampleThread[2], assistant: sampleThread[3] },
    ];
    const rendered = renderMessages(messages, { human_ai_pairs: pairs });

    // Pairs are objects, so they get JSON stringified
    expect(rendered[0].content).toContain("Pairs:");
    expect(rendered[0].content).toContain("human");
    expect(rendered[0].content).toContain("assistant");
  });

  it("{{#thread}}...{{/thread}} iterates over messages", () => {
    const messages: ChatCompletionMessageParam[] = [
      {
        role: "user",
        content: "Messages:{{#thread}}\n- {{role}}: {{content}}{{/thread}}",
      },
    ];
    const rendered = renderMessages(messages, { thread: sampleThread });

    expect(rendered[0].content).toBe(
      "Messages:\n- user: Hello, how are you?\n- assistant: I am doing well, thank you!\n- user: What is the weather like?\n- assistant: It is sunny and warm today.",
    );
  });

  it("handles empty thread gracefully", () => {
    const messages: ChatCompletionMessageParam[] = [
      { role: "user", content: "Thread: {{thread}}" },
    ];
    const rendered = renderMessages(messages, { thread: [] });

    expect(rendered[0].content).toBe("Thread: ");
  });

  it("handles thread with complex content (arrays)", () => {
    const complexThread = [
      {
        role: "user",
        content: [{ type: "text", text: "Hello with structured content" }],
      },
      { role: "assistant", content: "Simple response" },
    ];
    const messages: ChatCompletionMessageParam[] = [
      { role: "user", content: "{{thread}}" },
    ];
    const rendered = renderMessages(messages, { thread: complexThread });

    expect(rendered[0].content).toContain("User:");
    expect(rendered[0].content).toContain("Hello with structured content");
    expect(rendered[0].content).toContain("Assistant:");
    expect(rendered[0].content).toContain("Simple response");
  });
});
