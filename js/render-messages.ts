import mustache from "mustache";
import { ChatCompletionMessageParam } from "openai/resources";

export function renderMessages(
  messages: ChatCompletionMessageParam[],
  renderArgs: Record<string, unknown>,
): ChatCompletionMessageParam[] {
  return messages.map((m) => ({
    ...m,
    content: m.content
      ? mustache.render(
          (m.content as string).replace(/\{{3}/g, "{{").replace(/\}{3}/g, "}}"),
          renderArgs,
          undefined,
          {
            escape: (v: unknown) =>
              typeof v === "string" ? v : JSON.stringify(v),
          },
        )
      : "",
  }));
}
