import mustache from "mustache";
import { ChatCompletionMessageParam } from "openai/resources";
import {
  isLLMMessageArray,
  isRoleContentMessage,
  formatMessageArrayAsText,
} from "./thread-utils";

/**
 * Smart escape function for Mustache templates.
 * - Strings are passed through unchanged
 * - LLM message arrays are formatted as human-readable text
 * - Single messages are formatted with role and content
 * - Other values are JSON-stringified
 */
function escapeValue(v: unknown): string {
  if (typeof v === "string") {
    return v;
  }
  if (isLLMMessageArray(v)) {
    return formatMessageArrayAsText(v);
  }
  if (isRoleContentMessage(v)) {
    const content =
      typeof v.content === "string" ? v.content : JSON.stringify(v.content);
    return `${v.role}: ${content}`;
  }
  return JSON.stringify(v);
}

function explainMustacheError(error: unknown, template: string): unknown {
  if (!(error instanceof Error)) {
    return error;
  }
  const match = error.message.match(/at (\d+)$/);
  if (!match) {
    return error;
  }
  const offset = Number(match[1]);
  const before = template.slice(0, offset);
  const line = before.split("\n").length;
  const column = offset - before.lastIndexOf("\n");
  const snippet = template.slice(Math.max(0, offset - 30), offset + 10);
  return new Error(
    `${error.message} (line ${line}, column ${column}, near "…${snippet}…"). ` +
      "Check for an unclosed {{ }} tag or {{#section}} block in the template.",
  );
}

export function renderMessages(
  messages: ChatCompletionMessageParam[],
  renderArgs: Record<string, unknown>,
): ChatCompletionMessageParam[] {
  return messages.map((m) => {
    if (!m.content) {
      return { ...m, content: "" };
    }
    try {
      return {
        ...m,
        content: mustache.render(m.content as string, renderArgs, undefined, {
          escape: escapeValue,
        }),
      };
    } catch (e) {
      throw explainMustacheError(e, String(m.content));
    }
  });
}
