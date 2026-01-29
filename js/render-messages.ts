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

export function renderMessages(
  messages: ChatCompletionMessageParam[],
  renderArgs: Record<string, unknown>,
): ChatCompletionMessageParam[] {
  return messages.map((m) => ({
    ...m,
    content: m.content
      ? mustache.render(m.content as string, renderArgs, undefined, {
          escape: escapeValue,
        })
      : "",
  }));
}
