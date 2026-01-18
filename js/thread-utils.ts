/**
 * Thread utilities for LLM-as-a-judge scorers.
 *
 * This module provides utilities for working with preprocessed conversation
 * messages (threads) in LLM scorer templates.
 */

/**
 * A message with role and content fields (LLM chat message format).
 */
export interface LLMMessage {
  role: string;
  content: unknown;
}

function isObject(value: unknown): value is { [key: string]: unknown } {
  return value instanceof Object && !(value instanceof Array);
}

/**
 * Check if an item looks like an LLM message (has role and content).
 */
export function isRoleContentMessage(item: unknown): item is LLMMessage {
  return isObject(item) && "role" in item && "content" in item;
}

/**
 * Check if a value is an array of LLM messages.
 */
export function isLLMMessageArray(value: unknown): value is LLMMessage[] {
  return Array.isArray(value) && value.every(isRoleContentMessage);
}

function indent(text: string, prefix = "  "): string {
  return text
    .split("\n")
    .map((line) => (line ? prefix + line : prefix))
    .join("\n");
}

function truncateMiddle(text: string, maxLen: number): string {
  if (text.length <= maxLen) return text;
  const charsRemoved = text.length - maxLen + 30;
  const ellipsis = ` [...${charsRemoved} chars truncated...] `;
  const avail = maxLen - ellipsis.length;
  if (avail <= 0) return text.slice(0, maxLen);
  const left = Math.floor(avail / 2);
  const right = avail - left;
  return text.slice(0, left) + ellipsis + text.slice(-right);
}

interface PendingToolCall {
  name: string;
  args: string;
}

function isTypedPart(
  part: unknown,
): part is { type: string; [key: string]: unknown } {
  return isObject(part) && typeof part.type === "string";
}

function extractToolCalls(content: unknown[]): Map<string, PendingToolCall> {
  const toolCalls = new Map<string, PendingToolCall>();

  for (const part of content) {
    if (!isTypedPart(part) || part.type !== "tool_call") continue;

    const id = typeof part.tool_call_id === "string" ? part.tool_call_id : "";
    if (!id) continue;

    const name =
      typeof part.tool_name === "string" ? part.tool_name : "unknown";

    let args = "";
    if (isObject(part.arguments)) {
      const argsObj = part.arguments;
      if (argsObj.type === "valid") {
        args = JSON.stringify(argsObj.value);
      } else if (typeof argsObj.value === "string") {
        args = argsObj.value;
      } else {
        args = JSON.stringify(argsObj.value);
      }
    }

    toolCalls.set(id, { name, args });
  }

  return toolCalls;
}

function unwrapContent(content: unknown): string {
  if (typeof content === "string") {
    try {
      const parsed = JSON.parse(content);
      return unwrapContent(parsed);
    } catch {
      const errorMatch = content.match(/^error:\s*'(.+)'$/s);
      if (errorMatch) {
        return errorMatch[1];
      }
      return content;
    }
  }

  if (Array.isArray(content)) {
    const textParts: string[] = [];
    for (const item of content) {
      if (isObject(item) && typeof item.text === "string") {
        textParts.push(unwrapContent(item.text));
      } else if (typeof item === "string") {
        textParts.push(unwrapContent(item));
      }
    }
    if (textParts.length > 0) {
      return textParts.join("\n");
    }
  }

  if (isObject(content) && typeof content.text === "string") {
    return unwrapContent(content.text);
  }

  return typeof content === "string" ? content : JSON.stringify(content);
}

function formatToolResult(
  toolCallId: string,
  toolName: string,
  output: unknown,
  pendingToolCalls: Map<string, PendingToolCall>,
): string {
  const pendingCall = pendingToolCalls.get(toolCallId);
  const name = toolName || pendingCall?.name || "tool";
  const args = pendingCall?.args || "";

  const resultContent = unwrapContent(output);
  const lines = [`Tool (${name}):`];

  if (args) {
    lines.push(`  Args:`);
    lines.push(`    ${truncateMiddle(args, 500)}`);
  }

  const isError =
    resultContent.toLowerCase().includes("error:") ||
    resultContent.toLowerCase().includes('"error"') ||
    resultContent.toLowerCase().startsWith("error");

  if (isError) {
    lines.push(`  Error:`);
    lines.push(`    ${truncateMiddle(resultContent, 500)}`);
  } else {
    lines.push(`  Result:`);
    lines.push(`    ${truncateMiddle(resultContent, 500)}`);
  }

  if (pendingCall) {
    pendingToolCalls.delete(toolCallId);
  }

  return lines.join("\n");
}

function formatToolResults(
  content: unknown[],
  pendingToolCalls: Map<string, PendingToolCall>,
): string[] {
  const results: string[] = [];

  for (const part of content) {
    if (!isTypedPart(part) || part.type !== "tool_result") continue;

    const toolCallId =
      typeof part.tool_call_id === "string" ? part.tool_call_id : "";
    const toolName = typeof part.tool_name === "string" ? part.tool_name : "";

    results.push(
      formatToolResult(toolCallId, toolName, part.output, pendingToolCalls),
    );
  }

  return results;
}

function extractTextContent(content: unknown): string {
  if (typeof content === "string") {
    return content.trim() ? content : "";
  }

  if (!Array.isArray(content)) {
    return "";
  }

  const parts: string[] = [];
  for (const part of content) {
    if (typeof part === "string" && part.trim()) {
      parts.push(part);
    } else if (isTypedPart(part)) {
      if (part.type === "text" && typeof part.text === "string") {
        parts.push(part.text);
      } else if (part.type === "reasoning" && typeof part.text === "string") {
        parts.push(`[thinking: ${part.text.slice(0, 100)}...]`);
      }
    } else if (isObject(part) && typeof part.text === "string") {
      parts.push(part.text);
    }
  }

  return parts.join("\n");
}

/**
 * Format an array of LLM messages as human-readable text.
 */
export function formatMessageArrayAsText(messages: LLMMessage[]): string {
  const pendingToolCalls = new Map<string, PendingToolCall>();
  for (const msg of messages) {
    if (msg.role === "assistant" && Array.isArray(msg.content)) {
      const calls = extractToolCalls(msg.content);
      for (const [id, call] of calls) {
        pendingToolCalls.set(id, call);
      }
    }
  }

  const parts: string[] = [];
  for (const msg of messages) {
    const role = msg.role;
    const capitalizedRole = role.charAt(0).toUpperCase() + role.slice(1);

    if (role === "tool" && Array.isArray(msg.content)) {
      const toolResults = formatToolResults(msg.content, pendingToolCalls);
      parts.push(...toolResults);
    } else {
      const text = extractTextContent(msg.content);
      if (text) {
        parts.push(`${capitalizedRole}:\n${indent(text)}`);
      }
    }
  }

  return parts.join("\n\n");
}

/**
 * Template variables computed from a thread for use in LLM-as-a-judge scorers.
 */
export interface ThreadTemplateVars {
  thread: unknown[];
  thread_text: string;
  thread_count: number;
  first_message: unknown | null;
  last_message: unknown | null;
  user_messages: unknown[];
  assistant_messages: unknown[];
  human_ai_pairs: Array<{ human: unknown; assistant: unknown }>;
}

/**
 * Compute template variables from a thread for use in mustache templates.
 * Uses lazy getters so expensive computations only run when accessed.
 */
export function computeThreadTemplateVars(
  thread: unknown[],
): ThreadTemplateVars {
  let _thread_text: string | undefined;
  let _user_messages: unknown[] | undefined;
  let _assistant_messages: unknown[] | undefined;
  let _human_ai_pairs:
    | Array<{ human: unknown; assistant: unknown }>
    | undefined;

  return {
    thread,
    thread_count: thread.length,

    get thread_text(): string {
      if (_thread_text === undefined) {
        if (isLLMMessageArray(thread)) {
          _thread_text = formatMessageArrayAsText(thread);
        } else {
          _thread_text = thread
            .map((item) =>
              typeof item === "string" ? item : JSON.stringify(item),
            )
            .join("\n");
        }
      }
      return _thread_text;
    },

    get first_message(): unknown | null {
      return thread[0] ?? null;
    },

    get last_message(): unknown | null {
      return thread[thread.length - 1] ?? null;
    },

    get user_messages(): unknown[] {
      if (_user_messages === undefined) {
        _user_messages = thread.filter(
          (m) => isRoleContentMessage(m) && m.role === "user",
        );
      }
      return _user_messages;
    },

    get assistant_messages(): unknown[] {
      if (_assistant_messages === undefined) {
        _assistant_messages = thread.filter(
          (m) => isRoleContentMessage(m) && m.role === "assistant",
        );
      }
      return _assistant_messages;
    },

    get human_ai_pairs(): Array<{ human: unknown; assistant: unknown }> {
      if (_human_ai_pairs === undefined) {
        _human_ai_pairs = [];
        const users = thread.filter(
          (m) => isRoleContentMessage(m) && m.role === "user",
        );
        const assistants = thread.filter(
          (m) => isRoleContentMessage(m) && m.role === "assistant",
        );
        const pairCount = Math.min(users.length, assistants.length);
        for (let i = 0; i < pairCount; i++) {
          _human_ai_pairs.push({ human: users[i], assistant: assistants[i] });
        }
      }
      return _human_ai_pairs;
    },
  };
}
