"""Thread utilities for LLM-as-a-judge scorers.

This module provides helpers for deriving template variables from conversation
threads and formatting thread values for Mustache rendering.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

THREAD_VARIABLE_NAMES = [
    "thread",
    "thread_count",
    "first_message",
    "last_message",
    "user_messages",
    "assistant_messages",
    "human_ai_pairs",
]

# Match variables after "{{" or "{%" (e.g., {{thread}}, {{ thread }}, {% if thread %}).
THREAD_VARIABLE_PATTERN = re.compile(r"\{[\{%]\s*(" + "|".join(THREAD_VARIABLE_NAMES) + r")")


def template_uses_thread_variables(template: str) -> bool:
    return bool(THREAD_VARIABLE_PATTERN.search(template))


def is_role_content_message(item: Any) -> bool:
    return isinstance(item, Mapping) and "role" in item and "content" in item


def is_llm_message_array(value: Any) -> bool:
    return isinstance(value, list) and all(is_role_content_message(item) for item in value)


def filter_system_messages_from_thread(thread: list[Any]) -> list[Any]:
    return [message for message in thread if not (isinstance(message, Mapping) and message.get("role") == "system")]


def _indent(text: str, prefix: str = "  ") -> str:
    return "\n".join(prefix + line for line in text.split("\n"))


def _truncate_middle(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    chars_removed = len(text) - max_len + 30
    ellipsis = f" [...{chars_removed} chars truncated...] "
    avail = max_len - len(ellipsis)
    if avail <= 0:
        return text[:max_len]
    left = avail // 2
    right = avail - left
    return text[:left] + ellipsis + text[-right:]


def _is_typed_part(part: Any) -> bool:
    return isinstance(part, Mapping) and isinstance(part.get("type"), str)


@dataclass
class _PendingToolCall:
    name: str
    args: str


def _extract_tool_calls(content: list[Any]) -> dict[str, _PendingToolCall]:
    tool_calls: dict[str, _PendingToolCall] = {}
    for part in content:
        if not _is_typed_part(part) or part["type"] != "tool_call":
            continue

        tool_call_id = part.get("tool_call_id")
        if not isinstance(tool_call_id, str) or not tool_call_id:
            continue

        tool_name = part.get("tool_name") if isinstance(part.get("tool_name"), str) else "unknown"

        args = ""
        args_obj = part.get("arguments")
        if isinstance(args_obj, Mapping):
            if args_obj.get("type") == "valid":
                args = json.dumps(args_obj.get("value"))
            else:
                value = args_obj.get("value")
                if isinstance(value, str):
                    args = value
                else:
                    args = json.dumps(value)

        tool_calls[tool_call_id] = _PendingToolCall(name=tool_name, args=args)

    return tool_calls


def _unwrap_content(content: Any) -> str:
    if isinstance(content, str):
        try:
            return _unwrap_content(json.loads(content))
        except Exception:
            error_match = re.match(r"^error:\s*'(.+)'$", content, flags=re.DOTALL)
            if error_match:
                return error_match.group(1)
            return content

    if isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            if isinstance(item, Mapping) and isinstance(item.get("text"), str):
                text_parts.append(_unwrap_content(item["text"]))
            elif isinstance(item, str):
                text_parts.append(_unwrap_content(item))
        if text_parts:
            return "\n".join(text_parts)

    if isinstance(content, Mapping) and isinstance(content.get("text"), str):
        return _unwrap_content(content["text"])

    return content if isinstance(content, str) else json.dumps(content)


def _format_tool_result(
    tool_call_id: str,
    tool_name: str,
    output: Any,
    pending_tool_calls: dict[str, _PendingToolCall],
) -> str:
    pending_call = pending_tool_calls.get(tool_call_id)
    name = tool_name or (pending_call.name if pending_call else "tool")
    args = pending_call.args if pending_call else ""

    result_content = _unwrap_content(output)
    lines = [f"Tool ({name}):"]

    if args:
        lines.append("  Args:")
        lines.append(f"    {_truncate_middle(args, 500)}")

    is_error = (
        "error:" in result_content.lower()
        or '"error"' in result_content.lower()
        or result_content.lower().startswith("error")
    )
    if is_error:
        lines.append("  Error:")
        lines.append(f"    {_truncate_middle(result_content, 500)}")
    else:
        lines.append("  Result:")
        lines.append(f"    {_truncate_middle(result_content, 500)}")

    if pending_call:
        del pending_tool_calls[tool_call_id]

    return "\n".join(lines)


def _format_tool_results(content: list[Any], pending_tool_calls: dict[str, _PendingToolCall]) -> list[str]:
    results: list[str] = []
    for part in content:
        if not _is_typed_part(part) or part["type"] != "tool_result":
            continue

        tool_call_id = part.get("tool_call_id") if isinstance(part.get("tool_call_id"), str) else ""
        tool_name = part.get("tool_name") if isinstance(part.get("tool_name"), str) else ""
        results.append(_format_tool_result(tool_call_id, tool_name, part.get("output"), pending_tool_calls))

    return results


def _extract_text_content(content: Any) -> str:
    if isinstance(content, str):
        return content if content.strip() else ""
    if not isinstance(content, list):
        return ""

    parts: list[str] = []
    for part in content:
        if isinstance(part, str) and part.strip():
            parts.append(part)
        elif _is_typed_part(part):
            if part["type"] == "text" and isinstance(part.get("text"), str):
                parts.append(part["text"])
            elif part["type"] == "reasoning" and isinstance(part.get("text"), str):
                parts.append(f"[thinking: {part['text'][:100]}...]")
        elif isinstance(part, Mapping) and isinstance(part.get("text"), str):
            parts.append(part["text"])

    return "\n".join(parts)


def format_message_array_as_text(messages: list[Mapping[str, Any]]) -> str:
    pending_tool_calls: dict[str, _PendingToolCall] = {}
    for msg in messages:
        if msg.get("role") == "assistant" and isinstance(msg.get("content"), list):
            for tool_call_id, pending in _extract_tool_calls(msg["content"]).items():
                pending_tool_calls[tool_call_id] = pending

    parts: list[str] = []
    for msg in messages:
        role = msg.get("role", "")
        if not isinstance(role, str):
            role = str(role)
        capitalized_role = role[:1].upper() + role[1:]
        content = msg.get("content")

        if role == "tool" and isinstance(content, list):
            parts.extend(_format_tool_results(content, pending_tool_calls))
        else:
            text = _extract_text_content(content)
            if text:
                parts.append(f"{capitalized_role}:\n{_indent(text)}")

    return "\n\n".join(parts)


class RenderableMessage(dict[str, Any]):
    def __str__(self) -> str:
        role = self.get("role", "")
        content = self.get("content")
        content_str = content if isinstance(content, str) else json.dumps(content)
        return f"{role}: {content_str}"


class RenderableMessageArray(list[Any]):
    def __str__(self) -> str:
        return format_message_array_as_text(self)


def _to_renderable_message(message: Mapping[str, Any]) -> RenderableMessage:
    return RenderableMessage(message)


def _to_renderable_message_array(messages: list[Any]) -> RenderableMessageArray:
    wrapped: list[Any] = []
    for message in messages:
        if is_role_content_message(message):
            wrapped.append(_to_renderable_message(message))
        else:
            wrapped.append(message)
    return RenderableMessageArray(wrapped)


def compute_thread_template_vars(thread: list[Any]) -> dict[str, Any]:
    renderable_thread = _to_renderable_message_array(thread) if is_llm_message_array(thread) else thread

    first_message = renderable_thread[0] if len(renderable_thread) > 0 else None
    last_message = renderable_thread[-1] if len(renderable_thread) > 0 else None

    user_messages = [
        message for message in renderable_thread if is_role_content_message(message) and message.get("role") == "user"
    ]
    assistant_messages = [
        message
        for message in renderable_thread
        if is_role_content_message(message) and message.get("role") == "assistant"
    ]
    pair_count = min(len(user_messages), len(assistant_messages))
    human_ai_pairs = [{"human": user_messages[idx], "assistant": assistant_messages[idx]} for idx in range(pair_count)]

    return {
        "thread": renderable_thread,
        "thread_count": len(thread),
        "first_message": first_message,
        "last_message": last_message,
        "user_messages": _to_renderable_message_array(user_messages),
        "assistant_messages": _to_renderable_message_array(assistant_messages),
        "human_ai_pairs": human_ai_pairs,
    }
