"""Renders `trace.get_trajectory()` as plain text for LLM-as-a-judge templates.

`{{trajectory}}` omits the agent's instructions; `{{trajectory_with_system}}` includes them.
"""

import json
import re
from typing import Any


def template_uses_trajectory_variables(template: str) -> bool:
    return re.search(r"\{\{\s*trajectory(_with_system)?\s*\}\}", template) is not None


def compute_trajectory_template_vars(trajectories: list[Any]) -> dict[str, str]:
    return {
        "trajectory": _render(trajectories, with_system=False),
        "trajectory_with_system": _render(trajectories, with_system=True),
    }


# One block per trajectory, separated by a blank line.
def _render(trajectories: list[Any], with_system: bool) -> str:
    blocks = [_render_turns(t.get("turns") or [], with_system) for t in trajectories]
    return "\n\n".join(b for b in blocks if b)


def _render_turns(turns: list[Any], with_system: bool) -> str:
    lines: list[str] = []
    last_system = ""
    for turn in turns:
        request = turn.get("request") or []
        work = turn.get("work") or []

        # Some traces keep the instructions only in the LLM call. They can change between turns.
        llm_messages = [
            m for s in work if (s.get("work") or {}).get("type") == "llm_analysis" for m in s["work"].get("work") or []
        ]
        system_texts = [_text(m.get("content")) for m in request + llm_messages if m.get("role") == "system"]
        system = "\n".join(dict.fromkeys(system_texts))
        if with_system and system and system != last_system:
            lines.append(f"System: {system}")
            last_system = system

        lines += [f"User: {_text(m.get('content'))}" for m in request if m.get("role") == "user"]

        for step in work:
            # Skip steps that are not tool calls, such as span_type "task".
            step_work = step.get("work") or {}
            if step.get("span_type") != "tool" or step_work.get("type") != "tool_result":
                continue
            # When the tool call has an ID, Lingua moves the output into `content`.
            parts = [p for p in step_work.get("content") or [] if p.get("type") == "tool_result"]
            output = parts[0].get("output") if parts else step_work.get("output")
            error = step.get("error")
            result = f"error: {_json(error)}" if error is not None else _json(output)
            name = step.get("name") or "tool"
            lines.append(f"Tool: {name}({_json(step_work.get('input'))}) -> {result}")

        # An interrupted response is already cut off where it stopped.
        response = _text((turn.get("response") or {}).get("response"))
        if turn.get("interrupted"):
            response += " [interrupted]"
        if response.strip():
            lines.append(f"Assistant: {response.strip()}")
    return "\n".join(lines)


# Message content is a string or a list of parts. Only text renders; audio and images don't.
def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "\n".join(p["text"] for p in content if isinstance(p, dict) and p.get("type") == "text")


# Tool inputs, outputs and errors: strings as they are, anything else as compact JSON.
def _json(value: Any) -> str:
    if value is None:
        return ""
    return value if isinstance(value, str) else json.dumps(value, separators=(",", ":"), ensure_ascii=False)
