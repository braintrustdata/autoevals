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


def _render(trajectories: list[Any], with_system: bool) -> str:
    lines: list[str] = []
    last_system = ""
    for turn in [turn for t in trajectories for turn in t.get("turns") or []]:
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
            step_work = step.get("work") or {}
            if step.get("span_type") != "tool" or step_work.get("type") != "tool_result":
                continue
            # When the tool call has an ID, Lingua moves the output into `content`.
            parts = [p for p in step_work.get("content") or [] if p.get("type") == "tool_result"]
            output = parts[0].get("output") if parts else step_work.get("output")
            error = step.get("error")
            result = f"error: {_json(error)}" if error is not None else _json(output)
            lines.append(f"Tool: {step.get('name') or 'tool'}({_json(step_work.get('input'))}) -> {result}")

        response = _text((turn.get("response") or {}).get("response")).strip()
        if response:
            lines.append(f"Assistant: {response}")
    return "\n".join(lines)


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "\n".join(p["text"] for p in content if isinstance(p, dict) and p.get("type") == "text")


def _json(value: Any) -> str:
    if value is None:
        return ""
    return value if isinstance(value, str) else json.dumps(value, separators=(",", ":"), ensure_ascii=False)
