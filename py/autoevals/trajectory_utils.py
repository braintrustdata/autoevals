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
    for trajectory in trajectories:
        _render_trajectory(trajectory, with_system, "", lines)
    return "\n".join(lines)


# Each item is `Label: body`. Continuation lines are indented under it, so content can't look like a label.
def _render_trajectory(trajectory: Any, with_system: bool, indent: str, lines: list[str]) -> None:
    def emit(label: str, body: str) -> None:
        # split("\n") matches JS; splitlines() also splits on other line breaks.
        first, *rest = body.split("\n")
        lines.append(f"{indent}{label}: {first}")
        lines.extend(f"{indent}  {line}" if line else "" for line in rest)

    last_system = ""
    for turn in trajectory.get("turns") or []:
        request = turn.get("request") or []
        work = turn.get("work") or []

        # Some traces keep the instructions only in the LLM call. They can change between turns.
        llm_messages = [
            m for s in work if (s.get("work") or {}).get("type") == "llm_analysis" for m in s["work"].get("work") or []
        ]
        system_texts = [
            _text(m.get("content")) for m in request + llm_messages if m.get("role") in ("system", "developer")
        ]
        system = "\n".join(dict.fromkeys(system_texts))
        if with_system and system and system != last_system:
            emit("System", system)
            last_system = system

        for m in request:
            if m.get("role") == "user":
                emit("User", _text(m.get("content")))

        for step in work:
            if step.get("span_type") == "tool" and (step.get("work") or {}).get("type") == "tool_result":
                emit("Tool", _tool_call(step))
            if step.get("sub_agent"):
                lines.append(f"{indent}  Sub-agent:")
                _render_trajectory(step["sub_agent"], with_system, f"{indent}    ", lines)

        response = _text((turn.get("response") or {}).get("response")).strip()
        if response:
            emit("Assistant", response)


def _tool_call(step: Any) -> str:
    step_work = step["work"]
    # When the tool call has an ID, Lingua moves the output into `content`.
    parts = [p for p in step_work.get("content") or [] if p.get("type") == "tool_result"]
    output = parts[0].get("output") if parts else step_work.get("output")
    error = step.get("error")
    result = f"error: {_json(error)}" if error is not None else _json(output)
    return f"{step.get('name') or 'tool'}({_json(step_work.get('input'))}) -> {result}"


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
