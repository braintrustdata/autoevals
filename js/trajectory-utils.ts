/* eslint-disable @typescript-eslint/no-explicit-any */

export function templateUsesTrajectoryVariables(template: string): boolean {
  return /\{\{\s*trajectory(_with_system)?\s*\}\}/.test(template);
}

export function computeTrajectoryTemplateVars(trajectories: any[]) {
  return {
    trajectory: render(trajectories, false),
    trajectory_with_system: render(trajectories, true),
  };
}

function render(trajectories: any[], withSystem: boolean): string {
  const lines: string[] = [];
  for (const trajectory of trajectories) {
    renderTrajectory(trajectory, withSystem, "", lines);
  }
  return lines.join("\n");
}

// Each item is `Label: body`. Continuation lines are indented under it, so content can't look like a label.
function renderTrajectory(
  trajectory: any,
  withSystem: boolean,
  indent: string,
  lines: string[],
) {
  const emit = (label: string, body: string) => {
    const [first, ...rest] = body.split("\n");
    lines.push(`${indent}${label}: ${first}`);
    for (const line of rest) lines.push(line ? `${indent}  ${line}` : "");
  };

  let lastSystem = "";
  for (const turn of trajectory.turns ?? []) {
    const request: any[] = turn.request ?? [];
    const work: any[] = turn.work ?? [];

    // Some traces keep the instructions only in the LLM call. They can change between turns.
    const llmMessages = work
      .filter((s) => s.work?.type === "llm_analysis")
      .flatMap((s) => s.work.work ?? []);
    const systemTexts = [...request, ...llmMessages]
      .filter((m) => m.role === "system" || m.role === "developer")
      .map((m) => text(m.content));
    const system = [...new Set(systemTexts)].join("\n");
    if (withSystem && system && system !== lastSystem) {
      emit("System", system);
      lastSystem = system;
    }

    for (const m of request) {
      if (m.role === "user") emit("User", text(m.content));
    }

    for (const step of work) {
      if (step.span_type === "tool" && step.work?.type === "tool_result") {
        emit("Tool", toolCall(step));
      }
      if (step.sub_agent) {
        lines.push(`${indent}  Sub-agent:`);
        renderTrajectory(step.sub_agent, withSystem, `${indent}    `, lines);
      }
    }

    const response = text(turn.response?.response).trim();
    if (response) emit("Assistant", response);
  }
}

function toolCall(step: any): string {
  // When the tool call has an ID, Lingua moves the output into `content`.
  const part = step.work.content?.find((p: any) => p.type === "tool_result");
  const output = part ? part.output : step.work.output;
  const result =
    step.error != null ? `error: ${json(step.error)}` : json(output);
  return `${step.name || "tool"}(${json(step.work.input)}) -> ${result}`;
}

function text(content: any): string {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  return content
    .filter((p) => p?.type === "text")
    .map((p) => p.text)
    .join("\n");
}

function json(value: any): string {
  if (value === undefined || value === null) return "";
  return typeof value === "string" ? value : JSON.stringify(value);
}
