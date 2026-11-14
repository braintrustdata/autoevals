/**
 * Renders `trace.getTrajectory()` as plain text for LLM-as-a-judge templates.
 * `{{trajectory}}` omits the agent's instructions; `{{trajectory_with_system}}` includes them.
 */
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

// One block per trajectory, separated by a blank line.
function render(trajectories: any[], withSystem: boolean): string {
  return trajectories
    .map((t) => renderTurns(t.turns ?? [], withSystem))
    .filter(Boolean)
    .join("\n\n");
}

function renderTurns(turns: any[], withSystem: boolean): string {
  const lines: string[] = [];
  let lastSystem = "";
  for (const turn of turns) {
    const request: any[] = turn.request ?? [];
    const work: any[] = turn.work ?? [];

    // Some traces keep the instructions only in the LLM call. They can change between turns.
    const llmMessages = work
      .filter((s) => s.work?.type === "llm_analysis")
      .flatMap((s) => s.work.work ?? []);
    const systemTexts = [...request, ...llmMessages]
      .filter((m) => m.role === "system")
      .map((m) => text(m.content));
    const system = [...new Set(systemTexts)].join("\n");
    if (withSystem && system && system !== lastSystem) {
      lines.push(`System: ${system}`);
      lastSystem = system;
    }

    for (const m of request) {
      if (m.role === "user") lines.push(`User: ${text(m.content)}`);
    }

    for (const step of work) {
      // Skip steps that are not tool calls, such as span_type "task".
      if (step.span_type !== "tool" || step.work?.type !== "tool_result") {
        continue;
      }
      // When the tool call has an ID, Lingua moves the output into `content`.
      const part = step.work.content?.find(
        (p: any) => p.type === "tool_result",
      );
      const output = part ? part.output : step.work.output;
      const result =
        step.error != null ? `error: ${json(step.error)}` : json(output);
      const name = step.name || "tool";
      lines.push(`Tool: ${name}(${json(step.work.input)}) -> ${result}`);
    }

    // An interrupted response is already cut off where it stopped.
    let response = text(turn.response?.response);
    if (turn.interrupted) response += " [interrupted]";
    if (response.trim()) lines.push(`Assistant: ${response.trim()}`);
  }
  return lines.join("\n");
}

// Message content is a string or a list of parts. Only text renders; audio and images don't.
function text(content: any): string {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  return content
    .filter((p) => p?.type === "text")
    .map((p) => p.text)
    .join("\n");
}

// Tool inputs, outputs and errors: strings as they are, anything else as compact JSON.
function json(value: any): string {
  if (value === undefined || value === null) return "";
  return typeof value === "string" ? value : JSON.stringify(value);
}
