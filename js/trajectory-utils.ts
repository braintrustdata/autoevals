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
  let lastSystem = "";
  for (const turn of trajectories.flatMap((t) => t.turns ?? [])) {
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
      lines.push(
        `Tool: ${step.name || "tool"}(${json(step.work.input)}) -> ${result}`,
      );
    }

    const response = text(turn.response?.response).trim();
    if (response) lines.push(`Assistant: ${response}`);
  }
  return lines.join("\n");
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
