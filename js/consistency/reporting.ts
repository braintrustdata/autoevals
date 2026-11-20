import type { Score } from "../score";
import type { ConsistencyReport, ReferenceAccuracy } from "./types";

function safeText(value: string): string {
  return value
    .replace(
      /[\u0000-\u001f\u007f-\u009f\u202a-\u202e\u2066-\u2069]/g,
      (char) => `\\u${char.charCodeAt(0).toString(16).padStart(4, "0")}`,
    )
    .replace(/[\\`*_{}\[\]()#+\-.!|]/g, (char) => `\\${char}`)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#x27;");
}

function number(value: number | null): string {
  return value === null ? "undefined" : Number(value.toPrecision(6)).toString();
}

/** Render uncertainty, unavailable evidence and conditional accuracy explicitly. */
export function renderConsistencyMarkdown(report: ConsistencyReport): string {
  const lines = [
    `# Judge consistency: ${safeText(report.score_name)}`,
    "",
    `Observed benchmark verdict: **${report.verdict}**. Cohort: ${report.cohort_provenance}.`,
    "",
    "Consistency is not correctness. Gates describe the supplied finite benchmark; bootstrap intervals are marginal diagnostics and do not determine gate decisions.",
    "",
  ];
  for (const group of report.groups) {
    const coverage = group.coverage;
    lines.push(
      `## ${safeText(group.name)}`,
      "",
      `Scored ratings: ${coverage.scored_cells}/${coverage.expected_cells}; absent: ${coverage.absent_cells}; skipped: ${coverage.missing_cells}; errors: ${coverage.error_cells}.`,
      "",
      "| Statistic | Value | Supporting cases | Denominator | Weighting / undefined reason |",
      "| --- | ---: | ---: | ---: | --- |",
    );
    for (const [name, statistic] of Object.entries(group.statistics))
      lines.push(
        `| ${safeText(name)} | ${number(statistic.value)} | ${statistic.support_cases} | ${statistic.denominator_cases} | ${statistic.weighting} / ${statistic.reason ?? "defined"} |`,
      );
    lines.push(
      "",
      "### Reference accuracy",
      "",
      "| Rater | Correct / evaluated / labeled | Conditional accuracy | Reference coverage |",
      "| --- | ---: | ---: | ---: | --- |",
    );
    const references: [string, ReferenceAccuracy][] = group.raters.map(
      (rater) => [rater.rater_id, rater.reference],
    );
    references.push(["consensus", group.consensus.reference]);
    for (const [name, reference] of references)
      lines.push(
        `| ${safeText(name)} | ${reference.correct_cases} / ${reference.evaluated_cases} / ${reference.labeled_cases} | ${number(reference.accuracy)} | ${number(reference.coverage)} |`,
      );
    lines.push(
      "",
      "### Pairwise overlap",
      "",
      "| Left rater | Right rater | Cohen kappa | Overlap cases | Reason |",
      "| --- | --- | ---: | ---: | --- |",
    );
    for (const pair of group.pairwise)
      lines.push(
        `| ${safeText(pair.left_rater)} | ${safeText(pair.right_rater)} | ${number(pair.statistic.value)} | ${pair.statistic.support_cases} | ${pair.statistic.reason ?? "defined"} |`,
      );
    if (group.intervals.length) {
      lines.push("", "### Marginal bootstrap diagnostics", "");
      for (const interval of group.intervals)
        lines.push(
          `- ${interval.statistic}: [${number(interval.lower)}, ${number(interval.upper)}]; valid/attempted=${interval.valid}/${interval.attempted}; reason=${interval.reason ?? "defined"}.`,
        );
    }
    lines.push(
      "",
      `Consensus decided/abstained: ${group.consensus.decided_cases}/${group.consensus.abstained_cases}. Omitted case details: ${group.omitted_cases}.`,
      "",
    );
    for (const row of group.cases)
      lines.push(
        `- ${safeText(row.case_id)}: consensus=${row.consensus === null ? "abstained" : safeText(row.consensus)}; agreement=${number(row.raw_agreement)}; reason=${row.consensus_reason ?? "decided"}.`,
      );
    lines.push("");
  }
  if (report.gates.length) {
    lines.push("## Observed requirements", "");
    for (const gate of report.gates)
      lines.push(
        `- ${safeText(gate.group)}/${gate.statistic}: ${gate.verdict}; value=${number(gate.value)}, minimum=${number(gate.minimum)}; ${gate.reasons.join(", ")}.`,
      );
  }
  return lines.join("\n") + "\n";
}

/**
 * Native Score projections are bounded agreement, coverage and accuracy only.
 * Signed chance-corrected coefficients stay in the full report. Null means
 * undefined, never a fabricated zero. Full report metadata is explicit opt-in.
 */
export function consistencyScores(
  report: ConsistencyReport,
  options: { includeReport?: boolean } = {},
): Score[] {
  const includeReport = options.includeReport ?? false;
  if (typeof includeReport !== "boolean")
    throw new Error("includeReport must be boolean");
  const results: Score[] = [];
  for (const group of report.groups) {
    const summaries: [string, number | null, object][] = [
      [
        "agreement",
        group.statistics.raw_agreement.value,
        group.statistics.raw_agreement,
      ],
      ["coverage", group.coverage.rating_coverage, group.coverage],
      [
        "consensus_accuracy",
        group.consensus.reference.accuracy,
        group.consensus.reference,
      ],
    ];
    for (const [suffix, score, support] of summaries) {
      const metadata: Record<string, unknown> = {
        cohort_provenance: report.cohort_provenance,
        support: { ...support },
        interpretation: "observed_finite_benchmark",
      };
      if (includeReport) metadata.report = JSON.parse(JSON.stringify(report));
      results.push({
        name: `${report.score_name}/${group.name}/${suffix}`,
        score,
        metadata,
      });
    }
  }
  return results;
}
