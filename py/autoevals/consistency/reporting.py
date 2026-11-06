"""Human-readable and native Score projections with explicit interpretation."""

import html
from typing import Any

from ..score import Score


def safe_text(value: str) -> str:
    """Escape terminal controls, bidirectional controls and Markdown table syntax."""
    clean = "".join(
        (
            f"\\u{ord(char):04x}"
            if ord(char) < 32
            or 127 <= ord(char) <= 159
            or 0x202A <= ord(char) <= 0x202E
            or 0x2066 <= ord(char) <= 0x2069
            else char
        )
        for char in value
    )
    literal = "".join("\\" + char if char in "\\`*_{}[]()#+-.!|" else char for char in clean)
    return html.escape(literal, quote=True)


def _number(value: float | None) -> str:
    return "undefined" if value is None else f"{value:.6g}"


def render_consistency_markdown(report: dict[str, Any]) -> str:
    """Render bounded report details without converting missing statistics to zero."""
    lines = [
        f"# Judge consistency: {safe_text(report['score_name'])}",
        "",
        f"Observed benchmark verdict: **{report['verdict']}**. Cohort: {report['cohort_provenance']}.",
        "",
        "Consistency is not correctness. Gates describe the supplied finite benchmark; "
        "bootstrap intervals are marginal diagnostics and do not determine gate decisions.",
        "",
    ]
    for group in report["groups"]:
        coverage = group["coverage"]
        lines.extend(
            [
                f"## {safe_text(group['name'])}",
                "",
                f"Scored ratings: {coverage['scored_cells']}/{coverage['expected_cells']}; "
                f"absent: {coverage['absent_cells']}; skipped: {coverage['missing_cells']}; errors: {coverage['error_cells']}.",
                "",
                "| Statistic | Value | Supporting cases | Denominator | Weighting / undefined reason |",
                "| --- | ---: | ---: | ---: | --- |",
            ]
        )
        for name, statistic in group["statistics"].items():
            lines.append(
                f"| {safe_text(name)} | {_number(statistic['value'])} | {statistic['support_cases']} | "
                f"{statistic['denominator_cases']} | {statistic['weighting']} / {statistic['reason'] or 'defined'} |"
            )
        lines.extend(
            [
                "",
                "### Reference accuracy",
                "",
                "| Rater | Correct / evaluated / labeled | Conditional accuracy | Reference coverage |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        reference_rows = [(rater["rater_id"], rater["reference"]) for rater in group["raters"]]
        reference_rows.append(("consensus", group["consensus"]["reference"]))
        for name, reference in reference_rows:
            lines.append(
                f"| {safe_text(name)} | {reference['correct_cases']} / {reference['evaluated_cases']} / "
                f"{reference['labeled_cases']} | {_number(reference['accuracy'])} | {_number(reference['coverage'])} |"
            )
        lines.extend(
            [
                "",
                "### Pairwise overlap",
                "",
                "| Left rater | Right rater | Cohen kappa | Overlap cases | Reason |",
                "| --- | --- | ---: | ---: | --- |",
            ]
        )
        for pair in group["pairwise"]:
            statistic = pair["statistic"]
            lines.append(
                f"| {safe_text(pair['left_rater'])} | {safe_text(pair['right_rater'])} | "
                f"{_number(statistic['value'])} | {statistic['support_cases']} | {statistic['reason'] or 'defined'} |"
            )
        if group["intervals"]:
            lines.extend(["", "### Marginal bootstrap diagnostics", ""])
            for interval in group["intervals"]:
                lines.append(
                    f"- {interval['statistic']}: [{_number(interval['lower'])}, {_number(interval['upper'])}]; "
                    f"valid/attempted={interval['valid']}/{interval['attempted']}; "
                    f"reason={interval['reason'] or 'defined'}."
                )
        lines.extend(
            [
                "",
                f"Consensus decided/abstained: {group['consensus']['decided_cases']}/"
                f"{group['consensus']['abstained_cases']}. Omitted case details: {group['omitted_cases']}.",
                "",
            ]
        )
        for case in group["cases"]:
            label = "abstained" if case["consensus"] is None else safe_text(case["consensus"])
            lines.append(
                f"- {safe_text(case['case_id'])}: consensus={label}; "
                f"agreement={_number(case['raw_agreement'])}; reason={case['consensus_reason'] or 'decided'}."
            )
        lines.append("")
    if report["gates"]:
        lines.extend(["## Observed requirements", ""])
        for gate in report["gates"]:
            lines.append(
                f"- {safe_text(gate['group'])}/{gate['statistic']}: {gate['verdict']}; "
                f"value={_number(gate['value'])}, minimum={_number(gate['minimum'])}; {', '.join(gate['reasons'])}."
            )
    return "\n".join(lines) + "\n"


def consistency_scores(report: dict[str, Any], *, include_report: bool = False) -> list[Score]:
    """Project bounded agreement/coverage/accuracy into native Scores.

    Signed kappa and alpha coefficients deliberately remain in the full report,
    because AutoEvals Score values are restricted to[0,1]. Undefined summaries
    use None, never zero. Full case-level report metadata is explicit opt-in.
    """
    if type(include_report) is not bool:
        raise ValueError("include_report must be boolean")
    results = []
    for group in report["groups"]:
        for suffix, value, support in (
            ("agreement", group["statistics"]["raw_agreement"]["value"], group["statistics"]["raw_agreement"]),
            ("coverage", group["coverage"]["rating_coverage"], group["coverage"]),
            ("consensus_accuracy", group["consensus"]["reference"]["accuracy"], group["consensus"]["reference"]),
        ):
            metadata = {
                "cohort_provenance": report["cohort_provenance"],
                "support": dict(support),
                "interpretation": "observed_finite_benchmark",
            }
            if include_report:
                # Detach metadata rather than exposing the caller's mutable report.
                import copy

                metadata["report"] = copy.deepcopy(report)
            results.append(Score(f"{report['score_name']}/{group['name']}/{suffix}", value, metadata))
    return results
