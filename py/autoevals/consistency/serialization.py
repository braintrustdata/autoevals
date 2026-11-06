"""Bounded non-executable input bundles and versioned JSON reports."""

import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import asdict
from typing import Any

from .analysis import analyze_consistency
from .config import ConsistencyBootstrap, ConsistencyGate, ConsistencyGroup, ConsistencyLimits, ConsistencyPlan, Rating


def _object(value: Any, allowed: set[str], required: set[str], name: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    if set(value) - allowed:
        raise ValueError(f"{name} contains unknown fields")
    if required - set(value):
        raise ValueError(f"{name} is missing required fields")
    return dict(value)


def _list(value: Any, maximum: int, name: str) -> list:
    if not isinstance(value, list) or len(value) > maximum:
        raise ValueError(f"{name} must be a bounded array")
    return value


def _unique_object(entries: list[tuple[str, Any]]) -> dict:
    value = {}
    for key, item in entries:
        if key in value:
            raise ValueError("duplicate JSON key")
        value[key] = item
    return value


def _constant(value: str) -> None:
    raise ValueError("nonfinite JSON constants are not supported")


def _structure(value: Any, limits: ConsistencyLimits) -> None:
    stack = [(value, 0)]
    count = 0
    while stack:
        item, depth = stack.pop()
        count += 1
        if count > limits.max_json_nodes or depth > limits.max_json_depth:
            raise ValueError("consistency JSON structure limit exceeded")
        if isinstance(item, dict):
            stack.extend((key, depth + 1) for key in item)
            stack.extend((child, depth + 1) for child in item.values())
        elif isinstance(item, (tuple, list)):
            stack.extend((child, depth + 1) for child in item)
        elif type(item) is float and not math.isfinite(item):
            raise ValueError("nonfinite JSON numbers are not supported")
        elif isinstance(item, str):
            try:
                size = len(item.encode("utf-8"))
            except UnicodeError as exc:
                raise ValueError("invalid Unicode in JSON") from exc
            if size > 16384:
                raise ValueError("consistency JSON string limit exceeded")


def plan_from_dict(value: Any, *, limits: ConsistencyLimits | None = None) -> ConsistencyPlan:
    """Read only declared plan fields; no inherited permissive model decoder."""
    limits = limits or ConsistencyLimits()
    data = _object(value, set(ConsistencyPlan.__dataclass_fields__), {"score_name", "raters", "categories"}, "plan")
    groups = _list(data.get("groups", []), limits.max_groups, "groups")
    data["groups"] = tuple(
        ConsistencyGroup(**_object(group, {"name", "cases"}, {"name", "cases"}, "group")) for group in groups
    )
    gates = _list(data.get("gates", []), (limits.max_groups + 1) * 4, "gates")
    data["gates"] = tuple(
        ConsistencyGate(**_object(gate, set(ConsistencyGate.__dataclass_fields__), {"statistic", "minimum"}, "gate"))
        for gate in gates
    )
    bootstrap = _object(data.get("bootstrap", {}), set(ConsistencyBootstrap.__dataclass_fields__), set(), "bootstrap")
    data["bootstrap"] = ConsistencyBootstrap(**bootstrap)
    plan = ConsistencyPlan(**data)
    plan.check_limits(limits)
    return plan


def analyze_consistency_json(content: str, *, limits: ConsistencyLimits | None = None) -> dict[str, Any]:
    """Read a strict portable input bundle and recompute evidence locally."""
    limits = limits or ConsistencyLimits()
    if type(content) is not str:
        raise ValueError("input JSON must be a string")
    try:
        size = len(content.encode("utf-8"))
    except UnicodeError as exc:
        raise ValueError("invalid Unicode in JSON") from exc
    if size > limits.max_bytes:
        raise ValueError("consistency input byte limit exceeded")
    try:
        value = json.loads(content, object_pairs_hook=_unique_object, parse_constant=_constant)
    except (json.JSONDecodeError, RecursionError) as exc:
        raise ValueError("invalid consistency JSON") from exc
    _structure(value, limits)
    data = _object(
        value,
        {"format", "version", "plan", "ratings", "references"},
        {"format", "version", "plan", "ratings"},
        "input bundle",
    )
    if data["format"] != "autoevals.consistency.input" or type(data["version"]) is not int or data["version"] != 1:
        raise ValueError("unsupported consistency input format/version")
    plan = plan_from_dict(data["plan"], limits=limits)
    ratings = tuple(
        Rating(**_object(row, set(Rating.__dataclass_fields__), {"case_id", "rater_id"}, "rating"))
        for row in _list(data["ratings"], limits.max_ratings, "ratings")
    )
    return analyze_consistency(ratings, plan=plan, references=data.get("references"), limits=limits)


def _encode(value: Any, limits: ConsistencyLimits) -> str:
    _structure(value, limits)
    try:
        content = json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise ValueError("consistency data must be finite JSON") from exc
    if len(content.encode("utf-8")) > limits.max_bytes:
        raise ValueError("consistency output byte limit exceeded")
    return content


def input_to_json(
    ratings: Iterable[Rating],
    *,
    plan: ConsistencyPlan,
    references: Mapping[str, str] | None = None,
    limits: ConsistencyLimits | None = None,
) -> str:
    """Snapshot selected observations into a reusable input bundle, without metadata."""
    limits = limits or ConsistencyLimits()
    collected = []
    for index, rating in enumerate(ratings):
        if index >= limits.max_ratings:
            raise ValueError("consistency rating limit exceeded")
        collected.append(rating)
    # Validate without performing optional resampling or report construction.
    from .alignment import align

    align(collected, plan, references, limits)
    return _encode(
        {
            "format": "autoevals.consistency.input",
            "version": 1,
            "plan": asdict(plan),
            "ratings": [asdict(rating) for rating in collected],
            "references": dict(references or {}),
        },
        limits,
    )


def report_to_json(report: dict[str, Any], *, limits: ConsistencyLimits | None = None) -> str:
    """Serialize a report as bounded finite JSON; this is not an authenticity proof."""
    if report.get("format") != "autoevals.consistency.report" or report.get("version") != 1:
        raise ValueError("unsupported consistency report")
    return _encode(report, limits or ConsistencyLimits())
