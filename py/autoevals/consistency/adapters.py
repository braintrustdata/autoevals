"""Convert native single-result Scores into explicit categorical observations."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ..score import Score
from .config import Rating, finite, identifier


@dataclass(frozen=True)
class ScoreMapping:
    """Explicit exact numeric-to-category mapping; no inferred thresholds.

    Use entries rather than JSON object keys so numeric types remain numeric.
    Multiple numeric values may map to one category. Each numeric value appears
    once, and an unmapped finite score is an error rather than a missing vote.
    """

    score_name: str
    entries: tuple[tuple[float, str], ...]

    def __post_init__(self) -> None:
        identifier(self.score_name, "score_name")
        if not isinstance(self.entries, (tuple, list)) or not 1 <= len(self.entries) <= 1000:
            raise ValueError("mapping entries must be a nonempty bounded sequence")
        normalized = []
        for entry in self.entries:
            if not isinstance(entry, (tuple, list)) or len(entry) != 2:
                raise ValueError("mapping entries require score/category pairs")
            normalized.append((finite(entry[0], "mapped score", 0, 1), identifier(entry[1], "category")))
        if len({value for value, _ in normalized}) != len(normalized):
            raise ValueError("duplicate mapped numeric score")
        object.__setattr__(self, "entries", tuple(sorted(normalized)))


def missing_rating(case_id: str, rater_id: str) -> Rating:
    """Record an explicitly skipped evaluation, distinct from an absent cell."""
    return Rating(case_id, rater_id, status="missing")


def error_rating(case_id: str, rater_id: str) -> Rating:
    """Record a caught scorer exception without retaining its message or payload."""
    return Rating(case_id, rater_id, status="error")


def rating_from_score(
    case_id: str, rater_id: str, score: Score | Mapping[str, Any], *, mapping: ScoreMapping
) -> Rating:
    """Adapt one native result, independently validating its numeric contract.

    AutoEvals scorers may raise: callers decide whether to propagate or explicitly
    record ``error_rating``. The deprecated live ``Score.error`` takes precedence
    over its numeric value. ``Score.as_dict`` omits that field; adapt live results
    before serialization when error information matters. Metadata is never read.
    """
    if not isinstance(mapping, ScoreMapping):
        raise ValueError("mapping must be a ScoreMapping")
    if isinstance(score, Score):
        name, value, error = score.name, score.score, score.error
    elif isinstance(score, Mapping):
        if "name" not in score or "score" not in score:
            raise ValueError("score requires name and score fields")
        name, value, error = score["name"], score["score"], score.get("error")
    else:
        raise ValueError("expected a native Score or score mapping")
    if name != mapping.score_name:
        raise ValueError("score name does not match the selected criterion")
    if error is not None:
        return error_rating(case_id, rater_id)
    if value is None:
        return missing_rating(case_id, rater_id)
    numeric = finite(value, "score", 0, 1)
    for expected, category in mapping.entries:
        if numeric == expected:
            return Rating(case_id, rater_id, category)
    raise ValueError("numeric score is not mapped to a category")
