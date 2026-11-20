"""Offline consistency of repeated categorical judge results, distinct from accuracy."""

from .adapters import ScoreMapping, error_rating, missing_rating, rating_from_score
from .analysis import analyze_consistency
from .config import ConsistencyBootstrap, ConsistencyGate, ConsistencyGroup, ConsistencyLimits, ConsistencyPlan, Rating
from .reporting import consistency_scores, render_consistency_markdown
from .serialization import analyze_consistency_json, input_to_json, report_to_json

__all__ = [
    "ConsistencyBootstrap",
    "ConsistencyGate",
    "ConsistencyGroup",
    "ConsistencyLimits",
    "ConsistencyPlan",
    "Rating",
    "ScoreMapping",
    "analyze_consistency",
    "error_rating",
    "missing_rating",
    "rating_from_score",
    "analyze_consistency_json",
    "input_to_json",
    "report_to_json",
    "consistency_scores",
    "render_consistency_markdown",
]
