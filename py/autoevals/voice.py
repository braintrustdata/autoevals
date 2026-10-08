"""Timing scorers for voice agents.

This module provides scorers that read the timing of a voice call:
- VoiceLatency: The share of agent replies that start within `max_gap_ms` of the
  user finishing a turn.
- VoiceInterruptions: The share of user turns the agent let finish without talking
  over them.

Both take `utterances`, a list of dicts with `speaker` ("user" or "agent"),
`start_unix_ms`, `end_unix_ms`, and optionally `interrupted` (the user cut the agent
off, as reported by the voice framework). Times are Unix epoch milliseconds.
"""

import math
from typing import Literal, TypedDict

from autoevals.partial import ScorerWithPartial

from .score import Score


class Utterance(TypedDict, total=False):
    speaker: Literal["user", "agent"]
    start_unix_ms: float
    end_unix_ms: float
    interrupted: bool


def _is_time(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _is_timed(u):
    return (
        u.get("speaker") in ("user", "agent")
        and _is_time(u.get("start_unix_ms"))
        and _is_time(u.get("end_unix_ms"))
        and u["end_unix_ms"] >= u["start_unix_ms"]
    )


def _user_turns(utterances, min_overlap_ms):
    """Groups the user's speech into turns and finds the agent's reply to each.

    Both scorers use this so they count turns the same way. Each turn has `talk_over` and
    `gap_ms`, the milliseconds from the end of the turn to the agent's reply (None without one).
    """
    timed = sorted((u for u in utterances if _is_timed(u)), key=lambda u: u["start_unix_ms"])
    agents = [u for u in timed if u["speaker"] == "agent"]

    turns = []
    for u in timed:
        if u["speaker"] != "user":
            continue
        # User speech wholly inside the agent's, like "mm-hmm", isn't a turn.
        if any(a["start_unix_ms"] <= u["start_unix_ms"] and u["end_unix_ms"] <= a["end_unix_ms"] for a in agents):
            continue
        last = turns[-1] if turns else None
        if last and not any(last["start"] < a["start_unix_ms"] < u["start_unix_ms"] for a in agents):
            last["end"] = max(last["end"], u["end_unix_ms"])
        else:
            turns.append({"start": u["start_unix_ms"], "end": u["end_unix_ms"]})

    for i, turn in enumerate(turns):
        next_start = turns[i + 1]["start"] if i + 1 < len(turns) else math.inf
        turn["talk_over"] = any(
            turn["start"] < a["start_unix_ms"] < turn["end"]
            and min(a["end_unix_ms"], turn["end"]) - a["start_unix_ms"] > min_overlap_ms
            for a in agents
        )
        # A reply runs past the end of the turn, so a short cut-in inside it isn't one.
        reply = next(
            (a for a in agents if turn["start"] < a["start_unix_ms"] < next_start and a["end_unix_ms"] >= turn["end"]),
            None,
        )
        turn["gap_ms"] = max(0, reply["start_unix_ms"] - turn["end"]) if reply else None

    return turns, len(agents), len(utterances) - len(timed)


def _percentile(values, p):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(p / 100 * len(ordered)) - 1)]


class VoiceLatency(ScorerWithPartial):
    """The share of agent replies that start within `max_gap_ms` of the user finishing a turn.

    Turns the agent talked over aren't counted.

    Example:
        ```python
        result = VoiceLatency().eval(output=None, utterances=utterances, max_gap_ms=1500)
        print(result.score)  # 0.5 if half the replies were fast enough
        ```

    Args:
        utterances: The call's utterances
        max_gap_ms: Longest acceptable gap before a reply (default: 1500)
        min_overlap_ms: Overlap needed to count as talking over the user (default: 300)

    Returns:
        Score with the share of fast replies, or None when there are no replies.
    """

    def _run_eval_sync(self, output, expected=None, utterances=None, max_gap_ms=1500, min_overlap_ms=300, **kwargs):
        turns, _, skipped = _user_turns(utterances or [], min_overlap_ms)
        gaps = [t["gap_ms"] for t in turns if not t["talk_over"] and t["gap_ms"] is not None]
        return Score(
            name=self._name(),
            score=sum(gap <= max_gap_ms for gap in gaps) / len(gaps) if gaps else None,
            metadata={
                "replies": len(gaps),
                "unanswered_turns": sum(t["gap_ms"] is None for t in turns),
                "max_gap_ms": max_gap_ms,
                "p50_ms": _percentile(gaps, 50),
                "p95_ms": _percentile(gaps, 95),
                "gaps_ms": gaps,
                "skipped": skipped,
            },
        )


class VoiceInterruptions(ScorerWithPartial):
    """The share of user turns the agent let finish without talking over them.

    The agent talks over a turn when it starts speaking during the turn and overlaps it by
    more than `min_overlap_ms`. Barge-ins, where the user cuts the agent off, are counted in
    metadata but don't affect the score.

    Example:
        ```python
        result = VoiceInterruptions().eval(output=None, utterances=utterances)
        print(result.score)  # 1 if the agent never talked over the user
        ```

    Args:
        utterances: The call's utterances
        min_overlap_ms: Overlap needed to count as talking over the user (default: 300)

    Returns:
        Score with the share of user turns not talked over, or None unless the call has timed
        user and agent utterances.
    """

    def _run_eval_sync(self, output, expected=None, utterances=None, min_overlap_ms=300, **kwargs):
        utterances = utterances or []
        turns, agent_count, skipped = _user_turns(utterances, min_overlap_ms)
        talk_overs = sum(t["talk_over"] for t in turns)
        return Score(
            name=self._name(),
            score=1 - talk_overs / len(turns) if turns and agent_count else None,
            metadata={
                "user_turns": len(turns),
                "talk_overs": talk_overs,
                "min_overlap_ms": min_overlap_ms,
                "barge_ins": sum(u.get("speaker") == "agent" and u.get("interrupted") is True for u in utterances),
                "skipped": skipped,
            },
        )


__all__ = ["Utterance", "VoiceLatency", "VoiceInterruptions"]
