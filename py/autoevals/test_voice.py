import asyncio

from autoevals import VoiceInterruptions, VoiceLatency


def user(start, end):
    return {"speaker": "user", "start_unix_ms": start, "end_unix_ms": end}


def agent(start, end, interrupted=None):
    return {"speaker": "agent", "start_unix_ms": start, "end_unix_ms": end, "interrupted": interrupted}


def latency(utterances):
    return VoiceLatency().eval(output=None, utterances=utterances)


def interruptions(utterances):
    return VoiceInterruptions().eval(output=None, utterances=utterances)


def test_latency_scores_share_of_fast_replies():
    result = latency([user(0, 2000), agent(2800, 5000), user(6000, 7000), agent(10000, 11000)])
    assert result.score == 0.5
    assert result.metadata["replies"] == 2
    assert result.metadata["unanswered_turns"] == 0
    assert result.metadata["p50_ms"] == 800
    assert result.metadata["p95_ms"] == 3000
    assert result.metadata["gaps_ms"] == [800, 3000]


def test_latency_leaves_out_turns_the_agent_talked_over():
    result = latency([user(0, 3000), agent(2000, 4000), user(5000, 6000), agent(6500, 8000)])
    assert result.score == 1
    assert result.metadata["gaps_ms"] == [500]


def test_latency_counts_a_slightly_early_reply_as_instant():
    assert latency([user(0, 3000), agent(2800, 5000)]).metadata["gaps_ms"] == [0]


def test_latency_does_not_take_a_short_cut_in_as_the_reply():
    assert latency([user(0, 3000), agent(2000, 2200), agent(3500, 5000)]).metadata["gaps_ms"] == [500]


def test_scorers_merge_user_fragments_and_skip_backchannels():
    fragmented = [user(0, 1000), user(1200, 3000), agent(3500, 5000)]
    assert latency(fragmented).metadata["gaps_ms"] == [500]
    assert interruptions(fragmented).metadata["user_turns"] == 1

    backchannel = [user(0, 1000), agent(1500, 6000), user(3000, 3300), agent(6200, 8000)]
    assert latency(backchannel).metadata["gaps_ms"] == [500]
    assert interruptions(backchannel).metadata["user_turns"] == 1


def test_interruptions_scores_talk_over_not_barge_ins():
    result = interruptions([user(0, 3000), agent(2000, 4000), user(5000, 6000), agent(6500, 8000, False)])
    assert result.score == 0.5
    assert result.metadata["talk_overs"] == 1
    assert result.metadata["user_turns"] == 2
    assert result.metadata["barge_ins"] == 0

    barge_in = interruptions([agent(0, 3000, True), user(2000, 4000)])
    assert barge_in.score == 1
    assert barge_in.metadata["barge_ins"] == 1


def test_interruptions_ignores_overlaps_under_min_overlap_ms():
    utterances = [user(0, 3000), agent(2900, 4000)]
    assert interruptions(utterances).score == 1
    assert VoiceInterruptions().eval(output=None, utterances=utterances, min_overlap_ms=50).score == 0


def test_scorers_skip_invalid_utterances_and_return_none_without_data():
    utterances = [
        {"speaker": "user"},
        {"start_unix_ms": 0, "end_unix_ms": 1000},
        {"speaker": "agent", "start_unix_ms": float("nan"), "end_unix_ms": 1000},
        {"speaker": "agent", "start_unix_ms": 2000, "end_unix_ms": 1000},
        {"speaker": "assistant", "start_unix_ms": 0, "end_unix_ms": 1000},
    ]
    result = latency(utterances)
    assert result.score is None
    assert result.metadata["skipped"] == 5
    assert interruptions(utterances).score is None
    assert interruptions([user(0, 1000)]).score is None


def test_partial_and_async():
    scorer = VoiceLatency.partial(max_gap_ms=5000)()
    result = asyncio.run(scorer.eval_async(output=None, utterances=[user(0, 2000), agent(5000, 6000)]))
    assert result.score == 1
