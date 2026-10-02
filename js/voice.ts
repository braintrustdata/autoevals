import { makePartial, ScorerWithPartial } from "./partial";

/**
 * One stretch of speech in a voice call. Times are Unix epoch milliseconds.
 * These are the timing fields of `Utterance` in Braintrust's voice trace reader.
 */
export interface Utterance {
  speaker: "user" | "agent";
  start_unix_ms?: number;
  end_unix_ms?: number;
  /** The user cut the agent off, as reported by the voice framework. */
  interrupted?: boolean;
}

type TimedUtterance = Utterance & {
  start_unix_ms: number;
  end_unix_ms: number;
};

type Turn = {
  start: number;
  end: number;
  talkOver: boolean;
  /** Milliseconds from the end of the turn to the agent's reply, or null without a reply. */
  gapMs: number | null;
};

function isTimed(u: Utterance): u is TimedUtterance {
  return (
    (u.speaker === "user" || u.speaker === "agent") &&
    Number.isFinite(u.start_unix_ms) &&
    Number.isFinite(u.end_unix_ms) &&
    u.end_unix_ms! >= u.start_unix_ms!
  );
}

/**
 * Groups the user's speech into turns and finds the agent's reply to each.
 * Both scorers use this so they count turns the same way.
 */
function userTurns(
  utterances: Utterance[],
  minOverlapMs: number,
): { turns: Turn[]; agents: number; skipped: number } {
  const timed = utterances
    .filter(isTimed)
    .sort((a, b) => a.start_unix_ms - b.start_unix_ms);
  const agents = timed.filter((u) => u.speaker === "agent");

  const turns: Turn[] = [];
  for (const u of timed) {
    if (u.speaker !== "user") {
      continue;
    }
    // User speech wholly inside the agent's, like "mm-hmm", isn't a turn.
    if (
      agents.some(
        (a) =>
          a.start_unix_ms <= u.start_unix_ms && u.end_unix_ms <= a.end_unix_ms,
      )
    ) {
      continue;
    }
    const last = turns[turns.length - 1];
    const agentStartedBetween =
      last &&
      agents.some(
        (a) =>
          a.start_unix_ms > last.start && a.start_unix_ms < u.start_unix_ms,
      );
    if (last && !agentStartedBetween) {
      last.end = Math.max(last.end, u.end_unix_ms);
    } else {
      turns.push({
        start: u.start_unix_ms,
        end: u.end_unix_ms,
        talkOver: false,
        gapMs: null,
      });
    }
  }

  turns.forEach((turn, i) => {
    const nextStart = i + 1 < turns.length ? turns[i + 1].start : Infinity;
    turn.talkOver = agents.some(
      (a) =>
        a.start_unix_ms > turn.start &&
        a.start_unix_ms < turn.end &&
        Math.min(a.end_unix_ms, turn.end) - a.start_unix_ms > minOverlapMs,
    );
    // A reply runs past the end of the turn, so a short cut-in inside it isn't one.
    const reply = agents.find(
      (a) =>
        a.start_unix_ms > turn.start &&
        a.start_unix_ms < nextStart &&
        a.end_unix_ms >= turn.end,
    );
    turn.gapMs = reply ? Math.max(0, reply.start_unix_ms - turn.end) : null;
  });

  return {
    turns,
    agents: agents.length,
    skipped: utterances.length - timed.length,
  };
}

function percentile(values: number[], p: number): number | null {
  if (values.length === 0) {
    return null;
  }
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[
    Math.min(sorted.length - 1, Math.ceil((p / 100) * sorted.length) - 1)
  ];
}

type VoiceLatencyArgs = {
  utterances?: Utterance[];
  maxGapMs?: number;
  minOverlapMs?: number;
};

/**
 * The share of agent replies that start within `maxGapMs` of the user
 * finishing a turn. Turns the agent talked over aren't counted. Returns a
 * null score when there are no replies.
 */
export const VoiceLatency: ScorerWithPartial<unknown, VoiceLatencyArgs> =
  makePartial(
    async ({ utterances = [], maxGapMs = 1500, minOverlapMs = 300 }) => {
      const { turns, skipped } = userTurns(utterances, minOverlapMs);
      const gaps = turns
        .filter((t) => !t.talkOver && t.gapMs !== null)
        .map((t) => t.gapMs!);
      return {
        name: "VoiceLatency",
        score: gaps.length
          ? gaps.filter((gap) => gap <= maxGapMs).length / gaps.length
          : null,
        metadata: {
          replies: gaps.length,
          unanswered_turns: turns.filter((t) => t.gapMs === null).length,
          max_gap_ms: maxGapMs,
          p50_ms: percentile(gaps, 50),
          p95_ms: percentile(gaps, 95),
          gaps_ms: gaps,
          skipped,
        },
      };
    },
    "VoiceLatency",
  );

type VoiceInterruptionsArgs = {
  utterances?: Utterance[];
  minOverlapMs?: number;
};

/**
 * The share of user turns the agent let finish without talking over them.
 * The agent talks over a turn when it starts speaking during the turn and
 * overlaps it by more than `minOverlapMs`. Returns a null score unless the
 * call has timed user and agent utterances.
 */
export const VoiceInterruptions: ScorerWithPartial<
  unknown,
  VoiceInterruptionsArgs
> = makePartial(async ({ utterances = [], minOverlapMs = 300 }) => {
  const { turns, agents, skipped } = userTurns(utterances, minOverlapMs);
  const talkOvers = turns.filter((t) => t.talkOver).length;
  return {
    name: "VoiceInterruptions",
    score: turns.length && agents ? 1 - talkOvers / turns.length : null,
    metadata: {
      user_turns: turns.length,
      talk_overs: talkOvers,
      min_overlap_ms: minOverlapMs,
      barge_ins: utterances.filter(
        (u) => u.speaker === "agent" && u.interrupted === true,
      ).length,
      skipped,
    },
  };
}, "VoiceInterruptions");
