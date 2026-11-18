import { expect, test } from "vitest";
import { type Utterance, VoiceInterruptions, VoiceLatency } from "./voice";

const user = (start: number, end: number): Utterance => ({
  speaker: "user",
  start_unix_ms: start,
  end_unix_ms: end,
});
const agent = (
  start: number,
  end: number,
  interrupted?: boolean,
): Utterance => ({
  speaker: "agent",
  start_unix_ms: start,
  end_unix_ms: end,
  interrupted,
});

const latency = (utterances: Utterance[]) =>
  VoiceLatency({ output: null, utterances });
const interruptions = (utterances: Utterance[]) =>
  VoiceInterruptions({ output: null, utterances });

test("VoiceLatency scores the share of fast replies", async () => {
  const result = await latency([
    user(0, 2000),
    agent(2800, 5000),
    user(6000, 7000),
    agent(10000, 11000),
  ]);
  expect(result.score).toBe(0.5);
  expect(result.metadata).toMatchObject({
    replies: 2,
    unanswered_turns: 0,
    p50_ms: 800,
    p95_ms: 3000,
    gaps_ms: [800, 3000],
  });
});

test("VoiceLatency leaves out turns the agent talked over", async () => {
  const result = await latency([
    user(0, 3000),
    agent(2000, 4000),
    user(5000, 6000),
    agent(6500, 8000),
  ]);
  expect(result.score).toBe(1);
  expect(result.metadata?.gaps_ms).toEqual([500]);
});

test("VoiceLatency counts a slightly early reply as instant", async () => {
  const result = await latency([user(0, 3000), agent(2800, 5000)]);
  expect(result.metadata?.gaps_ms).toEqual([0]);
});

test("VoiceLatency doesn't take a short cut-in as the reply", async () => {
  const result = await latency([
    user(0, 3000),
    agent(2000, 2200),
    agent(3500, 5000),
  ]);
  expect(result.metadata?.gaps_ms).toEqual([500]);
});

test("voice scorers merge user fragments and skip backchannels", async () => {
  const fragmented = [user(0, 1000), user(1200, 3000), agent(3500, 5000)];
  expect((await latency(fragmented)).metadata?.gaps_ms).toEqual([500]);
  expect((await interruptions(fragmented)).metadata?.user_turns).toBe(1);

  const backchannel = [
    user(0, 1000),
    agent(1500, 6000),
    user(3000, 3300),
    agent(6200, 8000),
  ];
  expect((await latency(backchannel)).metadata?.gaps_ms).toEqual([500]);
  expect((await interruptions(backchannel)).metadata?.user_turns).toBe(1);
});

test("VoiceInterruptions scores talk-over, not barge-ins", async () => {
  const result = await interruptions([
    user(0, 3000),
    agent(2000, 4000),
    user(5000, 6000),
    agent(6500, 8000, false),
  ]);
  expect(result.score).toBe(0.5);
  expect(result.metadata).toMatchObject({
    talk_overs: 1,
    user_turns: 2,
    barge_ins: 0,
  });

  const bargeIn = await interruptions([agent(0, 3000, true), user(2000, 4000)]);
  expect(bargeIn.score).toBe(1);
  expect(bargeIn.metadata?.barge_ins).toBe(1);
});

test("VoiceInterruptions ignores overlaps under minOverlapMs", async () => {
  const utterances = [user(0, 3000), agent(2900, 4000)];
  expect((await interruptions(utterances)).score).toBe(1);
  expect(
    (await VoiceInterruptions({ output: null, utterances, minOverlapMs: 50 }))
      .score,
  ).toBe(0);
});

test("voice scorers skip invalid utterances and return null without data", async () => {
  const utterances = [
    { speaker: "user" },
    { speaker: "agent", start_unix_ms: NaN, end_unix_ms: 1000 },
    { speaker: "agent", start_unix_ms: 2000, end_unix_ms: 1000 },
    { speaker: "assistant", start_unix_ms: 0, end_unix_ms: 1000 },
  ] as Utterance[];
  const result = await latency(utterances);
  expect(result.score).toBeNull();
  expect(result.metadata?.skipped).toBe(4);
  expect((await interruptions(utterances)).score).toBeNull();
  expect((await interruptions([user(0, 1000)])).score).toBeNull();
});
