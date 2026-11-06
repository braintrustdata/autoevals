import { coefficients } from "./analysis";
import type {
  ConsistencyInterval,
  NormalizedPlan,
  PanelCase,
  Statistic,
  StatisticName,
} from "./types";

/** Specified FNV1a namespace plus unsigned xorshift32, identical in Python. */
export class CaseRandom {
  private state: number;
  constructor(seed: number, namespace: string) {
    let state = (2166136261 ^ seed) >>> 0;
    for (const value of new TextEncoder().encode(namespace))
      state = Math.imul(state ^ value, 16777619) >>> 0;
    this.state = state || 0x9e3779b9;
  }
  nextU32(): number {
    let value = this.state;
    value ^= value << 13;
    value ^= value >>> 17;
    value ^= value << 5;
    this.state = value >>> 0;
    return this.state;
  }
  /** Rejection sampling avoids modulo bias. */
  index(size: number): number {
    const ceiling = Math.floor(2 ** 32 / size) * size;
    for (;;) {
      const value = this.nextU32();
      if (value < ceiling) return value % size;
    }
  }
}

function percentile(ordered: readonly number[], probability: number): number {
  const position = (ordered.length - 1) * probability;
  const low = Math.floor(position),
    high = Math.min(low + 1, ordered.length - 1);
  const weight = position - low;
  return ordered[low] * (1 - weight) + ordered[high] * weight;
}

/**
 * Preserve repeated case occurrences during resampling. An undefined draw
 * invalidates the interval; never discard it and claim an unconditional CI.
 */
export function bootstrapIntervals(
  cases: readonly PanelCase[],
  plan: NormalizedPlan,
  group: string,
  observed: Record<StatisticName, Statistic>,
): ConsistencyInterval[] {
  const options = plan.bootstrap;
  if (!options.resamples) return [];
  const results: ConsistencyInterval[] = [],
    eligible: ConsistencyInterval[] = [];
  const distributions = new Map<StatisticName, number[]>();
  for (const name of options.statistics) {
    const statistic = observed[name];
    let reason: string | null = null;
    if (statistic.value === null) reason = "undefined_observed_statistic";
    else if (statistic.support_cases < options.minimum_cases)
      reason = "insufficient_support";
    else if ((options.resamples * (1 - options.confidence)) / 2 < 20)
      reason = "insufficient_tail_resamples";
    const result: ConsistencyInterval = {
      statistic: name,
      confidence: options.confidence,
      resamples: options.resamples,
      attempted: 0,
      valid: 0,
      invalid: 0,
      lower: null,
      upper: null,
      reason,
      seed: options.seed,
      algorithm: "case-xorshift32-percentile-v1",
    };
    results.push(result);
    if (reason === null) {
      eligible.push(result);
      distributions.set(name, []);
    }
  }
  if (!eligible.length) return results;
  const rng = new CaseRandom(options.seed, group);
  for (let draw = 0; draw < options.resamples; draw++) {
    const sample = cases.map(() => cases[rng.index(cases.length)]);
    const values = coefficients(sample, plan);
    for (const result of eligible) {
      result.attempted++;
      const value = values[result.statistic].value;
      if (value === null) result.invalid++;
      else {
        result.valid++;
        distributions.get(result.statistic)!.push(value);
      }
    }
  }
  for (const result of eligible) {
    const distribution = distributions
      .get(result.statistic)!
      .sort((a, b) => a - b);
    if (result.invalid) result.reason = "undefined_resamples";
    else if (distribution[0] === distribution[distribution.length - 1])
      result.reason = "degenerate_resamples";
    else {
      const alpha = (1 - options.confidence) / 2;
      result.lower = percentile(distribution, alpha);
      result.upper = percentile(distribution, 1 - alpha);
    }
  }
  return results;
}
