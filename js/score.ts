// Keep this in sync with @braintrust/core

export interface Score {
  name: string;
  score: number | null;
  metadata?: Record<string, unknown>;
  // DEPRECATION_NOTICE: this field is deprecated, as errors are propagated up to the caller.
  /**
   * @deprecated
   */
  error?: unknown;
}

export type ScorerArgs<Output, Extra> = {
  output: Output;
  expected?: Output;
} & Extra;

export type Scorer<Output, Extra> = (
  args: ScorerArgs<Output, Extra>,
) => Score | Promise<Score>;
