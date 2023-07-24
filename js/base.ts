export interface Score {
  score: number;
  metadata?: Record<string, unknown>;
  error?: unknown;
}

export type ScorerArgs<Output, Extra> = {
  output: Output;
  expected?: Output;
} & Extra;

export type Scorer<Output = string, Extra = {}> =
  | ((args: ScorerArgs<Output, Extra>) => Promise<Score>)
  | ((args: ScorerArgs<Output, Extra>) => Score);
