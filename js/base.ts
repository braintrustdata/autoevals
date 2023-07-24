export interface Score {
  name: string;
  score: number;
  metadata?: Record<string, unknown>;
  error?: unknown;
}

export type ScorerArgs<Output, Extra> = {
  output: Output;
  expected?: Output;
} & Extra;

export type Scorer<Output, Extra> =
  | ((args: ScorerArgs<Output, Extra>) => Promise<Score>)
  | ((args: ScorerArgs<Output, Extra>) => Score);
