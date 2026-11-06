export { analyzeConsistency } from "./analysis";
export { ratingFromScore, missingRating, errorRating } from "./adapters";
export {
  analyzeConsistencyJSON,
  inputToJSON,
  reportToJSON,
} from "./serialization";
export { consistencyScores, renderConsistencyMarkdown } from "./reporting";
export type {
  Rating,
  ScoreMapping,
  ConsistencyPlan,
  ConsistencyLimits,
  ConsistencyGroup,
  ConsistencyGate,
  ConsistencyBootstrap,
  ConsistencyReport,
  Statistic,
  StatisticName,
  ConsistencyInterval,
  Coverage,
  GroupReport,
  GateResult,
  ReferenceAccuracy,
  References,
} from "./types";
