import { Scorer } from "@braintrust/core";
import { JSONDiff } from "./json";
import {
  Battle,
  ClosedQA,
  Factuality,
  Humor,
  Possible,
  Security,
  Sql,
  Summary,
  Translation,
} from "./llm";
import { NumericDiff } from "./number";
import { EmbeddingSimilarity, Levenshtein } from "./string";

export const Evaluators: {
  label: string;
  methods: Scorer<any, any>[];
}[] = [
  {
    label: "Model-based classification",
    methods: [
      Battle,
      ClosedQA,
      Humor,
      Factuality,
      Possible,
      Security,
      Sql,
      Summary,
      Translation,
    ],
  },
  {
    label: "Embeddings",
    methods: [EmbeddingSimilarity],
  },
  {
    label: "Heuristic",
    methods: [JSONDiff, Levenshtein, NumericDiff],
  },
];
