import { Scorer } from "@braintrust/core";
import { JSONDiff, ValidJSON } from "./json";
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
import {
  ContextEntityRecall,
  ContextRelevancy,
  ContextRecall,
  ContextPrecision,
  AnswerRelevancy,
  AnswerSimilarity,
  AnswerCorrectness,
} from "./ragas";
import { ListContains } from "./list";

interface AutoevalMethod {
  method: Scorer<any, any>;
  description: string;
}

export const Evaluators: {
  label: string;
  methods: AutoevalMethod[];
}[] = [
  {
    label: "Model-based classification",
    methods: [
      {
        method: Battle,
        description:
          "Test whether an output _better_ performs the `instructions` than the original (expected) value.",
      },
      {
        method: ClosedQA,
        description:
          "Test whether an output answers the `input` using knowledge built into the model. You can specify `criteria` to further constrain the answer.",
      },
      {
        method: Humor,
        description: "Test whether an output is funny.",
      },
      {
        method: Factuality,
        description:
          "Test whether an output is factual, compared to an original (`expected`) value.",
      },
      {
        method: Possible,
        description:
          "Test whether an output is a possible solution to the challenge posed in the input.",
      },
      {
        method: Security,
        description: "Test whether an output is malicious.",
      },
      {
        method: Sql,
        description:
          "Test whether a SQL query is semantically the same as a reference (output) query.",
      },
      {
        method: Summary,
        description:
          "Test whether an output is a better summary of the `input` than the original (`expected`) value.",
      },
      {
        method: Translation,
        description:
          "Test whether an `output` is as good of a translation of the `input` in the specified `language` as an expert (`expected`) value.",
      },
    ],
  },
  {
    label: "RAG",
    methods: [
      {
        method: ContextEntityRecall,
        description:
          "Estimates context recall by estimating TP and FN using annotated answer and retrieved context.",
      },
      {
        method: ContextRelevancy,
        description:
          "Extracts relevant sentences from the provided context that are absolutely required to answer the given question.",
      },
      {
        method: ContextRecall,
        description:
          "Analyzes each sentence in the answer and classifies if the sentence can be attributed to the given context or not.",
      },
      {
        method: ContextPrecision,
        description:
          "Verifies if the context was useful in arriving at the given answer.",
      },
      {
        method: AnswerRelevancy,
        description:
          "Scores the relevancy of the generated answer to the given question.",
      },
      {
        method: AnswerSimilarity,
        description:
          "Scores the semantic similarity between the generated answer and ground truth.",
      },
      {
        method: AnswerCorrectness,
        description:
          "Measures answer correctness compared to ground truth using a weighted average of factuality and semantic similarity.",
      },
    ],
  },
  {
    label: "Composite",
    methods: [
      {
        method: ListContains,
        description:
          "Semantically evaluates the overlap between two lists of strings using pairwise similarity and Linear Sum Assignment.",
      },
      {
        method: ValidJSON,
        description:
          "Evaluates the validity of JSON output, optionally validating against a JSON Schema definition.",
      },
    ],
  },
  {
    label: "Embeddings",
    methods: [
      {
        method: EmbeddingSimilarity,
        description:
          "Evaluates the semantic similarity between two embeddings using cosine distance.",
      },
    ],
  },
  {
    label: "Heuristic",
    methods: [
      {
        method: JSONDiff,
        description:
          "Compares JSON objects using customizable comparison methods for strings and numbers.",
      },
      {
        method: Levenshtein,
        description: "Uses the Levenshtein distance to compare two strings.",
      },
      {
        method: NumericDiff,
        description: "Compares numbers by normalizing their difference.",
      },
    ],
  },
];
