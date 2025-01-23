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
import { ScorerWithPartial } from "./partial";
import { Moderation } from "./moderation";
import { ExactMatch } from "./value";
import { ModelGradedSpec, templates } from "./templates";

interface AutoevalMethod {
  method: ScorerWithPartial<any, any>;
  description: string;
  template?: ModelGradedSpec;
  requiresExtraParams?: boolean;
}

export const Evaluators: {
  label: string;
  methods: AutoevalMethod[];
}[] = [
  {
    label: "LLM-as-a-Judge",
    methods: [
      {
        method: Battle,
        description:
          "Test whether an output _better_ performs the `instructions` than the original (expected) value.",
        template: templates.battle,
        requiresExtraParams: true,
      },
      {
        method: ClosedQA,
        description:
          "Test whether an output answers the `input` using knowledge built into the model. You can specify `criteria` to further constrain the answer.",
        template: templates.closed_q_a,
        requiresExtraParams: true,
      },
      {
        method: Humor,
        description: "Test whether an output is funny.",
        template: templates.humor,
      },
      {
        method: Factuality,
        description:
          "Test whether an output is factual, compared to an original (`expected`) value.",
        template: templates.factuality,
      },
      {
        method: Moderation,
        description:
          "A scorer that uses OpenAI's moderation API to determine if AI response contains ANY flagged content.",
      },
      {
        method: Possible,
        description:
          "Test whether an output is a possible solution to the challenge posed in the input.",
        template: templates.possible,
      },
      {
        method: Security,
        description: "Test whether an output is malicious.",
        template: templates.security,
      },
      {
        method: Sql,
        description:
          "Test whether a SQL query is semantically the same as a reference (output) query.",
        template: templates.sql,
      },
      {
        method: Summary,
        description:
          "Test whether an output is a better summary of the `input` than the original (`expected`) value.",
        template: templates.summary,
      },
      {
        method: Translation,
        description:
          "Test whether an `output` is as good of a translation of the `input` in the specified `language` as an expert (`expected`) value.",
        template: templates.translation,
        requiresExtraParams: true,
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
        requiresExtraParams: true,
      },
      {
        method: ContextRelevancy,
        description:
          "Extracts relevant sentences from the provided context that are absolutely required to answer the given question.",
        requiresExtraParams: true,
      },
      {
        method: ContextRecall,
        description:
          "Analyzes each sentence in the answer and classifies if the sentence can be attributed to the given context or not.",
        requiresExtraParams: true,
      },
      {
        method: ContextPrecision,
        description:
          "Verifies if the context was useful in arriving at the given answer.",
        requiresExtraParams: true,
      },
      {
        method: AnswerRelevancy,
        description:
          "Scores the relevancy of the generated answer to the given question.",
        requiresExtraParams: true,
      },
      {
        method: AnswerSimilarity,
        description:
          "Scores the semantic similarity between the generated answer and ground truth.",
        requiresExtraParams: true,
      },
      {
        method: AnswerCorrectness,
        description:
          "Measures answer correctness compared to ground truth using a weighted average of factuality and semantic similarity.",
        requiresExtraParams: true,
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
        method: ExactMatch,
        description:
          "Compares two values for exact equality. If the values are objects, they are converted to JSON strings before comparison.",
      },
      {
        method: NumericDiff,
        description: "Compares numbers by normalizing their difference.",
      },
    ],
  },
];
