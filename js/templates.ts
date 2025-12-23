import { z } from "zod";
import * as yaml from "js-yaml";

import battle from "../templates/battle.yaml";
import closed_q_a from "../templates/closed_q_a.yaml";
import factuality from "../templates/factuality.yaml";
import humor from "../templates/humor.yaml";
import possible from "../templates/possible.yaml";
import security from "../templates/security.yaml";
import sql from "../templates/sql.yaml";
import summary from "../templates/summary.yaml";
import translation from "../templates/translation.yaml";

export const modelGradedSpecSchema = z.object({
  prompt: z.string(),
  choice_scores: z.record(z.string(), z.number()),
  model: z.string().optional(),
  use_cot: z.boolean().optional(),
  temperature: z.number().optional(),
});

export type ModelGradedSpec = z.infer<typeof modelGradedSpecSchema>;

const templateStrings = {
  battle,
  closed_q_a,
  factuality,
  humor,
  possible,
  security,
  sql,
  summary,
  translation,
} as const;

// eslint-disable-next-line @typescript-eslint/consistent-type-assertions
export const templates = Object.fromEntries(
  Object.entries(templateStrings).map(([name, template]) => [
    name,
    modelGradedSpecSchema.parse(
      typeof template === "string" ? yaml.load(template) : template,
    ),
  ]),
) as Record<keyof typeof templateStrings, ModelGradedSpec>;
