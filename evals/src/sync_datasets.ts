import { duckq, getDuckDBConn } from "./duckdb";

import { z } from "zod";
import {
  coqaSchema,
  dataDir,
  FactualityCase,
  ContextRelevancyCase,
  ClosedQACase,
} from "./datasets";
import path from "path";
import fs from "fs";

async function getCoqa(): Promise<z.infer<typeof coqaSchema>[]> {
  const conn = getDuckDBConn();
  return z.array(coqaSchema).parse(
    await duckq(
      conn,
      `SELECT * FROM 'hf://datasets/stanfordnlp/coqa/data/validation-00000-of-00001.parquet'
        LIMIT 20`,
    ),
  );
}

async function coqaFactuality(): Promise<FactualityCase[]> {
  const df = await getCoqa();

  // For each question, capture the correct answer, make a superset by concatenating answers
  // together, and pick a different answer as a completely wrong one
  const cases: FactualityCase[] = [];
  for (let document = 0; document < df.length; document++) {
    const metadata = df[document];
    const { questions, answers } = metadata;

    cases.push({
      input: {
        input: questions[0],
        output: answers.input_text[0],
        expected: answers.input_text[0],
      },
      expected: 1,
      metadata,
    });

    cases.push({
      input: {
        input: questions[0],
        output: answers.input_text[1],
        expected: answers.input_text[0],
      },
      expected: 0,
      metadata,
    });

    cases.push({
      input: {
        input: questions[0],
        output: `${answers.input_text[1]} ${answers.input_text[0]} ${answers.input_text[2]}`,
        expected: answers.input_text[0],
      },
      expected: 0.6,
      metadata,
    });
  }

  return cases;
}

async function coqaContextRelevancy(): Promise<ContextRelevancyCase[]> {
  const df = await getCoqa();

  const cases: ContextRelevancyCase[] = [];
  for (const metadata of df) {
    const { story, questions, answers } = metadata;

    const input = questions[0];
    const contexts = answers.answer_start.map((answer_start, i) =>
      story.substring(answer_start, answers.answer_end[i]),
    );

    cases.push({
      input: {
        input,
        context: contexts[0],
      },
      expected: 1,
      metadata,
    });

    cases.push({
      input: {
        input,
        context: contexts[1],
      },
      expected: 0,
      metadata,
    });

    const concat = `${contexts[0]} ${contexts[1]}`;
    cases.push({
      input: {
        input,
        context: concat,
      },
      expected: contexts[0].length / concat.length,
      metadata,
    });
  }

  return cases;
}

async function coqaClosedQA(): Promise<ClosedQACase[]> {
  const df = await getCoqa();

  const cases: ClosedQACase[] = [];
  for (const metadata of df) {
    const { questions, answers, story } = metadata;

    const input = `Given the following context: ${story}, \n\n Answer the question: ${questions[0]}`;
    const criteria = "Is the answer correct?";
    cases.push({
      input: { input, output: answers.input_text[0], criteria },
      expected: 1,
      metadata,
    });
    cases.push({
      input: { input, output: answers.input_text[1], criteria },
      expected: 0,
      metadata,
    });
  }
  return cases;
}

function saveFile(cases: unknown[], fname: string) {
  fs.writeFileSync(path.join(dataDir, fname), JSON.stringify(cases, null, 2));
}

async function main() {
  if (!fs.existsSync(dataDir)) {
    fs.mkdirSync(dataDir, { recursive: true });
  }

  saveFile(await coqaFactuality(), "coqa-factuality.json");
  saveFile(await coqaContextRelevancy(), "coqa-context-relevancy.json");
  saveFile(await coqaClosedQA(), "coqa-closed-qa.json");
}

main();
