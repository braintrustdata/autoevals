import { duckq, getDuckDBConn } from "./duckdb";

import { z } from "zod";
import { Factuality } from "autoevals";
import { coqaSchema, FactualityCase } from "./datasets";
import path from "path";
import fs from "fs";

async function coqaFactuality(): Promise<FactualityCase[]> {
  const conn = getDuckDBConn();
  const df = z.array(coqaSchema).parse(
    await duckq(
      conn,
      `SELECT * FROM 'hf://datasets/stanfordnlp/coqa/data/validation-00000-of-00001.parquet'
        LIMIT 10`
    )
  );

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
        output: `${answers.input_text[0]} and ${answers.input_text[1]}`,
        expected: answers.input_text[0],
      },
      expected: 0.6,
      metadata,
    });
  }

  return cases;
}

const dataDir = path.join(__dirname, "../data");
function saveFile(cases: unknown[], fname: string) {
  fs.writeFileSync(path.join(dataDir, fname), JSON.stringify(cases, null, 2));
}

async function main() {
  if (!fs.existsSync(dataDir)) {
    fs.mkdirSync(dataDir, { recursive: true });
  }

  saveFile(await coqaFactuality(), "coqa.json");
}

main();
