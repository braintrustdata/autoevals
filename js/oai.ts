import * as path from "path";
import * as sqlite3 from "sqlite3";
import * as fs from "fs";
import * as os from "os";

import { ChatCompletionRequestMessage, Configuration, OpenAIApi } from "openai";

let _cache_dir: string | null = null;

export function setCacheDir(path: string) {
  _cache_dir = path;
}

let _db: sqlite3.Database | null = null;
async function openCache() {
  if (_cache_dir === null) {
    _cache_dir = path.join(os.homedir(), ".cache", "braintrust");
  }

  if (_db === null) {
    fs.mkdirSync(_cache_dir, { recursive: true });
    const oai_cache = path.join(_cache_dir, "oai.sqlite");
    _db = new sqlite3.Database(oai_cache);

    await new Promise((resolve, reject) => {
      _db!.run(
        "CREATE TABLE IF NOT EXISTS cache (params text, response text)",
        (err: any) => {
          if (err) {
            reject(err);
          } else {
            resolve(undefined);
          }
        }
      );
    });
  }
  return _db!;
}

let _openai: OpenAIApi | null = null;
export function openAI() {
  if (_openai === null && process.env.OPENAI_API_KEY) {
    const config = new Configuration({ apiKey: process.env.OPENAI_API_KEY });
    _openai = new OpenAIApi(config);
  }
  return _openai;
}

export async function cachedChatCompletion(args: any) {
  const db = await openCache();

  const param_key = JSON.stringify(args);
  const query = `SELECT response FROM "cache" WHERE params=?`;
  const resp = await new Promise((resolve, reject) => {
    db.get(query, [param_key], (err, row) => {
      if (err) {
        reject(err);
      } else {
        resolve(row);
      }
    });
  });
  if (resp) {
    return JSON.parse((resp as any).response);
  }

  const openai = openAI();
  if (openai === null) {
    return new Error("OPENAI_API_KEY not set");
  }

  const completion = await openai.createChatCompletion(args);
  const data = completion.data;
  db.run(`INSERT INTO "cache" VALUES (?, ?)`, [
    param_key,
    JSON.stringify(data),
  ]);

  return data;
}
