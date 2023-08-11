import { CreateChatCompletionResponse } from "openai";
import { CachedLLMParams, ChatCache } from "../js/oai";
// @ts-ignore
import * as sqlite3 from "sqlite3";
import * as fs from 'node:fs';
import * as path from 'node:path';
import * as os from 'node:os';

// TODO where to put this? separate package? Remove in favor of isomorphic caches?

export class SQLiteCache implements ChatCache {
  private db: sqlite3.Database;
  private cacheDir: string;
  private initializePromise: Promise<void> | undefined;

  constructor({ cacheDir }: { cacheDir?: string } = {}) {
    this.cacheDir = cacheDir ?? path.join(os.homedir(), ".cache", "braintrust");
    this.db = this.openCache();
  }

  openCache() {
    fs.mkdirSync(this.cacheDir, { recursive: true });
    const oai_cache = path.join(this.cacheDir, "oai.sqlite");
    const db = new sqlite3.Database(oai_cache);

    this.initCache().catch(err => {
      console.error(`Failed to initialize LLM cache: ${err}`);
    });

    return db;
  }

  async initCache() {
    this.initializePromise = new Promise((resolve, reject) => {
      this.db!.run(
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

      return this.initializePromise;
  }

  async get(params: CachedLLMParams): Promise<CreateChatCompletionResponse | null> {
    await this.initializePromise;

    const param_key = JSON.stringify(params);
    const query = `SELECT response FROM "cache" WHERE params=?`;
    const resp = await new Promise((resolve, reject) => {
      this.db.get(query, [param_key], (err, row) => {
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

    return null;
  }
  async set(params: CachedLLMParams, response: CreateChatCompletionResponse): Promise<void> {
    await this.initializePromise;

    const param_key = JSON.stringify(params);

    this.db.run(`INSERT INTO "cache" VALUES (?, ?)`, [
      param_key,
      JSON.stringify(response),
    ]);
  }
}
