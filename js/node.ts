import { Env } from "./env.js";
Env.OPENAI_API_KEY = process.env.OPENAI_API_KEY;
Env.OPENAI_BASE_URL = process.env.OPENAI_BASE_URL;

export * from "./index.js";
