export interface EnvI {
  OPENAI_API_KEY?: string;
  OPENAI_BASE_URL?: string;
}

export const Env: EnvI =
  typeof process === "undefined"
    ? {}
    : {
        OPENAI_API_KEY: process.env.OPENAI_API_KEY,
        OPENAI_BASE_URL: process.env.OPENAI_BASE_URL,
      };
