export interface EnvI {
  OPENAI_API_KEY?: string;
  OPENAI_BASE_URL?: string;
  BRAINTRUST_API_KEY?: string;
}

export const Env: EnvI =
  typeof process === "undefined"
    ? {}
    : {
        OPENAI_API_KEY: process.env.OPENAI_API_KEY,
        OPENAI_BASE_URL: process.env.OPENAI_BASE_URL,
        BRAINTRUST_API_KEY: process.env.BRAINTRUST_API_KEY,
      };
