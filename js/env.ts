export interface EnvI {
  OPENAI_API_KEY?: string;
  OPENAI_BASE_URL?: string;
}

export const Env: EnvI = {
  OPENAI_API_KEY: undefined,
  OPENAI_BASE_URL: undefined,
};
