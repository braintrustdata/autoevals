import { expect, test, describe, beforeEach } from "vitest";
import { init, getDefaultModel, getDefaultEmbeddingModel } from "./oai";
import { OpenAI } from "openai";

describe("init with defaultModel parameter", () => {
  beforeEach(() => {
    // Reset to defaults
    init();
  });

  test("string form sets completion model (backward compatible)", () => {
    init({
      defaultModel: "gpt-4-turbo",
    });

    expect(getDefaultModel()).toBe("gpt-4-turbo");
    expect(getDefaultEmbeddingModel()).toBe("text-embedding-ada-002"); // Default
  });

  test("object form can set completion model only", () => {
    init({
      defaultModel: {
        completion: "gpt-4-turbo",
      },
    });

    expect(getDefaultModel()).toBe("gpt-4-turbo");
  });

  test("object form can set embedding model only", () => {
    init({
      defaultModel: {
        embedding: "text-embedding-3-large",
      },
    });

    expect(getDefaultEmbeddingModel()).toBe("text-embedding-3-large");
    // Completion model should remain at default since we didn't update it
    expect(getDefaultModel()).toBe("gpt-5-mini");
  });

  test("object form can set both models", () => {
    init({
      defaultModel: {
        completion: "claude-3-5-sonnet-20241022",
        embedding: "text-embedding-3-large",
      },
    });

    expect(getDefaultModel()).toBe("claude-3-5-sonnet-20241022");
    expect(getDefaultEmbeddingModel()).toBe("text-embedding-3-large");
  });

  test("partial updates preserve unspecified models", () => {
    // First set completion model
    init({
      defaultModel: {
        completion: "gpt-4-turbo",
      },
    });

    expect(getDefaultModel()).toBe("gpt-4-turbo");
    expect(getDefaultEmbeddingModel()).toBe("text-embedding-ada-002");

    // Then set only embedding model - completion should remain unchanged
    init({
      defaultModel: {
        embedding: "text-embedding-3-large",
      },
    });

    expect(getDefaultModel()).toBe("gpt-4-turbo"); // Should still be gpt-4-turbo
    expect(getDefaultEmbeddingModel()).toBe("text-embedding-3-large");
  });

  test("falls back to defaults when not set", () => {
    init();

    expect(getDefaultModel()).toBe("gpt-5-mini");
    expect(getDefaultEmbeddingModel()).toBe("text-embedding-ada-002");
  });

  test("string form resets embedding model to default", () => {
    // First set both models
    init({
      defaultModel: {
        completion: "gpt-4-turbo",
        embedding: "text-embedding-3-large",
      },
    });

    expect(getDefaultModel()).toBe("gpt-4-turbo");
    expect(getDefaultEmbeddingModel()).toBe("text-embedding-3-large");

    // Then use string form - should reset embedding to default
    init({
      defaultModel: "claude-3-5-sonnet-20241022",
    });

    expect(getDefaultModel()).toBe("claude-3-5-sonnet-20241022");
    expect(getDefaultEmbeddingModel()).toBe("text-embedding-ada-002"); // Reset to default
  });
});
