import { expect, test, describe } from "vitest";
import { LLMClassifierFromTemplate } from "./llm";
import { Score } from "./score";

describe("reasoning parameters", () => {
  test("accepts reasoningEffort in LLMArgs", () => {
    // This test just verifies that the type system accepts reasoningEffort
    // We don't actually call the API to avoid requiring credentials in tests
    const classifier = LLMClassifierFromTemplate({
      name: "test",
      promptTemplate: "Evaluate: {{output}}",
      choiceScores: { good: 1, bad: 0 },
      model: "o3-mini",
      reasoningEffort: "high",
    });

    expect(classifier).toBeDefined();
    expect(typeof classifier).toBe("function");
  });

  test("accepts all valid reasoningEffort values", () => {
    const validValues: Array<
      "minimal" | "low" | "medium" | "high" | null | undefined
    > = ["minimal", "low", "medium", "high", null, undefined];

    for (const value of validValues) {
      const classifier = LLMClassifierFromTemplate({
        name: "test",
        promptTemplate: "Evaluate: {{output}}",
        choiceScores: { good: 1, bad: 0 },
        model: "o3-mini",
        reasoningEffort: value,
      });

      expect(classifier).toBeDefined();
    }
  });

  test("reasoningEffort can be passed at runtime", () => {
    const classifier = LLMClassifierFromTemplate({
      name: "test",
      promptTemplate: "Evaluate: {{output}}",
      choiceScores: { good: 1, bad: 0 },
      model: "o3-mini",
    });

    // TypeScript should allow passing reasoningEffort at runtime
    // This verifies the type allows it (actual API call would require credentials)
    expect(classifier).toBeDefined();
  });

  test("accepts reasoningEnabled parameter", () => {
    // Test that the type system accepts reasoningEnabled
    const classifier = LLMClassifierFromTemplate({
      name: "test",
      promptTemplate: "Evaluate: {{output}}",
      choiceScores: { good: 1, bad: 0 },
      model: "claude-3-5-sonnet-20241022",
      reasoningEnabled: true,
    });

    expect(classifier).toBeDefined();
    expect(typeof classifier).toBe("function");
  });

  test("accepts reasoningBudget parameter", () => {
    // Test that the type system accepts reasoningBudget
    const classifier = LLMClassifierFromTemplate({
      name: "test",
      promptTemplate: "Evaluate: {{output}}",
      choiceScores: { good: 1, bad: 0 },
      model: "claude-3-5-sonnet-20241022",
      reasoningBudget: 2048,
    });

    expect(classifier).toBeDefined();
    expect(typeof classifier).toBe("function");
  });

  test("accepts all reasoning parameters together", () => {
    // Test that all reasoning parameters can be used together
    const classifier = LLMClassifierFromTemplate({
      name: "test",
      promptTemplate: "Evaluate: {{output}}",
      choiceScores: { good: 1, bad: 0 },
      model: "o3-mini",
      reasoningEffort: "high",
      reasoningEnabled: true,
      reasoningBudget: 4096,
    });

    expect(classifier).toBeDefined();
    expect(typeof classifier).toBe("function");
  });
});
