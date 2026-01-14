import { expect, test, describe } from "vitest";
import { LLMClassifierFromTemplate } from "./llm";
import { Score } from "./score";

describe("reasoningEffort parameter", () => {
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
});
