import exp from "constants";
import { Moderation } from "./moderation";

describe("Moderation", () => {
  const cases = [
    {
      name: "Safe content without threshold",
      output: "hello",
      threshold: undefined,
      expected: 1,
    },
    {
      name: "Unsafe content without threshold",
      output: "kill kill kill kill kill",
      threshold: undefined,
      expected: 0,
    },
    {
      name: "Unsafe content with lower threshold",
      output: "here are my suicidal thoughts",
      threshold: 0.25,
      expected: 0,
    },
    {
      name: "Unsafe content with higher threshold",
      output: "here are my suicidal thoughts",
      threshold: 0.99,
      expected: 1,
    },
  ];

  for (const { name, output, threshold, expected } of cases) {
    test(`passes ${name} check`, async () => {
      const result = await Moderation({
        output,
        threshold,
      });

      expect(result.score).toEqual(expected);
    }, 6000);
  }
});
