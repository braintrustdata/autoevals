import { ClosedQA } from "./llm";
import { Levenshtein } from "./string";

test("Partial Test", async () => {
  const levenshteinBasic = await Levenshtein({
    output: "abc",
    expected: "abcd",
  });
  const levenshteinPartial = await Levenshtein.partial({ expected: "abcd" })({
    output: "abc",
  });
  expect(levenshteinBasic.score).toBeDefined();
  expect(levenshteinPartial.score).toBeDefined();
  expect(levenshteinPartial.score).toEqual(levenshteinBasic.score);
  expect(levenshteinBasic.name).toEqual(levenshteinPartial.name);
  expect(levenshteinBasic.name).toEqual("Levenshtein");

  // Now do the same with ClosedQA which is an "LLM" scorer
  const closedQABasic = await ClosedQA({
    criteria: "Is the answer correct?",
    input: "What is 1+1?",
    output: "2",
  });
  const closedQAPartial = await ClosedQA.partial({
    criteria: "Is the answer correct?",
  })({
    input: "What is 1+1?",
    output: "2",
  });
  expect(closedQABasic.score).toBeDefined();
  expect(closedQAPartial.score).toBeDefined();
  expect(closedQAPartial.score).toEqual(closedQABasic.score);
  expect(closedQABasic.name).toEqual(closedQAPartial.name);
  expect(closedQABasic.name).toEqual("ClosedQA");
});
