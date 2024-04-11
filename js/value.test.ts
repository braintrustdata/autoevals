import { ListContains } from "./list";
import { NumericDiff } from "./number";
import { LevenshteinScorer } from "./string";

test("Levenshtein Test", async () => {
  const cases = [
    { a: "", b: "", expected: 1 },
    { a: "", b: "a", expected: 0 },
    { a: "a", b: "", expected: 0 },
    { a: "a", b: "a", expected: 1 },
    { a: "a", b: "b", expected: 0 },
    { a: "ab", b: "ac", expected: 0.5 },
    { a: "ac", b: "bc", expected: 0.5 },
    { a: "abc", b: "axc", expected: 0.6666666666666667 },
    { a: "xabxcdxxefxgx", b: "1ab2cd34ef5g6", expected: 0.5384615384615384 },
  ];

  for (const { a, b, expected } of cases) {
    const score = (await LevenshteinScorer({ output: a, expected: b })).score;
    expect(score).toBeCloseTo(expected);
  }
});

test("Numeric Test", async () => {
  const cases = [
    { a: 0, b: 0, expected: 1 },
    { a: 0, b: 1, expected: 0 },
    { a: 1, b: 2, expected: 0.66667 },
    { a: 1.0, b: 2.0, expected: 0.66667 },
    { a: -1, b: 2, expected: 0 },
  ];

  for (const { a, b, expected } of cases) {
    console.log(a, b, expected);
    const score = (await NumericDiff({ output: a, expected: b })).score;
    expect(score).toBeCloseTo(expected);
  }
});

test("ListContains Test", async () => {
  const cases = [
    { a: [], b: [], expected: 1 },
    { a: ["0"], b: [], expected: 0 },
    { a: [], b: ["0"], expected: 0 },
    { a: ["a"], b: ["a"], expected: 1 },
    { a: ["a"], b: ["a", "b"], expected: 0.5 },
    { a: ["a", "b"], b: ["a"], expected: 0.5 },
    {
      a: [
        "workspaces",
        "section",
        "view",
        "others",
        "workspace",
        "team",
        "pinning",
      ],
      b: ["starred", "multiple different workspaces", "shortcuts"],
      expected: 0.1218,
    },
    {
      a: ["starred", "multiple different workspaces", "shortcuts"],
      b: [
        "workspaces",
        "section",
        "view",
        "others",
        "workspace",
        "team",
        "pinning",
      ],
      expected: 0.1218,
    },
  ];

  for (const { a, b, expected } of cases) {
    console.log(a, b, expected);
    const score = (await ListContains({ output: a, expected: b })).score;
    expect(score).toBeCloseTo(expected, 4);
  }

  expect(
    (
      await ListContains({
        output: ["a", "b"],
        expected: ["b"],
        allowExtraEntities: true,
      })
    ).score
  ).toBe(1);
});
