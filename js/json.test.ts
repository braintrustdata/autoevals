import { JSONDiff, ValidJSON } from "./json";

test("JSON String Test", async () => {
  const cases = [
    { a: "", b: "", expected: 1 },
    { a: "", b: "a", expected: 0 },
    { a: "a", b: "", expected: 0 },
    { a: "a", b: "a", expected: 1 },
    { a: "a", b: "b", expected: 0 },
    { a: "ab", b: "ac", expected: 0.5 },
    { a: "ac", b: "bc", expected: 0.5 },
    { a: "abc", b: "axc", expected: 0.66667 },
    { a: "xabxcdxxefxgx", b: "1ab2cd34ef5g6", expected: 0.53846 },
  ];

  for (const { a, b, expected } of cases) {
    const score = (await JSONDiff({ output: a, expected: b })).score;
    expect(score).toBeCloseTo(expected);
  }
});

test("JSON Object Test", async () => {
  const cases = [
    { a: null, b: null, expected: 1 },
    { a: undefined, b: null, expected: 1 },
    { a: "", b: null, expected: 0 },
    { a: [], b: {}, expected: 0 },
    { a: [], b: [], expected: 1 },
    { a: {}, b: {}, expected: 1 },
    { a: { a: 1 }, b: { a: 1 }, expected: 1 },
    { a: { a: 1 }, b: { a: 2 }, expected: 0.66667 },
    { a: { a: 1 }, b: ["a", 1], expected: 0.5714285714285714 },
    { a: { a: 1 }, b: { b: { a: 1 } }, expected: 0 },
    { a: { a: 1 }, b: { a: null }, expected: 0 },
    {
      a: { mapping: { a: "foo", b: "bar" } },
      b: { mapping: { a: "Foo", b: "Bar" }, Extra: 5 },
      expected: 0.33333333333333337,
    },
  ];

  for (const { a, b, expected } of cases) {
    const score = (await JSONDiff({ output: a, expected: b })).score;
    expect(score).toBeCloseTo(expected);
  }
});

test("Valid JSON Test", async () => {
  const cases = [
    { output: "1", expected: 0 },
    { output: '{ "a": 1, "b": "hello" }', expected: 1 },
    { output: '[{ "a": 1 }]', expected: 1 },
    { output: '[{ "a": 1 }', expected: 0 },
    {
      output: '{ "mapping": { "a": "foo", "b": "bar" }, "extra": 4 }',
      expected: 1,
    },
    {
      output: '{ mapping: { "a": "foo", "b": "bar" }, "extra": 4 }',
      expected: 0,
    },
    {
      output: '{"a":"1"}',
      expected: 1,
      schema: {
        type: "object",
        properties: {
          a: { type: "string" },
        },
        required: ["a"],
      },
    },
    {
      output: '{ "a": "1", "b": "1" }',
      expected: 0,
      schema: {
        type: "object",
        properties: {
          a: { type: "string" },
          b: { type: "number" },
        },
        required: ["a", "b"],
      },
    },
    {
      output: '[{ "a": "1" }, { "a": "1", "b": 22 }]',
      expected: 1,
      schema: {
        type: "array",
        items: {
          type: "object",
          properties: {
            a: { type: "string" },
            b: { type: "number" },
          },
          required: ["a"],
        },
        uniqueItems: true,
      },
    },
  ];

  for (const { output, expected, schema } of cases) {
    const score = (await ValidJSON({ output, schema })).score;
    expect(score).toEqual(expected);
  }
});
