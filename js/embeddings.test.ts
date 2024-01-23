import { EmbeddingSimilarity } from "./string.js";

const SYNONYMS = [
  {
    word: "water",
    synonyms: ["water", "H2O", "agua"],
  },
  {
    word: "fire",
    synonyms: ["fire", "flame"],
  },
  {
    word: "earth",
    synonyms: ["earth", "Planet Earth"],
  },
];

const UNRELATED = [
  "water",
  "The quick brown fox jumps over the lazy dog",
  "I like to eat apples",
];

test("Embeddings Test", async () => {
  const prefix = "resource type: ";
  for (const { word, synonyms } of SYNONYMS) {
    for (const synonym of synonyms) {
      const result = await EmbeddingSimilarity({
        prefix,
        output: word,
        expected: synonym,
      });
      expect(result.score).toBeGreaterThan(0.6);
    }
  }

  for (let i = 0; i < UNRELATED.length; i++) {
    for (let j = 0; j < UNRELATED.length; j++) {
      if (i == j) {
        continue;
      }

      const word1 = UNRELATED[i];
      const word2 = UNRELATED[j];
      const result = await EmbeddingSimilarity({
        prefix,
        output: word1,
        expected: word2,
      });
      expect(result.score).toBeLessThan(0.5);
    }
  }
}, 600000);
