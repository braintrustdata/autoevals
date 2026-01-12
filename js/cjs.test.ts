import { execSync } from "child_process";
import { existsSync } from "fs";
import { describe, expect, test, beforeAll } from "vitest";

describe("CJS compatibility", () => {
  beforeAll(() => {
    // Ensure the build exists
    if (!existsSync("./jsdist/index.js")) {
      execSync("pnpm run build", { stdio: "inherit" });
    }
  });

  test("can require the CJS bundle", () => {
    // Test that CJS require works, including dependencies like linear-sum-assignment (#152)
    const result = execSync(
      `node -e "const m = require('./jsdist/index.js'); console.log(JSON.stringify(Object.keys(m).sort()))"`,
      { encoding: "utf-8" },
    );
    const exports = JSON.parse(result.trim());

    // Verify key exports are present
    expect(exports).toContain("Factuality");
    expect(exports).toContain("ListContains");
    expect(exports).toContain("ContextRelevancy");
    expect(exports).toContain("Levenshtein");
  });

  test("ListContains works in CJS (uses linear-sum-assignment)", () => {
    // ListContains uses linear-sum-assignment internally, which was ESM-only (#152)
    const result = execSync(
      `node -e "
        const { ListContains } = require('./jsdist/index.js');
        ListContains({ output: ['a', 'b'], expected: ['a', 'c'] }).then(r => {
          console.log(JSON.stringify({ score: r.score, name: r.name }));
        });
      "`,
      { encoding: "utf-8" },
    );
    const parsed = JSON.parse(result.trim());

    expect(parsed.name).toBe("ListContains");
    expect(parsed.score).toBeGreaterThanOrEqual(0);
    expect(parsed.score).toBeLessThanOrEqual(1);
  });
});
