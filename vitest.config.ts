import { defineConfig } from "vitest/config";
import yaml from "@rollup/plugin-yaml";

export default defineConfig({
  plugins: [yaml()],
  test: {
    environment: "node",
    testTimeout: 15_000,
  },
});
