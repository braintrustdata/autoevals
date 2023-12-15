import { defineConfig } from "tsup";

export default defineConfig([
  {
    entry: ["js/index.ts"],
    format: ["cjs", "esm"],
    outDir: "jsdist",
    dts: true,
    loader: {
      ".yaml": "text",
    },
  },
]);
