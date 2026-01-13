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
    // Bundle ESM-only dependencies to ensure CJS compatibility (#152)
    noExternal: ["linear-sum-assignment"],
  },
]);
