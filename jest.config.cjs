/** @type {import('ts-jest').JestConfigWithTsJest} */
module.exports = {
  testEnvironment: "node",
  transform: {
    "\\.yaml$": "jest-text-transformer",
    "\\.[jt]sx?$": "ts-jest",
  },
  globals: {
    "ts-jest": {
      useESM: true,
    },
  },
  extensionsToTreatAsEsm: [".ts"],
};
