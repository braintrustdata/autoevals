/** @type {import('ts-jest').JestConfigWithTsJest} */
module.exports = {
  preset: "ts-jest",
  testEnvironment: "node",
  // https://www.npmjs.com/package/jest-transform-yaml
  transform: {
    "\\.yaml$": "jest-text-transformer",
  },
};
