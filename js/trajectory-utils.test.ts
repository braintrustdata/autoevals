import { readFileSync } from "fs";
import { join } from "path";
import { expect, test } from "vitest";
import { computeTrajectoryTemplateVars } from "./trajectory-utils";

const fixtures = JSON.parse(
  readFileSync(join(__dirname, "../test-data/trajectories.json"), "utf8"),
);

test.each(Object.entries(fixtures))("%s", (_, fixture: any) => {
  const { trajectories, ...expected } = fixture;
  expect(computeTrajectoryTemplateVars(trajectories)).toEqual(expected);
});
