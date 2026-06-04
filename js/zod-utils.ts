import { zodToJsonSchema as zodToJsonSchemaV3 } from "zod-to-json-schema";
import * as z3 from "zod/v3";
import * as z4 from "zod/v4";

function isZodV4(zodObject: z3.ZodType | z4.ZodType): zodObject is z4.ZodType {
  return (
    typeof zodObject === "object" &&
    zodObject !== null &&
    "_zod" in zodObject &&
    zodObject._zod !== undefined
  );
}

export function zodToJsonSchema(schema: z4.ZodType | z3.ZodType) {
  if (isZodV4(schema)) {
    return z4.toJSONSchema(schema, {
      target: "draft-7",
    });
  }

  return zodToJsonSchemaV3(schema);
}
