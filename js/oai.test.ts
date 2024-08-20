import { buildOpenAIClient } from "./oai";

describe("OAI", () => {
  test("should use Azure OpenAI", async () => {
    /* TODO: I wasn't sure how to propagate these
    variables from the environment to the test.
    I just left placeholders here for now.
    You can plug in your own valid Azure OpenAI credentials
    to make sure it works.

    Also there weren't any tests for the `oai.ts` file
    previously, so not sure if we want to add this now.

    Defer to the Braintrust team on how to handle integration
    of Azure OpenAI support.
    */
    const client = buildOpenAIClient({
      azureOpenAi: {
        apiKey: "<some api key>",
        endpoint: "https://<some resource>.openai.azure.com/",
        apiVersion: "<some valid version>",
      },
    });
    const {
      choices: [
        {
          message: { content },
        },
      ],
    } = await client.chat.completions.create({
      model: "gpt-4o-mini", // or whatever your relevant Azure OpenAI deployment name is
      messages: [{ role: "system", content: "Hello" }],
    });
    expect(content).toBeTruthy();
  });
});
