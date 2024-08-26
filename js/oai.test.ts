import { buildOpenAIClient } from "./oai";

describe.skip("OAI", () => {
  test("should use Azure OpenAI", async () => {
    /*
     * You can plug in your own valid Azure OpenAI info
     * to make sure it works.
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
      model: "<Azure OpenAI LLM deployment name>",
      messages: [{ role: "system", content: "Hello" }],
    });
    expect(content).toBeTruthy();
  });
});
