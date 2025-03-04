import { setupServer } from "msw/node";

export const server = setupServer();

beforeAll(() => {
  server.listen({
    onUnhandledRequest: async (req) => {
      console.log("Unhandled request:", req.method, req.url);

      // Passthrough the request and get the response
      const response = await fetch(req);
      const clonedResponse = response.clone();

      // Log the response body
      try {
        const body = await clonedResponse.text();
        console.log("Response body:", body);
      } catch (error) {
        console.error("Failed to read response body:", error);
      }

      // Return the original response
      return response;
    },
  });
});

afterEach(() => {
  server.resetHandlers();
});

afterAll(() => {
  server.close();
});
