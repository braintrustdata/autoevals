export const openaiClassifierShouldEvaluateTitles = [
  {
    id: "chatcmpl-B7WxpqqPbHYiAOPDl3ViYNalDFbce",
    object: "chat.completion",
    created: 1741134709,
    model: "gpt-3.5-turbo-0125",
    choices: [
      {
        index: 0,
        message: {
          role: "assistant",
          content: null,
          tool_calls: [
            {
              id: "call_OlUJAex0cWI84acfE0XydrHz",
              type: "function",
              function: {
                name: "select_choice",
                arguments:
                  '{"reasons":"Title 1: Pros - Clearly states the goal of standardizing error responses for better developer experience. Cons - Might be too specific and not catchy. Title 2: Pros - Short and simple. Cons - Lacks information about the issue.","choice":"1"}',
              },
            },
          ],
          refusal: null,
        },
        logprobs: null,
        finish_reason: "stop",
      },
    ],
    usage: {
      prompt_tokens: 354,
      completion_tokens: 58,
      total_tokens: 412,
      prompt_tokens_details: {
        cached_tokens: 0,
        audio_tokens: 0,
      },
      completion_tokens_details: {
        reasoning_tokens: 0,
        audio_tokens: 0,
        accepted_prediction_tokens: 0,
        rejected_prediction_tokens: 0,
      },
    },
    service_tier: "default",
    system_fingerprint: null,
  },
];

export const openaiClassifierShouldEvaluateTitlesWithCoT = [
  {
    id: "chatcmpl-B7XFw0OCpCbMVwLizRts3Cl72Obg0",
    object: "chat.completion",
    created: 1741135832,
    model: "gpt-4o-2024-08-06",
    choices: [
      {
        index: 0,
        message: {
          role: "assistant",
          content: null,
          tool_calls: [
            {
              id: "call_jUzxFALMTbpzGX4DfFH57VdI",
              type: "function",
              function: {
                name: "select_choice",
                arguments:
                  '{"reasons":"1. The issue description talks about the need to standardize error responses from GoTrue, Postgres, and Realtime APIs to improve developer experience (DX).\\n2. Title 1 directly mentions the key components involved (GoTrue, Postgres, and Realtime APIs) and the goal (better DX), which aligns well with the issue description.\\n3. Title 2, \\"Good title,\\" is vague and does not provide any information about the issue or its context.\\n4. Therefore, Title 1 is more descriptive and relevant to the issue at hand.","choice":"1"}',
              },
            },
          ],
          refusal: null,
        },
        logprobs: null,
        finish_reason: "stop",
      },
    ],
    usage: {
      prompt_tokens: 370,
      completion_tokens: 125,
      total_tokens: 495,
      prompt_tokens_details: {
        cached_tokens: 0,
        audio_tokens: 0,
      },
      completion_tokens_details: {
        reasoning_tokens: 0,
        audio_tokens: 0,
        accepted_prediction_tokens: 0,
        rejected_prediction_tokens: 0,
      },
    },
    service_tier: "default",
    system_fingerprint: "fp_eb9dce56a8",
  },
  {
    id: "chatcmpl-B7YPU81s7cb2uzlwJ8w9aS5qhfhtJ",
    object: "chat.completion",
    created: 1741140268,
    model: "gpt-4o-2024-08-06",
    choices: [
      {
        index: 0,
        message: {
          role: "assistant",
          content: null,
          tool_calls: [
            {
              id: "call_3Z63hgrYvLuSZKc2rrHAYLI4",
              type: "function",
              function: {
                name: "select_choice",
                arguments:
                  '{"reasons":"1. The issue description talks about the need to standardize error responses from GoTrue, Postgres, and Realtime APIs to improve developer experience (DX).\\n2. Title 1, \\"Good title,\\" is vague and does not convey any specific information about the issue. It does not mention the APIs involved or the purpose of the standardization.\\n3. Title 2, \\"Standardize error responses from GoTrue, Postgres, and Realtime APIs for better DX,\\" directly reflects the main goal of the issue, which is to standardize error responses for better developer experience. It also specifies the APIs involved, making it clear and informative.\\n4. Therefore, Title 2 is a better choice as it accurately and clearly describes the issue at hand.","choice":"2"}',
              },
            },
          ],
          refusal: null,
        },
        logprobs: null,
        finish_reason: "stop",
      },
    ],
    usage: {
      prompt_tokens: 370,
      completion_tokens: 164,
      total_tokens: 534,
      prompt_tokens_details: { cached_tokens: 0, audio_tokens: 0 },
      completion_tokens_details: {
        reasoning_tokens: 0,
        audio_tokens: 0,
        accepted_prediction_tokens: 0,
        rejected_prediction_tokens: 0,
      },
    },
    service_tier: "default",
    system_fingerprint: "fp_eb9dce56a8",
  },
  {
    id: "chatcmpl-B7YQ9ILZ9DJR2AjY2s4qU15Rc6qII",
    object: "chat.completion",
    created: 1741140309,
    model: "gpt-4o-2024-08-06",
    choices: [
      {
        index: 0,
        message: {
          role: "assistant",
          content: null,
          tool_calls: [
            {
              id: "call_CxDdx3i9eaHg81kYjQIICPfd",
              type: "function",
              function: { name: "select_choice", arguments: '{"choice":"1"}' },
            },
          ],
          refusal: null,
        },
        logprobs: null,
        finish_reason: "stop",
      },
    ],
    usage: {
      prompt_tokens: 292,
      completion_tokens: 6,
      total_tokens: 298,
      prompt_tokens_details: { cached_tokens: 0, audio_tokens: 0 },
      completion_tokens_details: {
        reasoning_tokens: 0,
        audio_tokens: 0,
        accepted_prediction_tokens: 0,
        rejected_prediction_tokens: 0,
      },
    },
    service_tier: "default",
    system_fingerprint: "fp_eb9dce56a8",
  },
  {
    id: "chatcmpl-B7YQa80DGu61zUWpdPtXRaJdRQz6l",
    object: "chat.completion",
    created: 1741140336,
    model: "gpt-4o-2024-08-06",
    choices: [
      {
        index: 0,
        message: {
          role: "assistant",
          content: null,
          tool_calls: [
            {
              id: "call_ksuniPMn2w99hFt5Z1mzhWMe",
              type: "function",
              function: { name: "select_choice", arguments: '{"choice":"2"}' },
            },
          ],
          refusal: null,
        },
        logprobs: null,
        finish_reason: "stop",
      },
    ],
    usage: {
      prompt_tokens: 292,
      completion_tokens: 6,
      total_tokens: 298,
      prompt_tokens_details: { cached_tokens: 0, audio_tokens: 0 },
      completion_tokens_details: {
        reasoning_tokens: 0,
        audio_tokens: 0,
        accepted_prediction_tokens: 0,
        rejected_prediction_tokens: 0,
      },
    },
    service_tier: "default",
    system_fingerprint: "fp_eb9dce56a8",
  },
];

export const openaiClassifierShouldEvaluateArithmeticExpressions = [
  {
    id: "chatcmpl-B7YSMVJ7qaQTJ9OtR6zPUEdHxrNbT",
    object: "chat.completion",
    created: 1741140446,
    model: "gpt-4o-2024-08-06",
    choices: [
      {
        index: 0,
        message: {
          role: "assistant",
          content: null,
          tool_calls: [
            {
              id: "call_Iatq5uhNc05I95JHjM7v3N5Y",
              type: "function",
              function: {
                name: "select_choice",
                arguments:
                  '{"reasons":"1. The instruction is to add the numbers 1, 2, and 3.\\n2. The correct sum of these numbers is 1 + 2 + 3 = 6.\\n3. Response 1 provides the answer as 600, which is incorrect.\\n4. Response 2 provides the answer as 6, which is correct.\\n5. Since the task is to evaluate which response is better based on the correctness of the addition, Response 2 is better because it provides the correct sum.\\n6. Therefore, Response 1 is not better than Response 2.","choice":"No"}',
              },
            },
          ],
          refusal: null,
        },
        logprobs: null,
        finish_reason: "stop",
      },
    ],
    usage: {
      prompt_tokens: 248,
      completion_tokens: 133,
      total_tokens: 381,
      prompt_tokens_details: { cached_tokens: 0, audio_tokens: 0 },
      completion_tokens_details: {
        reasoning_tokens: 0,
        audio_tokens: 0,
        accepted_prediction_tokens: 0,
        rejected_prediction_tokens: 0,
      },
    },
    service_tier: "default",
    system_fingerprint: "fp_eb9dce56a8",
  },
  {
    id: "chatcmpl-B7YTPWIPOFpRcVOjEnU6s0kZXgPdB",
    object: "chat.completion",
    created: 1741140511,
    model: "gpt-4o-2024-08-06",
    choices: [
      {
        index: 0,
        message: {
          role: "assistant",
          content: null,
          tool_calls: [
            {
              id: "call_eYJIS5zb9S0qS3NW2XZ7HtPu",
              type: "function",
              function: {
                name: "select_choice",
                arguments:
                  '{"reasons":"1. The instruction in both cases is to add the numbers 1, 2, and 3.\\n2. The correct sum of these numbers is 1 + 2 + 3 = 6.\\n3. Response 1 provides the answer as 6, which is the correct sum of the numbers.\\n4. Response 2 provides the answer as 600, which is incorrect as it does not represent the sum of the numbers given in the instruction.\\n5. Since Response 1 correctly answers the instruction and Response 2 does not, Response 1 is objectively better than Response 2.\\n6. Therefore, based on the correctness of the responses, the first response is better than the second.","choice":"Yes"}',
              },
            },
          ],
          refusal: null,
        },
        logprobs: null,
        finish_reason: "stop",
      },
    ],
    usage: {
      prompt_tokens: 248,
      completion_tokens: 157,
      total_tokens: 405,
      prompt_tokens_details: { cached_tokens: 0, audio_tokens: 0 },
      completion_tokens_details: {
        reasoning_tokens: 0,
        audio_tokens: 0,
        accepted_prediction_tokens: 0,
        rejected_prediction_tokens: 0,
      },
    },
    service_tier: "default",
    system_fingerprint: "fp_eb9dce56a8",
  },
  {
    id: "chatcmpl-B7YU2qluNL0SenvL1zBiSzrka236n",
    object: "chat.completion",
    created: 1741140550,
    model: "gpt-4o-2024-08-06",
    choices: [
      {
        index: 0,
        message: {
          role: "assistant",
          content: null,
          tool_calls: [
            {
              id: "call_kfVuMD09ytJIQVocHTEBrYLW",
              type: "function",
              function: {
                name: "select_choice",
                arguments:
                  '{"reasons":"1. Both instructions are identical, asking to add the numbers 1, 2, and 3.\\n2. Both responses provide the correct sum of these numbers, which is 6.\\n3. There is no additional context, explanation, or formatting in either response that would differentiate them in terms of quality or clarity.\\n4. Since both responses are identical and correct, there is no basis to claim that one is better than the other.\\n5. Therefore, the first response is not better than the second; they are equally good.","choice":"No"}',
              },
            },
          ],
          refusal: null,
        },
        logprobs: null,
        finish_reason: "stop",
      },
    ],
    usage: {
      prompt_tokens: 248,
      completion_tokens: 121,
      total_tokens: 369,
      prompt_tokens_details: { cached_tokens: 0, audio_tokens: 0 },
      completion_tokens_details: {
        reasoning_tokens: 0,
        audio_tokens: 0,
        accepted_prediction_tokens: 0,
        rejected_prediction_tokens: 0,
      },
    },
    service_tier: "default",
    system_fingerprint: "fp_eb9dce56a8",
  },
  {
    id: "chatcmpl-B7YUTk3771FhLlXQNZPaobEC0d8R6",
    object: "chat.completion",
    created: 1741140577,
    model: "gpt-4o-2024-08-06",
    choices: [
      {
        index: 0,
        message: {
          role: "assistant",
          content: null,
          tool_calls: [
            {
              id: "call_lbRjfwrJVP8HgLupWflqoCBM",
              type: "function",
              function: { name: "select_choice", arguments: '{"choice":"No"}' },
            },
          ],
          refusal: null,
        },
        logprobs: null,
        finish_reason: "stop",
      },
    ],
    usage: {
      prompt_tokens: 170,
      completion_tokens: 6,
      total_tokens: 176,
      prompt_tokens_details: { cached_tokens: 0, audio_tokens: 0 },
      completion_tokens_details: {
        reasoning_tokens: 0,
        audio_tokens: 0,
        accepted_prediction_tokens: 0,
        rejected_prediction_tokens: 0,
      },
    },
    service_tier: "default",
    system_fingerprint: "fp_eb9dce56a8",
  },
  {
    id: "chatcmpl-B7YUtrpit4RvQCeqfOcZme9L6pMAP",
    object: "chat.completion",
    created: 1741140603,
    model: "gpt-4o-2024-08-06",
    choices: [
      {
        index: 0,
        message: {
          role: "assistant",
          content: null,
          tool_calls: [
            {
              id: "call_d3YnOawL5qadUmE46hoKds6B",
              type: "function",
              function: {
                name: "select_choice",
                arguments: '{"choice":"Yes"}',
              },
            },
          ],
          refusal: null,
        },
        logprobs: null,
        finish_reason: "stop",
      },
    ],
    usage: {
      prompt_tokens: 170,
      completion_tokens: 6,
      total_tokens: 176,
      prompt_tokens_details: { cached_tokens: 0, audio_tokens: 0 },
      completion_tokens_details: {
        reasoning_tokens: 0,
        audio_tokens: 0,
        accepted_prediction_tokens: 0,
        rejected_prediction_tokens: 0,
      },
    },
    service_tier: "default",
    system_fingerprint: "fp_eb9dce56a8",
  },
  {
    id: "chatcmpl-B7YV8HHTm4hZU58Zp9gcjwp3MigEl",
    object: "chat.completion",
    created: 1741140618,
    model: "gpt-4o-2024-08-06",
    choices: [
      {
        index: 0,
        message: {
          role: "assistant",
          content: null,
          tool_calls: [
            {
              id: "call_l3AonPTlmEhJ95fbq4M6J0sd",
              type: "function",
              function: { name: "select_choice", arguments: '{"choice":"No"}' },
            },
          ],
          refusal: null,
        },
        logprobs: null,
        finish_reason: "stop",
      },
    ],
    usage: {
      prompt_tokens: 170,
      completion_tokens: 6,
      total_tokens: 176,
      prompt_tokens_details: { cached_tokens: 0, audio_tokens: 0 },
      completion_tokens_details: {
        reasoning_tokens: 0,
        audio_tokens: 0,
        accepted_prediction_tokens: 0,
        rejected_prediction_tokens: 0,
      },
    },
    service_tier: "default",
    system_fingerprint: "fp_eb9dce56a8",
  },
];
