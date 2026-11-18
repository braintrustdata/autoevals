import { type ChatCompletionContentPart } from "openai/resources";

export interface Recording {
  id: string;
  mime_type: string;
  data: string;
}

export function templateUsesRecording(template: string): boolean {
  return /\{\{\s*recording\s*\}\}/.test(template);
}

export function isRecording(value: unknown): value is Recording {
  return typeof value === "object" && value !== null && "data" in value;
}

/** Split text at each `marker`, putting the recording's file part between the pieces. */
export function expandRecording(
  text: string,
  marker: string,
  recording: Recording,
): ChatCompletionContentPart[] {
  const parts: ChatCompletionContentPart[] = [];
  text.split(marker).forEach((piece, i) => {
    if (i > 0) {
      parts.push({
        type: "file",
        file: {
          filename: "recording",
          file_data: `data:${recording.mime_type};base64,${recording.data}`,
        },
      });
    }
    if (piece) {
      parts.push({ type: "text", text: piece });
    }
  });
  return parts;
}
