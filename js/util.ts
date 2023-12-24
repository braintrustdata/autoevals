export type SpanLogFn = (args: Record<string, any>) => void;

declare global {
  var __inherited_braintrust_state: any;
}

// Signature taken from sdk/js/src/logger.ts::Span::traced.
export function currentSpanTraced<R>(
  callback: (spanLog: SpanLogFn) => R,
  args?: any
): R {
  if (globalThis.__inherited_braintrust_state) {
    const currentSpan =
      globalThis.__inherited_braintrust_state.currentSpan.getStore();
    // Old versions of the API provide the name as the first positional
    // argument. Otherwise, the name is provided as an optional keyword argument
    // at the end.
    if (currentSpan.traced.length === 2) {
      return currentSpan.traced(
        (span: any) => callback(span.log.bind(span)),
        args
      );
    } else {
      return currentSpan.traced(
        args?.name ?? "subspan",
        (span: any) => callback(span.log.bind(span)),
        args
      );
    }
  } else {
    return callback(() => {});
  }
}
