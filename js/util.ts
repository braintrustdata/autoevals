export type SpanLogFn = (args: Record<string, any>) => void;

declare global {
  var __inherited_braintrust_state: any;
}

// Signature taken from sdk/js/src/logger.ts::Span::traced.
export function currentSpanTraced<R>(
  name: string,
  callback: (spanLog: SpanLogFn) => R,
  args?: any
): R {
  if (globalThis.__inherited_braintrust_state) {
    const currentSpan =
      globalThis.__inherited_braintrust_state.currentSpan.getStore();
    return currentSpan.traced(
      name,
      (span: any) => callback(span.log.bind(span)),
      args
    );
  } else {
    return callback(() => {});
  }
}
