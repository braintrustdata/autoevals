/* This is copy/pasted from braintrust-sdk*/
export class NoopSpan {
  public id: string;
  public span_id: string;
  public root_span_id: string;

  constructor() {
    this.id = "";
    this.span_id = "";
    this.root_span_id = "";
  }

  public log(_: any) {}

  public startSpan(_: any) {
    return this;
  }

  public startSpanWithCallback<R>(_: any, callback: (span: any) => R): R {
    return callback(this);
  }

  public end(args?: { endTime?: number }): number {
    return args?.endTime ?? new Date().getTime() / 1000;
  }

  public close(args?: { endTime?: number }): number {
    return this.end(args);
  }
}
declare global {
  var __inherited_braintrust_state: any;
}
export function currentSpan() {
  if (__inherited_braintrust_state) {
    return __inherited_braintrust_state.currentSpan.getStore();
  } else {
    return new NoopSpan();
  }
}
