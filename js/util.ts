/* This is copy/pasted from braintrust-sdk*/
export class NoopSpan {
  public id: string;
  public span_id: string;
  public root_span_id: string;
  public kind: "span" = "span";

  constructor() {
    this.id = "";
    this.span_id = "";
    this.root_span_id = "";
  }

  public log(_: any) {}

  public startSpan(_0: string, _1?: any) {
    return this;
  }

  public traced<R>(_0: string, callback: (span: any) => R, _1: any): R {
    return callback(this);
  }

  public end(args?: any): number {
    return args?.endTime ?? new Date().getTime() / 1000;
  }

  public close(args?: any): number {
    return this.end(args);
  }
}
declare global {
  var __inherited_braintrust_state: any;
}
export function currentSpan() {
  if (globalThis.__inherited_braintrust_state) {
    return globalThis.__inherited_braintrust_state.currentSpan.getStore();
  } else {
    return new NoopSpan();
  }
}
