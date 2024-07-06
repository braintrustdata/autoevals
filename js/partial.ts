import { Scorer, ScorerArgs } from "@braintrust/core";

export interface ScorerWithPartial<Output, Extra>
  extends Scorer<Output, Extra> {
  partial: (args: Partial<ScorerArgs<Output, Extra>>) => Scorer<Output, Extra>;
}

export function makePartial<Output, Extra>(
  fn: Scorer<Output, Extra>,
  name?: string
): ScorerWithPartial<Output, Extra> {
  const fnCopy = fn.bind({});
  const ret = {
    ...fnCopy,
    partial: (args: Partial<ScorerArgs<Output, Extra>>) => {
      const newFn = (newArgs: ScorerArgs<Output, Extra>) =>
        fnCopy({ ...args, ...newArgs });
      if (name) {
        Object.defineProperty(newFn, "name", {
          value: name,
          configurable: true,
        });
      }
      return newFn;
    },
  } as ScorerWithPartial<Output, Extra>;
  if (name) {
    Object.defineProperty(ret, "name", {
      value: name,
      configurable: true,
    });
  }
  return ret;
}
