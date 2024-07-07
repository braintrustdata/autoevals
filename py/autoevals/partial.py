from braintrust_core.score import Scorer


class ScorerWithPartial(Scorer):
    @classmethod
    def partial(cls, **partial_kwargs):
        class PartialScorer(cls):
            async def eval_async(self, output, expected=None, **kwargs):
                if expected is not None:
                    kwargs["expected"] = expected
                return await self._run_eval_async(output, **{**partial_kwargs, **kwargs})

            def eval(self, output, expected=None, **kwargs):
                if expected is not None:
                    kwargs["expected"] = expected
                return self._run_eval_sync(output, **{**partial_kwargs, **kwargs})

            @classmethod
            def _partial_args(cls):
                return {**partial_kwargs}

        PartialScorer.__name__ = cls.__name__
        return PartialScorer
