from .score import Scorer

_UNSET = object()


class ScorerWithPartial(Scorer):
    @classmethod
    def partial(cls, **partial_kwargs):
        class PartialScorer(cls):
            async def eval_async(self, output, expected=_UNSET, **kwargs):
                if expected is not _UNSET:
                    kwargs["expected"] = expected
                return await self._run_eval_async(output, **{**partial_kwargs, **kwargs})

            def eval(self, output, expected=_UNSET, **kwargs):
                if expected is not _UNSET:
                    kwargs["expected"] = expected
                return self._run_eval_sync(output, **{**partial_kwargs, **kwargs})

            def __call__(self, output, expected=_UNSET, **kwargs):
                return self.eval(output, expected, **kwargs)

            @classmethod
            def _partial_args(cls):
                return {**partial_kwargs}

        PartialScorer.__name__ = cls.__name__
        return PartialScorer
