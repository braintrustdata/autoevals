import dataclasses
from abc import ABC, abstractmethod

from .util import SerializableDataClass


@dataclasses.dataclass
class Evaluation(SerializableDataClass):
    score: float
    metadata: dict[str, any] = dataclasses.field(default_factory=dict)
    error: Exception = None

    def as_dict(self):
        return {
            "score": self.score,
            "metadata": self.metadata,
            "error": repr(self.error) if self.error else None,
        }

    def __post_init__(self):
        if self.score < 0 or self.score > 1:
            raise ValueError("score must be between 0 and 1")


class Evaluator(ABC):
    async def eval_async(self, output, expected=None, **kwargs):
        try:
            return await self._run_eval_async(output, expected, **kwargs)
        except Exception as e:
            return Evaluation(0, error=e)

    def eval(self, output, expected=None, **kwargs):
        try:
            return self._run_eval_sync(output, expected, **kwargs)
        except Exception as e:
            return Evaluation(0, error=e)

    def __call__(self, output, expected=None, **kwargs):
        return self.eval(output, expected, **kwargs)

    async def _run_eval_async(self, output, expected=None, **kwargs) -> Evaluation:
        # By default we just run the sync version in a thread
        return self._run_eval_sync(output, expected, **kwargs)

    @abstractmethod
    def _run_eval_sync(self, output, expected=None, **kwargs) -> Evaluation:
        ...


__all__ = ["Evaluation", "Evaluator"]
