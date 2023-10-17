import dataclasses
from abc import ABC, abstractmethod

from .util import SerializableDataClass, current_span


@dataclasses.dataclass
class Score(SerializableDataClass):
    name: str
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
            raise ValueError(f"score ({self.score}) must be between 0 and 1")


class Scorer(ABC):
    async def eval_async(self, output, expected=None, **kwargs):
        try:
            return await self._run_eval_async(output, expected, **kwargs)
        except Exception as e:
            return Score(name=self._name(), score=0, error=e)

    def eval(self, output, expected=None, **kwargs):
        try:
            return self._run_eval_sync(output, expected, **kwargs)
        except Exception as e:
            return Score(name=self._name(), score=0, error=e)

    def __call__(self, output, expected=None, **kwargs):
        return self.eval(output, expected, **kwargs)

    async def _run_eval_async(self, output, expected=None, **kwargs) -> Score:
        # By default we just run the sync version in a thread
        return self._run_eval_sync(output, expected, **kwargs)

    def _name(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def _run_eval_sync(self, output, expected=None, **kwargs) -> Score:
        ...


__all__ = ["Score", "Scorer"]
