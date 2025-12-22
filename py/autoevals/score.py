import dataclasses
import sys
from abc import ABC, abstractmethod
from typing import Any

from .serializable_data_class import SerializableDataClass


@dataclasses.dataclass
class Score(SerializableDataClass):
    """A score for an evaluation. The score is a float between 0 and 1."""

    name: str
    """The name of the score. This should be a unique name for the scorer."""

    score: float | None = None
    """The score for the evaluation. This should be a float between 0 and 1. If the score is None, the evaluation is considered to be skipped."""

    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)
    """Metadata for the score. This can be used to store additional information about the score."""

    # DEPRECATION_NOTICE: this field is deprecated, as errors are propagated up to the caller.
    error: Exception | None = None
    """Deprecated: The error field is deprecated, as errors are now propagated to the caller. The field will be removed in a future version of the library."""

    def as_dict(self):
        return {
            "score": self.score,
            "metadata": self.metadata,
        }

    def __post_init__(self):
        if self.score is not None and (self.score < 0 or self.score > 1):
            raise ValueError(f"score ({self.score}) must be between 0 and 1")
        if self.error is not None:
            print(
                "The error field is deprecated, as errors are now propagated to the caller. The field will be removed in a future version of the library",
                sys.stderr,
            )


class Scorer(ABC):
    async def eval_async(self, output: Any, expected: Any = None, **kwargs: Any) -> Score:
        return await self._run_eval_async(output, expected, **kwargs)

    def eval(self, output: Any, expected: Any = None, **kwargs: Any) -> Score:
        return self._run_eval_sync(output, expected, **kwargs)

    def __call__(self, output: Any, expected: Any = None, **kwargs: Any) -> Score:
        return self.eval(output, expected, **kwargs)

    async def _run_eval_async(self, output: Any, expected: Any = None, **kwargs: Any) -> Score:
        # By default we just run the sync version in a thread
        return self._run_eval_sync(output, expected, **kwargs)

    def _name(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def _run_eval_sync(self, output: Any, expected: Any = None, **kwargs: Any) -> Score:
        ...


__all__ = ["Score", "Scorer"]
