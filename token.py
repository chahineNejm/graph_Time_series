"""Token ABC — base class for all computational tokens."""

from abc import ABC, abstractmethod
import numpy as np


class Token(ABC):
    """
    A computational token that transforms a State.

    Subclasses must define:
        name        : str           unique identifier
        token_class : str           one of cleaning/feature/encoder/model/decoder/control
        reads       : list[str]     feature keys required in state.features
        writes      : list[str]     feature keys this token creates/overwrites
        description : str           human-readable summary

    And implement:
        apply(state) -> state
    """

    name: str = ""
    token_class: str = ""
    reads: list = []
    writes: list = []
    description: str = ""

    @abstractmethod
    def apply(self, state):
        """Transform the state in-place and return it."""
        ...

    def can_apply(self, state) -> bool:
        """Check that all required features are present."""
        return all(k in state.features for k in self.reads)

    def __repr__(self):
        return f"Token({self.name}, class={self.token_class})"


def _shapes(state, keys: list[str]) -> dict:
    """Helper: return {key: shape} for logging."""
    return {k: state.features[k].shape for k in keys if k in state.features}
