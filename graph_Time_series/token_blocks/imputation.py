"""Generic missing-value cleaning tokens."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..token import CleaningToken

if TYPE_CHECKING:
    from ..state import State


class _BaseFillToken(CleaningToken):
    """Fill non-finite history values and refresh the active history view."""

    reads = ("raw_history",)
    writes = ("clean_history", "raw_history")
    max_uses = 1
    method = "base"

    def check_specific_conditions(self, state: "State") -> bool:
        if state.flags.get("history_filled", False):
            return False
        return super().check_specific_conditions(state)

    def apply(self, state: "State") -> "State":
        raw = np.asarray(state.features["raw_history"], dtype=np.float64)
        clean = np.stack([self._fill_row(row) for row in raw], axis=0).astype(np.float32)

        state.features["raw_history"] = clean.copy()
        state.register_artifact(
            "raw_history",
            store="features",
            kind="sequence",
            role="raw_input",
            source_token=self.name,
            tags=("cleaned", self.method),
        )
        state.add_historical_feature("clean_history", clean)
        state.register_artifact(
            "clean_history",
            store="historical_features",
            kind="clean_history",
            role="clean_history",
            source_token=self.name,
            tags=("clean", self.method, "no_bundle"),
        )
        state.flags["history_filled"] = self.name
        self._log_execution(
            state,
            reads={"raw_history": raw.shape},
            writes={"raw_history": clean.shape, "clean_history": clean.shape},
        )
        return state

    def _fill_row(self, row: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class LinearFillToken(_BaseFillToken):
    """Linearly interpolate NaN/inf history values."""

    name = "LinearFill"
    method = "linear_fill"
    description = "Fill NaN/inf history values by linear interpolation."

    def _fill_row(self, row: np.ndarray) -> np.ndarray:
        return linear_fill_1d(row)


class ForwardFillToken(_BaseFillToken):
    """Forward-fill NaN/inf history values."""

    name = "ForwardFill"
    method = "forward_fill"
    description = "Fill NaN/inf history values by forward fill."

    def _fill_row(self, row: np.ndarray) -> np.ndarray:
        return forward_fill_1d(row)


def linear_fill_1d(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    good = np.isfinite(values)
    if np.all(good):
        return values.copy()
    if not np.any(good):
        return np.zeros_like(values, dtype=np.float64)
    x = np.arange(values.size)
    return np.interp(x, x[good], values[good]).astype(np.float64)


def forward_fill_1d(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    good = np.isfinite(values)
    if np.all(good):
        return values.copy()
    if not np.any(good):
        return np.zeros_like(values, dtype=np.float64)

    out = values.copy()
    first = int(np.argmax(good))
    out[:first] = values[first]
    last = values[first]
    for idx in range(first, values.size):
        if good[idx]:
            last = values[idx]
        else:
            out[idx] = last
    return out.astype(np.float64)
