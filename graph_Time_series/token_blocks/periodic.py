"""Periodic feature tokens."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..token import FeatureToken

if TYPE_CHECKING:
    from ..state import State


class PeriodPhaseOneHotToken(FeatureToken):
    """Encode positions inside the selected period.

    The token expects a previous period-selection step to write
    ``state.metadata["period"]``. It always writes compact phase-id features and
    a horizon-side one-hot encoding. A history-side one-hot can be written
    automatically when it is small enough, which keeps long histories from
    accidentally creating very large tensors.
    """

    name = "PeriodPhaseOneHot"
    reads = ("raw_history", "period")
    writes = (
        "history_period_phase_id",
        "history_period_phase_onehot",
        "future_period_phase_id",
        "future_period_phase_onehot",
    )
    description = "One-hot encode positions within the selected period."
    max_uses = 1

    def __init__(
        self,
        *,
        anchor: str = "fold",
        include_history_onehot: bool | str = "auto",
        max_history_onehot_cells: int = 20_000_000,
        max_future_onehot_cells: int = 50_000_000,
        dtype=np.float32,
    ):
        if anchor not in {"fold", "history_start"}:
            raise ValueError("anchor must be 'fold' or 'history_start'.")
        if include_history_onehot not in {True, False, "auto"}:
            raise ValueError("include_history_onehot must be True, False, or 'auto'.")
        self.anchor = anchor
        self.include_history_onehot = include_history_onehot
        self.max_history_onehot_cells = int(max_history_onehot_cells)
        self.max_future_onehot_cells = int(max_future_onehot_cells)
        self.dtype = dtype

    def check_specific_conditions(self, state: "State") -> bool:
        if not super().check_specific_conditions(state):
            return False
        if state.flags.get("period_phase_encoded", False):
            return False
        try:
            period = self._period(state)
        except (TypeError, ValueError):
            return False
        future_cells = state.n_samples * state.horizon * period
        if future_cells > self.max_future_onehot_cells:
            return False
        if self.include_history_onehot is True:
            hist = np.asarray(state.features["raw_history"])
            history_cells = state.n_samples * hist.shape[1] * period
            if history_cells > self.max_history_onehot_cells:
                return False
        return True

    def apply(self, state: "State") -> "State":
        hist = np.asarray(state.features["raw_history"])
        period = self._period(state)

        history_phase = self._phase_ids(
            hist.shape[1], period, is_future=False, history_length=hist.shape[1]
        )
        future_phase = self._phase_ids(
            state.horizon, period, is_future=True, history_length=hist.shape[1]
        )

        history_phase_ids = self._repeat_ids(history_phase, state.n_samples)
        future_phase_ids = self._repeat_ids(future_phase, state.n_samples)
        future_onehot = self._onehot(future_phase_ids, period)

        state.add_historical_feature(
            "history_period_phase_id", history_phase_ids, require_finite=True
        )
        state.register_artifact(
            "history_period_phase_id",
            store="historical_features",
            kind="period_phase_id",
            role="feature",
            source_token=self.name,
            tags=("periodic", "phase"),
        )

        history_onehot_written = False
        history_cells = state.n_samples * hist.shape[1] * period
        if self._should_write_history_onehot(history_cells):
            history_onehot = self._onehot(history_phase_ids, period)
            state.add_historical_feature(
                "history_period_phase_onehot", history_onehot, require_finite=True
            )
            state.register_artifact(
                "history_period_phase_onehot",
                store="historical_features",
                kind="period_phase_onehot",
                role="feature",
                source_token=self.name,
                tags=("periodic", "phase", "onehot"),
            )
            history_onehot_written = True

        state.add_future_feature(
            "future_period_phase_id", future_phase_ids, require_finite=True
        )
        state.register_artifact(
            "future_period_phase_id",
            store="future_features",
            kind="period_phase_id",
            role="known_future_feature",
            source_token=self.name,
            tags=("periodic", "phase"),
        )

        state.add_future_feature(
            "future_period_phase_onehot", future_onehot, require_finite=True
        )
        state.register_artifact(
            "future_period_phase_onehot",
            store="future_features",
            kind="period_phase_onehot",
            role="known_future_feature",
            source_token=self.name,
            tags=("periodic", "phase", "onehot"),
        )

        state.metadata["period_phase_onehot"] = {
            "period": period,
            "anchor": self.anchor,
            "history_phase_id_shape": tuple(history_phase_ids.shape),
            "history_onehot_written": history_onehot_written,
            "history_onehot_cells": int(history_cells),
            "future_phase_id_shape": tuple(future_phase_ids.shape),
            "future_onehot_shape": tuple(future_onehot.shape),
        }
        state.flags["period_phase_encoded"] = True

        writes = {
            "history_period_phase_id": history_phase_ids.shape,
            "future_period_phase_id": future_phase_ids.shape,
            "future_period_phase_onehot": future_onehot.shape,
        }
        if history_onehot_written:
            writes["history_period_phase_onehot"] = state.historical_features[
                "history_period_phase_onehot"
            ].shape

        self._log_execution(
            state,
            reads={"raw_history": hist.shape, "period": period},
            writes=writes,
        )
        return state

    @staticmethod
    def _period(state: "State") -> int:
        period = int(state.metadata["period"])
        if period <= 0:
            raise ValueError("period must be positive.")
        return period

    def _phase_ids(
        self,
        length: int,
        period: int,
        *,
        is_future: bool,
        history_length: int,
    ) -> np.ndarray:
        positions = np.arange(length, dtype=np.int64)
        if self.anchor == "history_start":
            if is_future:
                return ((history_length + positions) % period).astype(self._id_dtype(period))
            return (positions % period).astype(self._id_dtype(period))

        # Fold alignment matches PeriodFold: trailing complete periods define
        # the phase origin, so the first forecast step starts at phase zero.
        if is_future:
            return (positions % period).astype(self._id_dtype(period))
        offset = length % period
        return ((positions - offset) % period).astype(self._id_dtype(period))

    @staticmethod
    def _repeat_ids(ids: np.ndarray, n_samples: int) -> np.ndarray:
        return np.broadcast_to(ids[None, :], (n_samples, ids.shape[0])).copy()

    def _onehot(self, ids: np.ndarray, period: int) -> np.ndarray:
        eye = np.eye(period, dtype=self.dtype)
        return eye[ids]

    def _should_write_history_onehot(self, n_cells: int) -> bool:
        if self.include_history_onehot is True:
            if n_cells > self.max_history_onehot_cells:
                raise ValueError(
                    "history one-hot would create "
                    f"{n_cells:,} cells, above max_history_onehot_cells="
                    f"{self.max_history_onehot_cells:,}."
                )
            return True
        if self.include_history_onehot == "auto":
            return n_cells <= self.max_history_onehot_cells
        return False

    @staticmethod
    def _id_dtype(period: int):
        return np.int16 if period <= np.iinfo(np.int16).max else np.int32
