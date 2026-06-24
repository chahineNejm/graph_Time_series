"""Trivial baseline model tokens -- self-contained.

* ``LastValueToken`` ("last_value") -- persistence: repeat the last observed
  value across the whole horizon.
* ``ZeroToken``      ("zero")       -- predict 0 over the horizon. On a centered
  scale (e.g. after ZNormalization) that is the mean forecast; in a residual
  stack it contributes nothing (a no-op baseline).

Both are ``ModelToken``s: they push an (n, H) prediction onto the residual stack
on the active modeling scale, like every other model. Register with
``register_baseline_tokens(grammar)``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ..token import ModelToken

if TYPE_CHECKING:
    from ..state import State


def _resolve_history(state: "State", source_feature: str | None = None) -> np.ndarray:
    """Active modeling-scale history (n_samples, T): clean/scaled/raw fallback."""
    hf = state.historical_features
    names = ([source_feature] if source_feature else []) + ["clean_history", "scaled_history"]
    for name in names:
        if name and name in hf:
            h = np.asarray(hf[name], dtype=np.float64)
            if h.ndim == 2 and h.shape[0] == state.n_samples:
                return h
    if "raw_history" in state.features:
        h = np.asarray(state.features["raw_history"], dtype=np.float64)
        if h.ndim == 2 and h.shape[0] == state.n_samples:
            return h
    return np.asarray(state.original_history, dtype=np.float64)


class LastValueToken(ModelToken):
    """Persistence: forecast = last observed value, repeated over the horizon."""

    learning_scope = "within_series"
    name = "last_value"
    reads = ()
    writes = ("prediction_stack",)
    max_uses = 1
    description = "Persistence baseline: repeat the last observed value across the horizon."

    def __init__(self, source_feature: str | None = None):
        self.source_feature = source_feature

    def get_model(self) -> Any:
        return {"model": "last_value", "kind": "persistence"}

    def check_specific_conditions(self, state: "State") -> bool:
        if not super().check_specific_conditions(state):
            return False
        try:
            hist = _resolve_history(state, self.source_feature)
        except (KeyError, ValueError):
            return False
        return hist.shape[1] >= 1

    def apply(self, state: "State") -> "State":
        hist = _resolve_history(state, self.source_feature)
        H = int(state.horizon)
        pred = np.repeat(hist[:, -1:], H, axis=1).astype(np.float32)
        state.push_prediction(pred, self.name)
        self._log_execution(state, reads={"history": hist.shape},
                            writes={"prediction_stack[-1]": pred.shape})
        return state


class ZeroToken(ModelToken):
    """Zero forecast: predict 0 over the horizon (mean forecast on centered scales)."""

    learning_scope = "within_series"
    name = "zero"
    reads = ()
    writes = ("prediction_stack",)
    max_uses = 1
    description = "Zero baseline: predicts 0 over the horizon (mean on centered scales / no-op residual)."

    def get_model(self) -> Any:
        return {"model": "zero"}

    def apply(self, state: "State") -> "State":
        H = int(state.horizon)
        pred = np.zeros((state.n_samples, H), dtype=np.float32)
        state.push_prediction(pred, self.name)
        self._log_execution(state, reads={}, writes={"prediction_stack[-1]": pred.shape})
        return state


def register_baseline_tokens(grammar):
    """Register the two baselines tied to every filling and normalization token."""
    fillers = ["LinearFill", "ForwardFill", "FlairPreprocess"]   # all filling procedures
    scalers = ["ZNormalization", "MeanAbsScaling"]               # all normalization procedures
    sources = fillers + scalers
    grammar.register(LastValueToken(), follows=sources, leads_to=["STOP"])
    grammar.register(ZeroToken(), follows=sources, leads_to=["STOP"])
    return grammar


__all__ = ["LastValueToken", "ZeroToken", "register_baseline_tokens"]
