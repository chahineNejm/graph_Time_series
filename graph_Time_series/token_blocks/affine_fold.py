"""Affine seasonal decomposition -- self-contained two-token extension.

Mirrors ``SeasonalFold`` + the FLAIR level forecaster, but with the AFFINE fold

    value[phase, cycle] = shift[cycle] + scale[cycle] * shape[phase]

instead of the multiplicative  ``level[cycle] * shape[phase]``.

* ``AffineSeasonalFoldToken`` ("affine_fold") -- fold the history into
  ``shift`` (per-cycle baseline / mean), ``scale`` (per-cycle amplitude / std)
  and a ``shape`` that is mean-0 / var-1 over the period.
* ``AffineLevelForecastToken`` ("affine_forecast") -- forecast the ``shift`` and
  ``scale`` series independently and rebuild
  ``y_hat[t] = shift_hat[t // P] + scale_hat[t // P] * shape[t % P]``.

Affine vs multiplicative: decouples baseline from amplitude, handles signed
data, needs no positivity shift. Self-contained -- only depends on numpy and the
token base classes; register with ``register_affine_tokens(grammar)``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ..token import FeatureToken, ModelToken
from ..signal import Port

if TYPE_CHECKING:
    from ..state import State

_EPS = 1e-8


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

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


def _ridge_ar_forecast(y: np.ndarray, m: int, alpha: float = 1.0) -> np.ndarray:
    """Forecast a 1-D series ``m`` steps ahead via a tiny ridge AR.

    Design row at cycle t: ``[1, t_scaled, y[t-1]]`` predicting ``y[t]``
    (intercept + trend + lag-1), solved once and rolled forward. Simple,
    self-contained, and stable for the short shift/scale series.
    """
    y = np.asarray(y, dtype=np.float64)
    n = y.size
    if n < 3 or not np.all(np.isfinite(y)):
        return np.full(m, float(y[-1]) if n else 0.0)
    t = np.arange(1, n, dtype=np.float64)
    X = np.stack([np.ones(n - 1), t / max(n, 1), y[:-1]], axis=1)
    target = y[1:]
    beta = np.linalg.solve(X.T @ X + alpha * np.eye(3), X.T @ target)
    out = np.empty(m, dtype=np.float64)
    prev = float(y[-1])
    for j in range(m):
        tt = (n + j) / max(n, 1)
        nxt = float(beta[0] + beta[1] * tt + beta[2] * prev)
        out[j] = nxt
        prev = nxt
    return out


# ---------------------------------------------------------------------------
# Token 1: the affine fold (learns shift + scale + shape)
# ---------------------------------------------------------------------------

class AffineSeasonalFoldToken(FeatureToken):
    """Affine fold: value = shift[cycle] + scale[cycle] * shape[phase].

    Folds the active history by the primary period ``P = periods[0]`` and writes
    three artifacts:

    * ``affine_shift``  (n_samples, n_cycles) -- per-cycle baseline (mean),
    * ``affine_scale``  (n_samples, n_cycles) -- per-cycle amplitude (std),
    * ``affine_shape``  (n_samples, P)        -- mean-0 / var-1 within-period
      profile (the normalized curve).

    The amplitude (scale) and baseline (shift) are decoupled, so a series whose
    level drifts independently of its swing is captured exactly, and signed data
    needs no positivity shift. Single primary fold (no Shape2 here -- keep it
    minimal; the forecaster handles shift and scale).
    """

    name = "affine_fold"
    token_class = "feature"
    reads = ("raw_history",)
    writes = ("affine_shift", "affine_scale", "affine_shape")
    description = (
        "Affine seasonal fold: per-cycle shift (baseline) + scale (amplitude) + "
        "a mean-0/var-1 shape, i.e. shift + scale*shape (additive, signed-safe)."
    )
    max_uses = 1
    requires = (Port(sem="param:periods", alignment="static", space="any"),)

    def __init__(self, shape_k: int = 2, min_complete: int = 2,
                 source_feature: str | None = None):
        self.shape_k = int(shape_k)
        self.min_complete = int(min_complete)
        self.source_feature = source_feature

    def _periods(self, state: "State") -> list:
        periods = state.metadata.get("periods")
        if not periods:
            p = int(state.metadata.get("period", 0))
            periods = [p] if p > 0 else []
        return [int(p) for p in periods]

    def check_specific_conditions(self, state: "State") -> bool:
        if not super().check_specific_conditions(state):
            return False
        if "affine_shape" in state.historical_features:
            return False
        periods = self._periods(state)
        if not periods or periods[0] <= 0:
            return False
        try:
            hist = _resolve_history(state, self.source_feature)
        except (KeyError, ValueError):
            return False
        return hist.shape[1] // periods[0] >= self.min_complete

    def apply(self, state: "State") -> "State":
        hist = _resolve_history(state, self.source_feature)
        n_s, T = hist.shape
        P = self._periods(state)[0]
        n = T // P
        usable = n * P
        M = hist[:, -usable:].reshape(n_s, n, P).transpose(0, 2, 1)   # (n_s, P, n)

        shift = M.mean(axis=1)                                         # (n_s, n)  baseline (b)
        centered = M - shift[:, None, :]
        scale = np.maximum(centered.std(axis=1), _EPS)                 # (n_s, n)  amplitude (a)

        K = min(max(self.shape_k, 1), n)
        norm = centered / scale[:, None, :]                            # (n_s, P, n)
        shape = norm[:, :, -K:].mean(axis=2)                           # (n_s, P)
        shape = shape - shape.mean(axis=1, keepdims=True)
        shape = shape / np.maximum(shape.std(axis=1, keepdims=True), _EPS)

        shift32 = shift.astype(np.float32)
        scale32 = scale.astype(np.float32)
        shape32 = shape.astype(np.float32)
        state.add_historical_feature("affine_shift", shift32)
        state.add_historical_feature("affine_scale", scale32)
        state.add_historical_feature("affine_shape", shape32)
        state.metadata["period"] = int(P)
        state.metadata["n_complete_periods"] = int(n)
        state.metadata["affine_shape_k"] = int(K)
        for name in ("affine_shift", "affine_scale", "affine_shape"):
            state.register_artifact(
                name, store="historical_features", kind="affine_component",
                source_token=self.name, tags=("affine", "seasonal"),
            )
        self._log_execution(
            state,
            reads={"history": hist.shape, "period": P},
            writes={"affine_shift": shift32.shape, "affine_scale": scale32.shape,
                    "affine_shape": shape32.shape},
        )
        return state


# ---------------------------------------------------------------------------
# Token 2: the forecaster (predicts shift + scale, rebuilds the series)
# ---------------------------------------------------------------------------

class AffineLevelForecastToken(ModelToken):
    """Forecast the affine shift & scale series, rebuild shift + scale*shape.

    For each series, forecast the per-cycle ``shift`` (baseline) and ``scale``
    (amplitude) ``m = ceil(H / P)`` cycles ahead with a small ridge AR, then
    reconstruct the horizon:

        y_hat[t] = shift_hat[t // P] + scale_hat[t // P] * shape[t % P].

    The mean-0/var-1 ``shape`` is frozen (re-draped, not forecast). ``scale`` is
    floored at >= 0 (an amplitude). Pushes a point prediction onto the residual
    stack like every other model.
    """

    learning_scope = "within_series"
    name = "affine_forecast"
    reads = ("affine_shift", "affine_scale", "affine_shape", "period")
    writes = ("prediction_stack",)
    description = (
        "Forecast affine shift & scale (ridge AR) and rebuild "
        "shift + scale*shape over the horizon."
    )

    def __init__(self, alpha: float = 1.0):
        self.alpha = float(alpha)

    def get_model(self) -> Any:
        return {"model": "affine_forecast", "shift": "ridge_ar",
                "scale": "ridge_ar", "alpha": self.alpha}

    def check_specific_conditions(self, state: "State") -> bool:
        if not super().check_specific_conditions(state):
            return False
        hf = state.historical_features
        return (
            "affine_shift" in hf and "affine_scale" in hf and "affine_shape" in hf
            and int(state.metadata.get("period", 0)) > 0
        )

    def apply(self, state: "State") -> "State":
        shift = np.asarray(state.historical_features["affine_shift"], dtype=np.float64)
        scale = np.asarray(state.historical_features["affine_scale"], dtype=np.float64)
        shape = np.asarray(state.historical_features["affine_shape"], dtype=np.float64)
        P = int(state.metadata["period"])
        H = int(state.horizon)
        n_s = shift.shape[0]
        m = int(np.ceil(H / max(P, 1)))
        steps = np.arange(H) // max(P, 1)
        phases = np.arange(H) % max(P, 1)

        pred = np.zeros((n_s, H), dtype=np.float32)
        for i in range(n_s):
            shift_hat = _ridge_ar_forecast(shift[i], m, self.alpha)
            scale_hat = np.maximum(_ridge_ar_forecast(scale[i], m, self.alpha), 0.0)
            pred[i] = (shift_hat[steps] + scale_hat[steps] * shape[i, phases]).astype(np.float32)

        state.push_prediction(pred, self.name)
        self._log_execution(
            state,
            reads={"affine_shift": shift.shape, "affine_scale": scale.shape,
                   "affine_shape": shape.shape, "period": P},
            writes={"prediction_stack[-1]": pred.shape},
        )
        return state


# ---------------------------------------------------------------------------
# Registration helper (call after the base/FLAIR grammar is built)
# ---------------------------------------------------------------------------

def register_affine_tokens(grammar):
    """Register the affine fold + forecaster.

    ``affine_fold`` follows any period detector (it needs ``param:periods``);
    ``affine_forecast`` follows the fold and ends the pipeline.
    """
    detectors = ["PeriodDetect", "PeriodDetectBIC", "PeriodDetectSpectral"]
    grammar.register(
        AffineSeasonalFoldToken(),
        follows=detectors,
        leads_to=["affine_forecast"],
    )
    grammar.register(
        AffineLevelForecastToken(),
        follows=["affine_fold"],
        leads_to=["STOP"],
    )
    return grammar


__all__ = [
    "AffineSeasonalFoldToken",
    "AffineLevelForecastToken",
    "register_affine_tokens",
]
