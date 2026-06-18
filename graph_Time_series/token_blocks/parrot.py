"""Parrot (analog / nearest-neighbour) forecaster.

The simplest analog model: for each series, find the past window most strongly
correlated (in absolute value) with the most recent window, and copy forward
whatever followed it -- affine-rescaled to the current level.

It is the rank-1 corner of the constructed-analog family
``argmin_M ||query - X M||``: a single window, a scalar (affine) ``M``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ..token import ModelToken

if TYPE_CHECKING:
    from ..state import State


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    d = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / d) if d > 1e-12 else 0.0


class ParrotToken(ModelToken):
    """Copy the continuation of the best |correlation| past analog window.

    Per series (``within_series``): the query is the last ``H`` points (``H`` =
    horizon). Slide over the series' own past, score each ``H``-window by
    ``|Pearson|`` against the query, take the single best, and forecast the
    ``H`` points that followed it. Because correlation ignores level/scale, the
    copied continuation is affine-rescaled via the fit ``query ~= a*window + b``
    on the matched window (``a`` may be negative -- an anti-correlated analog is
    flipped). Predicts on the active modeling scale, like every other model.
    """

    learning_scope = "within_series"
    name = "parrot"
    reads = ()
    writes = ("prediction_stack",)
    max_uses = 1
    description = (
        "Analog forecast: copy the continuation of the past window with the "
        "highest |correlation| to the recent window, affine-rescaled."
    )

    def __init__(self, source_feature: str | None = None):
        self.source_feature = source_feature

    def get_model(self) -> Any:
        return {
            "model": "parrot",
            "match": "abs_pearson",
            "window": "horizon",
            "neighbours": 1,
            "source_feature": self.source_feature or "auto",
        }

    def check_specific_conditions(self, state: "State") -> bool:
        if not super().check_specific_conditions(state):
            return False
        try:
            hist, _ = self._resolve_history(state)
        except (KeyError, ValueError):
            return False
        return hist.shape[1] > 0 and np.all(np.isfinite(hist))

    def _resolve_history(self, state: "State") -> tuple[np.ndarray, str]:
        hf = state.historical_features
        candidates: list[tuple[str, np.ndarray]] = []
        if self.source_feature:
            for store_name, store in (
                ("historical_features", state.historical_features),
                ("features", state.features),
            ):
                if self.source_feature in store:
                    candidates.append(
                        (f"{store_name}.{self.source_feature}", store[self.source_feature])
                    )
                    break

        for name in ("scaled_history", "clean_history"):
            if name in hf:
                candidates.append((f"historical_features.{name}", hf[name]))
        if "raw_history" in state.features:
            candidates.append(("features.raw_history", state.features["raw_history"]))

        for source_name, values in candidates:
            hist = np.asarray(values, dtype=np.float64)
            if hist.ndim != 2:
                continue
            if hist.shape[0] != state.n_samples:
                continue
            return hist, source_name
        raise KeyError(
            "parrot requires scaled_history, clean_history, or active raw_history."
        )

    def _forecast_one(self, y: np.ndarray, H: int) -> np.ndarray:
        T = y.shape[0]
        if T < 2 * H or H < 1:
            # too short to hold an analog: persist the most recent values
            if T == 0:
                return np.zeros(H, dtype=np.float64)
            reps = int(np.ceil(H / T))
            return np.tile(y, reps)[-H:]
        query = y[-H:]
        best_abs, best_i = -1.0, 0
        for i in range(0, T - 2 * H + 1):
            c = _pearson(query, y[i:i + H])
            if abs(c) > best_abs:
                best_abs, best_i = abs(c), i
        cand = y[best_i:best_i + H]
        cont = y[best_i + H:best_i + 2 * H]
        var = float(((cand - cand.mean()) ** 2).sum())
        a = (float(((cand - cand.mean()) * (query - query.mean())).sum() / var)
             if var > 1e-12 else 1.0)
        b = float(query.mean() - a * cand.mean())
        return a * cont + b

    def apply(self, state: "State") -> "State":
        hist, input_name = self._resolve_history(state)
        H = int(state.horizon)
        n = hist.shape[0]
        pred = np.zeros((n, H), dtype=np.float32)
        for i in range(n):
            pred[i] = self._forecast_one(hist[i], H).astype(np.float32)

        state.push_prediction(pred, self.name)
        self._log_execution(
            state,
            reads={
                input_name: hist.shape,
                "current_target": state.current_target.shape,
                "n_previous_models": state.n_models_applied - 1,
            },
            writes={
                "prediction_stack[-1]": pred.shape,
                "current_residual": state.current_target.shape,
            },
        )
        return state
