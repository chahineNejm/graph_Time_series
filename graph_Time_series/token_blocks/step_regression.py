"""StepRegression: a standalone per-date regression model.

Rows are time steps (dates), not series. For each date it builds X from every
KNOWN-PER-STEP signal currently on the board -- any feature that exists both
over the history AND over the forecast horizon (seasonality, calendar, etc.) --
plus an optional normalized time index. The target y is the series value at that
date. It fits on the history rows and predicts the horizon from the known-future
covariates at each future date.

This makes it model-agnostic about WHERE its inputs come from: anything that
publishes a matching (history, future) per-step pair is automatically used. With
no covariates it degenerates to a pure index/trend regression, so it is always
runnable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ..token import ModelToken
from ..signal import sem_root

if TYPE_CHECKING:
    from ..state import State


def _as_3d(arr: np.ndarray, n: int) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 2:                       # (n, len) -> single channel
        arr = arr[:, :, None]
    return arr


def collect_step_covariates(state: "State"):
    """Find (history, future) per-step covariate pairs on the board.

    A pair is two array signals from the same source token, one history-aligned
    over ``time`` and one future-aligned over ``horizon``, with the same number
    of channels. Targets/predictions (series/forecast/samples/param) are
    excluded. Returns a list of dicts with aligned (n, T, C) / (n, H, C) arrays.
    """
    n = state.n_samples
    futures = {}
    histories = {}
    for sig in state.board:
        if not sig.is_array or sem_root(sig.sem) in {"forecast", "samples", "param", "mask"}:
            continue
        if sig.alignment == "future" and "horizon" in sig.axes:
            futures.setdefault(sig.source or sig.name, []).append(sig)
        elif sig.alignment == "history" and "time" in sig.axes and sem_root(sig.sem) == "features":
            histories.setdefault(sig.source or sig.name, []).append(sig)

    pairs = []
    for source, fut_sigs in futures.items():
        hist_sigs = histories.get(source, [])
        for fs in fut_sigs:
            fa = _as_3d(fs.value, n)
            match = next((hs for hs in hist_sigs
                          if _as_3d(hs.value, n).shape[-1] == fa.shape[-1]), None)
            if match is None:
                continue
            pairs.append({
                "name": match.name,
                "hist": _as_3d(match.value, n),     # (n, T, C)
                "fut": fa,                          # (n, H, C)
                "channels": int(fa.shape[-1]),
            })
    return pairs


class StepRegressionToken(ModelToken):
    """Per-date regression over known-future covariates (+ optional index)."""

    learning_scope = "configurable"  # per_series=True within-series, False cross-series

    name = "step_regression"
    token_class = "model"
    reads = ()
    writes = ("prediction_stack",)
    description = "Per-date regression: value <- known-future covariates + index."

    def __init__(self, *, estimator: str = "rf", per_series: bool = True,
                 include_index: bool = True, n_estimators: int = 60,
                 max_depth: int | None = 12, learning_rate: float = 0.08,
                 max_iter: int = 150, seed: int = 0,
                 target_feature: str = "scaled_history",
                 show_progress: bool = False, progress_min_samples: int = 16):
        self.estimator = estimator
        self.per_series = bool(per_series)
        self.include_index = bool(include_index)
        self.n_estimators = int(n_estimators)
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.max_iter = int(max_iter)
        self.seed = seed
        self.target_feature = target_feature
        self.show_progress = show_progress
        self.progress_min_samples = progress_min_samples

    def get_model(self) -> dict[str, Any]:
        return {"model": self.estimator, "target": "value_per_date",
                "per_series": self.per_series, "include_index": self.include_index}

    # ------------------------------------------------------------------

    def _target(self, state: "State") -> np.ndarray:
        for name in (self.target_feature, "scaled_history"):
            if name in state.historical_features:
                return np.asarray(state.historical_features[name], dtype=np.float32)
        return np.asarray(state.features["raw_history"], dtype=np.float32)

    def _build_estimator(self):
        if self.estimator == "gb":
            from sklearn.ensemble import HistGradientBoostingRegressor
            return HistGradientBoostingRegressor(
                max_iter=self.max_iter, learning_rate=self.learning_rate,
                max_depth=self.max_depth, random_state=self.seed)
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(
            n_estimators=self.n_estimators, max_depth=self.max_depth,
            random_state=self.seed, n_jobs=-1)

    def check_specific_conditions(self, state: "State") -> bool:
        if not super().check_specific_conditions(state):
            return False
        # Always runnable as long as there is at least one input column
        # (the index, or any known-future covariate).
        if self.include_index:
            return True
        return len(collect_step_covariates(state)) > 0

    def apply(self, state: "State") -> "State":
        y_hist = self._target(state)
        n, T = y_hist.shape
        H = state.horizon

        pairs = collect_step_covariates(state)
        hist_blocks = [p["hist"] for p in pairs]
        fut_blocks = [p["fut"] for p in pairs]

        if self.include_index:
            ih = np.broadcast_to((np.arange(T) / float(T))[None, :, None],
                                 (n, T, 1)).astype(np.float32)
            iff = np.broadcast_to(((np.arange(T, T + H)) / float(T))[None, :, None],
                                  (n, H, 1)).astype(np.float32)
            hist_blocks.append(ih)
            fut_blocks.append(iff)

        if not hist_blocks:
            raise ValueError("step_regression has no inputs (no covariates, no index).")

        Xh = np.concatenate(hist_blocks, axis=2)     # (n, T, F)
        Xf = np.concatenate(fut_blocks, axis=2)      # (n, H, F)
        pred = np.zeros((n, H), dtype=np.float32)

        if self.per_series:
            iterator = range(n)
            if self.show_progress and n >= self.progress_min_samples:
                try:
                    from tqdm.auto import tqdm
                    iterator = tqdm(iterator, desc="step_regression (per series)")
                except Exception:
                    pass
            for i in iterator:
                est = self._build_estimator()
                est.fit(Xh[i], y_hist[i])
                pred[i] = est.predict(Xf[i]).astype(np.float32)
        else:
            est = self._build_estimator()
            est.fit(Xh.reshape(n * T, -1), y_hist.reshape(n * T))
            pred = est.predict(Xf.reshape(n * H, -1)).reshape(n, H).astype(np.float32)

        state.push_prediction(pred, self.name)
        state.metadata[self.name] = {
            **self.get_model(),
            "covariates": [{"name": p["name"], "channels": p["channels"]} for p in pairs],
            "n_features": int(Xh.shape[2]),
            "rows_per_fit": int(T),
        }
        self._log_execution(
            state,
            reads={"covariates": [p["name"] for p in pairs],
                   "target": y_hist.shape},
            writes={"prediction_stack[-1]": pred.shape},
        )
        return state
