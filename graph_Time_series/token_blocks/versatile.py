"""Versatile, sequence-agnostic tokens built on the Signal board.

These tokens demonstrate the combinatorial architecture in action:

* ``VersatileTabularToken`` fits *any* tabular regressor on whatever feature
  signals exist in the active scale, with no binder token required. The same
  token works after normalization, after Fourier features, after a FLAIR fold,
  or after a calendar token, because it consumes the auto-built feature bundle
  instead of one hard-coded ``model_input`` array.

* ``DayOfWeekFeatureToken`` shows adding exogenous, known-future information
  (a calendar covariate) at any point in a sequence; downstream models pick it
  up automatically through the bundle.

* ``GBLevelForecastToken`` shows swapping a FLAIR sub-step for a different
  learner: it predicts the *scale of each period* with gradient boosting
  instead of FLAIR's soft-averaged ridge, then expands through the FLAIR shape
  to a full forecast. Any model with a ``level`` port could take its place.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from ..token import FeatureToken, ModelToken
from ..signal import Port

if TYPE_CHECKING:
    from ..state import State


# ---------------------------------------------------------------------------
# Versatile tabular model: any features -> any regressor
# ---------------------------------------------------------------------------


class VersatileTabularToken(ModelToken):
    """Fit a tabular regressor on the auto-built feature bundle.

    Subclasses only supply ``_make_regressor``; the data wiring is identical
    regardless of what produced the upstream features.
    """

    token_class = "model"
    writes = ("prediction_stack",)
    # One declarative requirement: some history-aligned features in the active
    # scale, satisfiable directly or via adapter coercion (series, level,
    # shape, period matrices all flatten into features).
    requires = (
        Port(
            sem="features",
            axes=("sample", "feature"),
            alignment="history",
            space="current",
            multiple=True,
            coerce=True,
        ),
    )

    select_tags: tuple[str, ...] = ()

    def __init__(self, *, select_tags: tuple[str, ...] = ()):
        self.select_tags = tuple(select_tags) or tuple(type(self).select_tags)

    def get_model(self) -> dict[str, Any]:
        return self.model_params()

    def model_params(self) -> dict[str, Any]:
        raise NotImplementedError

    def _make_regressor(self):
        raise NotImplementedError

    def check_specific_conditions(self, state: "State") -> bool:
        if not super().check_specific_conditions(state):
            return False
        try:
            bundle = state.feature_bundle(select_tags=frozenset(self.select_tags))
        except Exception:
            return False
        return bundle.matrix.shape[1] > 0

    def apply(self, state: "State") -> "State":
        from sklearn.multioutput import MultiOutputRegressor

        bundle = state.feature_bundle(select_tags=frozenset(self.select_tags))
        X = np.asarray(bundle.matrix, dtype=np.float32)
        Y = np.asarray(state.current_target, dtype=np.float32)

        base = self._make_regressor()
        model = base if _is_multioutput(base) else MultiOutputRegressor(base)
        model.fit(X, Y)
        pred = np.asarray(model.predict(X), dtype=np.float32)
        if pred.ndim == 1:
            pred = pred[:, None]

        state.push_prediction(pred, self.name)
        state.metadata[self.name] = {
            **self.get_model(),
            "input_shape": tuple(X.shape),
            "n_feature_blocks": len(bundle.blocks),
            "feature_blocks": bundle.blocks,
            "bundle_space": bundle.space.id,
            "mode": "auto_bundle_fit",
        }
        self._log_execution(
            state,
            reads={"feature_bundle": X.shape, "current_target": Y.shape},
            writes={"prediction_stack[-1]": pred.shape},
        )
        return state


def _is_multioutput(estimator) -> bool:
    # Estimators that natively support multi-output regression.
    name = type(estimator).__name__
    return name in {
        "RandomForestRegressor",
        "ExtraTreesRegressor",
        "KNeighborsRegressor",
        "LinearRegression",
        "Ridge",
        "MultiOutputRegressor",
    }


class VersatileRandomForestToken(VersatileTabularToken):
    name = "versatile_rf"
    description = "RandomForest on the auto-built feature bundle (no binder needed)."

    def __init__(self, *, n_estimators: int = 80, max_depth: int | None = 10,
                 seed: int = 0, select_tags: tuple[str, ...] = ()):
        super().__init__(select_tags=select_tags)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.seed = seed

    def model_params(self) -> dict[str, Any]:
        return {"model": "RandomForestRegressor",
                "n_estimators": self.n_estimators, "max_depth": self.max_depth}

    def _make_regressor(self):
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(
            n_estimators=self.n_estimators, max_depth=self.max_depth,
            random_state=self.seed, n_jobs=-1,
        )


class VersatileGradientBoostingToken(VersatileTabularToken):
    """Gradient-boosted trees (XGBoost-style) on the auto-built bundle.

    Uses sklearn's HistGradientBoostingRegressor so it runs without external
    XGBoost/LightGBM installs; swap ``_make_regressor`` for XGBRegressor to use
    XGBoost directly.
    """

    name = "versatile_gb"
    description = "Gradient boosting on the auto-built feature bundle."

    def __init__(self, *, max_iter: int = 150, learning_rate: float = 0.06,
                 max_depth: int | None = None, seed: int = 0,
                 select_tags: tuple[str, ...] = ()):
        super().__init__(select_tags=select_tags)
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.seed = seed

    def model_params(self) -> dict[str, Any]:
        return {"model": "HistGradientBoostingRegressor",
                "max_iter": self.max_iter, "learning_rate": self.learning_rate}

    def _make_regressor(self):
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(
            max_iter=self.max_iter, learning_rate=self.learning_rate,
            max_depth=self.max_depth, random_state=self.seed,
        )


# ---------------------------------------------------------------------------
# Exogenous calendar feature: add day-of-week-like info at any point
# ---------------------------------------------------------------------------


class DayOfWeekFeatureToken(FeatureToken):
    """Add a cyclical calendar covariate for history and known future.

    No timestamps are assumed: position index modulo ``period`` stands in for a
    seasonal calendar (e.g. day-of-week when period=7). Produces sine/cosine
    encodings on both the history and the forecast horizon, so any later model
    consumes them through the feature bundle without changes.
    """

    name = "DayOfWeekFeature"
    token_class = "feature"
    reads = ("raw_history",)
    writes = ("calendar_features", "future_calendar_features")
    description = "Cyclical calendar covariate (history + known future)."
    max_uses = 1
    provides = (
        Port(sem="features", alignment="history", space="any"),
        Port(sem="features", alignment="future", space="any"),
    )

    def __init__(self, period: int = 7):
        if period <= 1:
            raise ValueError("period must be > 1.")
        self.period = int(period)

    def check_specific_conditions(self, state: "State") -> bool:
        if state.flags.get("calendar_encoded", False):
            return False
        return super().check_specific_conditions(state)

    def apply(self, state: "State") -> "State":
        n = state.n_samples
        T = state.original_history.shape[1]
        H = state.horizon

        hist_pos = np.arange(T)
        fut_pos = np.arange(T, T + H)

        hist_feat = self._encode(hist_pos, n)
        fut_feat = self._encode(fut_pos, n)

        state.add_historical_feature("calendar_features", hist_feat)
        state.register_artifact(
            "calendar_features", store="historical_features", kind="tabular",
            role="feature", source_token=self.name, tags=("calendar", "exog"),
        )
        state.add_future_feature("future_calendar_features", fut_feat)
        state.register_artifact(
            "future_calendar_features", store="future_features", kind="tabular",
            role="known_future_feature", source_token=self.name,
            tags=("calendar", "exog"),
        )
        state.flags["calendar_encoded"] = True
        self._log_execution(
            state,
            reads={"raw_history": state.original_history.shape},
            writes={"calendar_features": hist_feat.shape,
                    "future_calendar_features": fut_feat.shape},
        )
        return state

    def _encode(self, positions: np.ndarray, n_samples: int) -> np.ndarray:
        phase = (positions % self.period).astype(np.float32)
        angle = 2.0 * np.pi * phase / self.period
        block = np.stack([np.sin(angle), np.cos(angle)], axis=1).astype(np.float32)
        return np.broadcast_to(block[None, :, :], (n_samples,) + block.shape).reshape(
            n_samples, -1
        ).astype(np.float32)


# ---------------------------------------------------------------------------
# Swappable FLAIR sub-step: predict the period scale with gradient boosting
# ---------------------------------------------------------------------------


class GBLevelForecastToken(ModelToken):
    """Forecast each period's level (scale) with gradient boosting, then expand.

    Drop-in alternative to ``FlairRidgeLevel``: it consumes the same FLAIR
    decomposition (``level_series`` + ``shape_vector`` + selected ``period``),
    learns a pooled autoregression over period levels with gradient-boosted
    trees, rolls it forward, and expands through the within-period shape to a
    full-horizon point forecast.
    """

    name = "gb_level_forecast"
    token_class = "model"
    reads = ("level_series", "shape_vector", "period")
    writes = ("prediction_stack", "gb_level_point")
    description = "Gradient-boosted period-level forecaster + FLAIR shape expansion."
    requires = (
        Port(sem="level", axes=("sample", "period_idx"), space="any"),
        Port(sem="shape", axes=("sample", "phase"), space="any"),
    )

    def __init__(self, *, n_lags: int = 3, max_iter: int = 120,
                 learning_rate: float = 0.08, seed: int = 0):
        self.n_lags = int(n_lags)
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.seed = seed

    def get_model(self) -> dict[str, Any]:
        return {"model": "HistGradientBoostingRegressor", "target": "period_level",
                "n_lags": self.n_lags}

    def check_specific_conditions(self, state: "State") -> bool:
        if not super().check_specific_conditions(state):
            return False
        if "period" not in state.metadata:
            return False
        non_flair = [t for t in state.transform_stack if t.name != "flair_shift"]
        if state.n_models_applied > 0 or non_flair:
            return False
        return "shape_vector" in state.historical_features

    def apply(self, state: "State") -> "State":
        from sklearn.ensemble import HistGradientBoostingRegressor

        level = np.asarray(state.historical_features["level_series"], dtype=np.float64)
        shape = np.asarray(state.historical_features["shape_vector"], dtype=np.float64)
        period = int(state.metadata["period"])
        horizon = state.horizon
        m = int(np.ceil(horizon / max(period, 1)))
        n_samples, n_complete = level.shape
        L = max(1, min(self.n_lags, n_complete - 1))

        # Pool (lag-window -> next-level) pairs across all series and positions.
        rows, targets = [], []
        for i in range(n_samples):
            for t in range(L, n_complete):
                rows.append(level[i, t - L:t])
                targets.append(level[i, t])
        level_point = np.repeat(level[:, -1:], m, axis=1).astype(np.float64)
        if rows:
            X = np.asarray(rows, dtype=np.float32)
            y = np.asarray(targets, dtype=np.float32)
            model = HistGradientBoostingRegressor(
                max_iter=self.max_iter, learning_rate=self.learning_rate,
                random_state=self.seed,
            )
            model.fit(X, y)
            for i in range(n_samples):
                window = list(level[i, -L:])
                preds = []
                for _ in range(m):
                    x = np.asarray(window[-L:], dtype=np.float32)[None, :]
                    nxt = float(model.predict(x)[0])
                    preds.append(nxt)
                    window.append(nxt)
                level_point[i] = preds

        pred = np.zeros((n_samples, horizon), dtype=np.float32)
        phases = np.arange(horizon) % max(period, 1)
        steps = np.arange(horizon) // max(period, 1)
        for i in range(n_samples):
            pred[i] = (level_point[i, steps] * shape[i, phases]).astype(np.float32)

        state.features["gb_level_point"] = level_point.astype(np.float32)
        state.metadata[self.name] = {**self.get_model(), "period": period,
                                     "level_steps": m, "n_lags": L}
        state.push_prediction(pred, self.name)
        self._log_execution(
            state,
            reads={"level_series": level.shape, "shape_vector": shape.shape,
                   "period": period},
            writes={"prediction_stack[-1]": pred.shape,
                    "gb_level_point": level_point.shape},
        )
        return state
