"""Versatile, sequence-agnostic tokens built on the Signal board.

These tokens demonstrate the combinatorial architecture in action:

* ``VersatileTabularToken`` fits *any* tabular regressor on whatever feature
  signals exist in the active scale, with no binder token required. The same
  token works after normalization, after Fourier features, or after a FLAIR
  fold, because it consumes the auto-built feature bundle instead of one
  hard-coded ``model_input`` array.

* ``GBLevelForecastToken`` shows swapping a FLAIR sub-step for a different
  learner: it predicts the *scale of each period* with gradient boosting
  instead of FLAIR's soft-averaged ridge, then expands through the FLAIR shape
  to a full forecast. Any model with a ``level`` port could take its place.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from ..token import ModelToken
from ..signal import Port

if TYPE_CHECKING:
    from ..state import State


# ---------------------------------------------------------------------------
# Versatile tabular model: any features -> any regressor
# ---------------------------------------------------------------------------


class VersatileTabularToken(ModelToken):
    learning_scope = "cross_series"
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
# Swappable FLAIR sub-step: predict the period scale with gradient boosting
# ---------------------------------------------------------------------------


class GBLevelForecastToken(ModelToken):
    learning_scope = "cross_series"  # pooled period-level booster across series
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
