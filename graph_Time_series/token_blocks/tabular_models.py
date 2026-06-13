"""Tabular model tokens for multi-horizon forecasting."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

from ..token import ModelToken
from ..signal import Port

if TYPE_CHECKING:
    from ..state import State


class _TabularRegressorToken(ModelToken):
    """Base class for fit-on-current-state tabular regressors."""

    learning_scope = "cross_series"  # one model over all series (rows = series)

    reads = ()
    writes = ("prediction_stack",)
    accepted_input_kinds = {"tabular"}
    requires = (
        Port(sem="features", axes=("sample", "feature"), space="current",
             multiple=True, coerce=True),
    )

    def check_specific_conditions(self, state: "State") -> bool:
        if not super().check_specific_conditions(state):
            return False
        try:
            _, bundle = self._resolve_model_input(state)
        except (KeyError, ValueError):
            return False
        return bundle is None or bundle.kind in self.accepted_input_kinds

    def apply(self, state: "State") -> "State":
        X, bundle = self._resolve_model_input(state)
        Y = np.asarray(state.current_target, dtype=np.float32)
        model = self._build_model()
        model.fit(X, Y)
        pred = np.asarray(model.predict(X), dtype=np.float32)
        if pred.ndim == 1:
            pred = pred[:, None]

        state.push_prediction(pred, self.name)
        state.metadata[self.name] = {
            **self.get_model(),
            "input_shape": tuple(X.shape),
            "target_shape": tuple(Y.shape),
            "bundle": bundle.to_dict() if bundle is not None else None,
            "mode": "fit_once_in_state",
        }
        self._log_execution(
            state,
            reads={
                "model_input": X.shape,
                "active_input_bundle": bundle.to_dict() if bundle else None,
                "current_target": Y.shape,
            },
            writes={"prediction_stack[-1]": pred.shape},
        )
        return state

    def _resolve_model_input(self, state: "State"):
        # Explicit binder bundle wins when present (back-compat); otherwise the
        # model fuses whatever features exist via the Signal-board auto-bundle.
        if "model_input" in state.historical_features:
            bundle = state.active_input_bundle()
            if bundle is not None and bundle.kind not in self.accepted_input_kinds:
                raise ValueError(
                    f"{self.name} accepts {sorted(self.accepted_input_kinds)}, "
                    f"got bundle kind {bundle.kind!r}."
                )
            X = np.asarray(state.historical_features["model_input"], dtype=np.float32)
            return X.reshape(state.n_samples, -1), bundle
        fb = state.feature_bundle()
        return np.asarray(fb.matrix, dtype=np.float32), None

    def get_model(self) -> dict[str, Any]:
        return self.model_params()

    def model_params(self) -> dict[str, Any]:
        raise NotImplementedError

    def _build_model(self):
        raise NotImplementedError


class RandomForestTabularToken(_TabularRegressorToken):
    """Random forest regressor over active tabular model_input."""

    name = "rf_tabular"
    description = "RandomForestRegressor on active tabular model_input."

    def __init__(
        self,
        *,
        n_estimators: int = 120,
        max_depth: int | None = 12,
        min_samples_leaf: int = 2,
        seed: int = 0,
        n_jobs: int = -1,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.seed = seed
        self.n_jobs = n_jobs

    def model_params(self) -> dict[str, Any]:
        return {
            "model": "RandomForestRegressor",
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_leaf": self.min_samples_leaf,
            "seed": self.seed,
        }

    def _build_model(self) -> RandomForestRegressor:
        return RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.seed,
            n_jobs=self.n_jobs,
        )


class LightGBMTabularToken(_TabularRegressorToken):
    """LightGBM regressor over active tabular model_input."""

    name = "lightgbm_tabular"
    description = "LightGBM LGBMRegressor on active tabular model_input."

    def __init__(
        self,
        *,
        n_estimators: int = 200,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        max_depth: int = -1,
        min_child_samples: int = 20,
        subsample: float = 0.9,
        colsample_bytree: float = 0.9,
        seed: int = 0,
        n_jobs: int = -1,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.min_child_samples = min_child_samples
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.seed = seed
        self.n_jobs = n_jobs

    def model_params(self) -> dict[str, Any]:
        return {
            "model": "LightGBMRegressor",
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "max_depth": self.max_depth,
            "min_child_samples": self.min_child_samples,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "seed": self.seed,
        }

    def _build_model(self) -> MultiOutputRegressor:
        try:
            from lightgbm import LGBMRegressor
        except ImportError as exc:
            raise ImportError(
                "LightGBM is required for lightgbm_tabular. "
                "Install it with `pip install lightgbm`."
            ) from exc

        base = LGBMRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            max_depth=self.max_depth,
            min_child_samples=self.min_child_samples,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=self.seed,
            n_jobs=self.n_jobs,
            verbose=-1,
        )
        return MultiOutputRegressor(base)
