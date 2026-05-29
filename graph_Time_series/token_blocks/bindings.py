"""Binder tokens that turn broad state artifacts into active model inputs."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..token import FeatureToken

if TYPE_CHECKING:
    from ..state import State


class BindFeatureToken(FeatureToken):
    """Bind one named artifact as the active model input."""

    token_class = "binding"
    writes = ("model_input", "active_input_bundle")
    description = "Bind a single named artifact into the active model_input slot."

    def __init__(
        self,
        feature_name: str,
        *,
        token_name: str | None = None,
        bundle_name: str | None = None,
        kind: str = "sequence_flat",
        target_space: str = "active_target",
    ):
        self.feature_name = feature_name
        self.name = token_name or f"Bind[{feature_name}]"
        self.bundle_name = bundle_name or feature_name
        self.kind = kind
        self.target_space = target_space
        self.reads = (feature_name,)

    def apply(self, state: "State") -> "State":
        value = self._lookup_feature(state, self.feature_name)
        bundle = state.set_model_input(
            self.bundle_name,
            value,
            artifact_names=(self.feature_name,),
            kind=self.kind,
            target_space=self.target_space,
            source_token=self.name,
        )
        self._log_execution(
            state,
            reads={self.feature_name: value.shape},
            writes={
                "model_input": value.shape,
                "active_input_bundle": bundle.to_dict(),
            },
        )
        return state

    @staticmethod
    def _lookup_feature(state: "State", name: str) -> np.ndarray:
        for store_name in ("historical_features", "features", "future_features"):
            store = getattr(state, store_name)
            if name in store:
                return np.asarray(store[name])
        raise KeyError(f"Feature {name!r} is not available in State.")


class BindScaledHistoryToken(BindFeatureToken):
    """Bind Z-normalized history for generic model tokens."""

    def __init__(self):
        super().__init__(
            "scaled_history",
            token_name="BindScaledHistory",
            bundle_name="scaled_history",
            kind="sequence_flat",
        )


class StackFeatureBundleToken(FeatureToken):
    """Flatten and concatenate selected artifacts into a tabular bundle."""

    token_class = "binding"
    writes = ("model_input", "active_input_bundle")
    description = "Bind several named artifacts as one tabular model_input."

    def __init__(
        self,
        feature_names: tuple[str, ...],
        *,
        token_name: str | None = None,
        bundle_name: str | None = None,
        kind: str = "tabular",
        target_space: str = "active_target",
    ):
        if not feature_names:
            raise ValueError("feature_names must not be empty.")
        self.feature_names = tuple(feature_names)
        self.name = token_name or "Stack[" + "+".join(self.feature_names) + "]"
        self.bundle_name = bundle_name or "+".join(self.feature_names)
        self.kind = kind
        self.target_space = target_space
        self.reads = self.feature_names

    def apply(self, state: "State") -> "State":
        arrays = [BindFeatureToken._lookup_feature(state, name) for name in self.feature_names]
        flat = [np.asarray(arr, dtype=np.float32).reshape(state.n_samples, -1) for arr in arrays]
        model_input = np.concatenate(flat, axis=1)
        bundle = state.set_model_input(
            self.bundle_name,
            model_input,
            artifact_names=self.feature_names,
            kind=self.kind,
            target_space=self.target_space,
            source_token=self.name,
            metadata={"component_shapes": [tuple(arr.shape) for arr in arrays]},
        )
        self._log_execution(
            state,
            reads={name: arr.shape for name, arr in zip(self.feature_names, arrays)},
            writes={
                "model_input": model_input.shape,
                "active_input_bundle": bundle.to_dict(),
            },
        )
        return state


class BindAllSafeTabularToken(FeatureToken):
    """Bind all finite historical artifacts into one curated tabular bundle."""

    name = "BindAllSafeTabular"
    token_class = "binding"
    reads = ()
    writes = ("model_input", "active_input_bundle")
    description = "Flatten all finite historical artifacts except model_input."

    def check_specific_conditions(self, state: "State") -> bool:
        return any(name != "model_input" for name in state.historical_features)

    def apply(self, state: "State") -> "State":
        feature_names = []
        arrays = []
        for name, value in state.historical_features.items():
            if name == "model_input":
                continue
            arr = np.asarray(value)
            if arr.shape[0] != state.n_samples or not np.all(np.isfinite(arr)):
                continue
            feature_names.append(name)
            arrays.append(arr)

        if not arrays:
            raise ValueError("No safe historical artifacts are available to bind.")

        flat = [np.asarray(arr, dtype=np.float32).reshape(state.n_samples, -1) for arr in arrays]
        model_input = np.concatenate(flat, axis=1)
        bundle = state.set_model_input(
            "all_safe_tabular",
            model_input,
            artifact_names=tuple(feature_names),
            kind="tabular",
            source_token=self.name,
            metadata={"component_shapes": [tuple(arr.shape) for arr in arrays]},
        )
        self._log_execution(
            state,
            reads={name: arr.shape for name, arr in zip(feature_names, arrays)},
            writes={
                "model_input": model_input.shape,
                "active_input_bundle": bundle.to_dict(),
            },
        )
        return state
