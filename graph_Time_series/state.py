"""State contract for grammar-guided time-series pipeline search.

The State object is the shared runtime contract between the grammar, tokens,
and MCTS rollouts. Tokens may create features, transform the target scale, fit
models, and push predictions, but State owns the invariants that keep those
steps consistent.

Important design rules:
    * original_history and original_future are treated as immutable inputs.
    * active_target_base is the target in the current modeling scale.
    * current_target is always the residual in that active modeling scale.
    * prediction_stack contains model outputs in the active modeling scale.
    * transform_stack stores structured inverse transforms for final decoding.

This keeps residual stacking correct even after target transforms such as
z-normalization, log transforms, differencing, or Box-Cox transforms.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from .artifacts import ArtifactSpec, InputBundle


Array = np.ndarray
InverseFn = Callable[[Array], Array]


@dataclass
class TransformRecord:
    """Inspectable record for a target/feature transform.

    Raw callables are flexible but opaque in logs. A record keeps the inverse
    function together with the transform name and useful debug parameters.
    """

    name: str
    inverse_fn: InverseFn | None = None
    params: dict[str, Any] = field(default_factory=dict)
    affects: str = "target"


class State:
    """Runtime state for a forecasting pipeline.

    The class intentionally centralizes bookkeeping and validation. Concrete
    tokens should do their domain work, then ask State to record the action,
    add features, register transforms, or push predictions.
    """

    def __init__(self, original_history: Array, original_future: Array):
        self.original_history = self._as_2d_numeric_array(
            original_history, "original_history"
        )
        self.original_future = self._as_2d_numeric_array(
            original_future, "original_future"
        )
        if self.original_history.shape[0] != self.original_future.shape[0]:
            raise ValueError(
                "original_history and original_future must have the same "
                f"number of samples; got {self.original_history.shape[0]} "
                f"and {self.original_future.shape[0]}."
            )

        # Feature stores are separated by alignment. historical_features are
        # model-training inputs; future_features are known horizon-side inputs
        # such as calendar variables, holidays, prices, or recursive lags.
        self.historical_features: dict[str, Array] = {}
        self.future_features: dict[str, Array] = {}

        # General-purpose output/compatibility store. Older tokens may still
        # read "raw_history" or write values such as "final_forecast" here.
        self.features: dict[str, Any] = {"raw_history": self.original_history.copy()}

        self.metadata: dict[str, Any] = {}
        self.flags: dict[str, Any] = {}

        # Visible artifact/bundle registry. Feature tokens may create many
        # named artifacts; binder tokens choose one approved active model input.
        self.artifacts: dict[str, ArtifactSpec] = {}
        self.input_bundles: dict[str, InputBundle] = {}
        self.active_input_bundle_name: str | None = None
        self.register_artifact(
            "raw_history",
            store="features",
            kind="sequence",
            role="raw_input",
            source_token="State.__init__",
        )

        self.token_sequence: list[str] = []
        self.class_counts: dict[str, int] = {}

        self.active_target_base = self.original_future.copy()
        self.current_target = self.active_target_base.copy()

        self.prediction_stack: list[Array] = []
        self.prediction_names: list[str] = []

        # Structured transform history for debugging/replay. pending_conditions
        # is kept as a compatibility alias for older code that expects a list
        # of inverse callables.
        self.transform_stack: list[TransformRecord] = []
        self.pending_conditions: list[InverseFn] = []

        self.log: list[dict[str, Any]] = []
        self.terminated = False
        self.mase: float | None = None

    # ------------------------------------------------------------------
    # Basic properties
    # ------------------------------------------------------------------

    @property
    def n_samples(self) -> int:
        return self.original_history.shape[0]

    @property
    def horizon(self) -> int:
        return self.original_future.shape[1]

    @property
    def last_token(self) -> str:
        return self.token_sequence[-1] if self.token_sequence else "START"

    @property
    def depth(self) -> int:
        return len(self.token_sequence)

    @property
    def n_models_applied(self) -> int:
        return len(self.prediction_stack)

    # ------------------------------------------------------------------
    # Feature validation and storage
    # ------------------------------------------------------------------

    def add_historical_feature(
        self, name: str, value: Array, *, require_finite: bool = True
    ) -> Array:
        """Validate and store a feature aligned to the input series."""
        feature = self._validate_feature(name, value, require_finite=require_finite)
        self.historical_features[name] = feature
        self.register_artifact(name, store="historical_features")
        return feature

    def add_future_feature(
        self, name: str, value: Array, *, require_finite: bool = True
    ) -> Array:
        """Validate and store a feature aligned to the forecast horizon."""
        feature = self._validate_feature(name, value, require_finite=require_finite)
        self.future_features[name] = feature
        self.register_artifact(name, store="future_features")
        return feature

    def register_artifact(
        self,
        name: str,
        *,
        store: str,
        kind: str = "array",
        role: str = "feature",
        target_space: str = "active_target",
        source_token: str | None = None,
        tags: tuple[str, ...] = (),
    ) -> ArtifactSpec:
        """Register inspectable metadata for a named value in State."""
        value = self._store_by_name(store).get(name)
        shape = tuple(value.shape) if hasattr(value, "shape") else None
        spec = ArtifactSpec(
            name=name,
            store=store,
            kind=kind,
            shape=shape,
            role=role,
            target_space=target_space,
            source_token=source_token,
            tags=tuple(tags),
        )
        self.artifacts[name] = spec
        self.metadata.setdefault("artifacts", {})[name] = spec.to_dict()
        return spec

    def set_model_input(
        self,
        name: str,
        value: Array,
        *,
        artifact_names: tuple[str, ...],
        kind: str,
        target_space: str = "active_target",
        source_token: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> InputBundle:
        """Set the active model input while preserving broad named artifacts."""
        model_input = self._validate_feature(
            "model_input", value, require_finite=True
        )
        self.historical_features["model_input"] = model_input
        self.features["model_input"] = model_input

        bundle = InputBundle(
            name=name,
            kind=kind,
            artifact_names=tuple(artifact_names),
            shape=tuple(model_input.shape),
            target_space=target_space,
            source_token=source_token,
            metadata=metadata or {},
        )
        self.input_bundles[name] = bundle
        self.active_input_bundle_name = name
        self.metadata["active_input_bundle"] = bundle.to_dict()
        self.metadata.setdefault("input_bundles", {})[name] = bundle.to_dict()
        self.register_artifact(
            "model_input",
            store="historical_features",
            kind=kind,
            role="active_model_input",
            target_space=target_space,
            source_token=source_token,
            tags=("bundle",),
        )
        return bundle

    def active_input_bundle(self) -> InputBundle | None:
        """Return metadata for the currently bound model input."""
        if self.active_input_bundle_name is None:
            return None
        return self.input_bundles.get(self.active_input_bundle_name)

    def describe_artifacts(self) -> list[dict[str, Any]]:
        """Return artifact metadata in a notebook-friendly format."""
        return [spec.to_dict() for spec in self.artifacts.values()]

    def describe_input_bundles(self) -> list[dict[str, Any]]:
        """Return model-input bundle metadata in a notebook-friendly format."""
        return [bundle.to_dict() for bundle in self.input_bundles.values()]

    def _validate_feature(
        self, name: str, value: Array, *, require_finite: bool = True
    ) -> Array:
        feature = np.asarray(value)
        if feature.ndim == 0:
            raise ValueError(f"Feature {name!r} must be array-like, got a scalar.")
        if feature.shape[0] != self.n_samples:
            raise ValueError(
                f"Feature {name!r} has first dimension {feature.shape[0]}, "
                f"expected {self.n_samples}."
            )
        if require_finite and not np.all(np.isfinite(feature)):
            raise ValueError(f"Feature {name!r} contains NaN or infinite values.")
        return feature

    # ------------------------------------------------------------------
    # Target transforms and residual stacking
    # ------------------------------------------------------------------

    def target_base(self) -> Array:
        """Compatibility accessor for the current modeling target base."""
        return self.active_target_base

    def set_target_base(self, transformed_target: Array) -> Array:
        """Set the target scale used by future model tokens.

        This is the core fix for transform-aware residual stacking:
        current_target is recomputed as active_target_base minus the cumulative
        prediction stack, so every model learns residuals in the same scale.
        """
        target = self._validate_target_like(transformed_target, "transformed_target")
        self.active_target_base = target
        self.current_target = self.active_target_base - self.cumulative_prediction()
        return self.active_target_base

    def register_transform(
        self,
        name: str,
        inverse_fn: InverseFn | None = None,
        *,
        transformed_target: Array | None = None,
        params: dict[str, Any] | None = None,
        affects: str = "target",
    ) -> TransformRecord:
        """Register a transform and optionally update the active target scale."""
        if transformed_target is not None:
            self.set_target_base(transformed_target)

        record = TransformRecord(
            name=name,
            inverse_fn=inverse_fn,
            params=params or {},
            affects=affects,
        )
        self.transform_stack.append(record)
        if inverse_fn is not None:
            self.pending_conditions.append(inverse_fn)
        return record

    def register_inverse_transform(
        self,
        inverse_fn: InverseFn,
        *,
        name: str = "anonymous_transform",
        params: dict[str, Any] | None = None,
        affects: str = "target",
    ) -> TransformRecord:
        """Backward-compatible helper for older transform tokens.

        Older tokens often mutate current_target first and then register an
        inverse function. To keep residuals correct, this method treats the
        current_target value as the new active target base.
        """
        transformed_target = (
            self.current_target if affects in {"target", "both"} else None
        )
        return self.register_transform(
            name,
            inverse_fn,
            transformed_target=transformed_target,
            params=params,
            affects=affects,
        )

    def get_final_prediction(self) -> Array:
        """Decode the cumulative prediction back to the original data scale."""
        pred = self.cumulative_prediction()

        if self.transform_stack:
            for transform in reversed(self.transform_stack):
                if transform.inverse_fn is not None and transform.affects in {
                    "target",
                    "both",
                }:
                    pred = transform.inverse_fn(pred)
            return pred

        # Compatibility path for states created before transform_stack existed.
        for inverse_fn in reversed(self.pending_conditions):
            pred = inverse_fn(pred)
        return pred

    def cumulative_prediction(self) -> Array:
        """Sum all model predictions in the active modeling scale."""
        if not self.prediction_stack:
            return np.zeros_like(self.active_target_base)
        return np.sum(self.prediction_stack, axis=0)

    def last_prediction(self) -> Array:
        """Most recent model output, useful for decoders and inspectors."""
        if not self.prediction_stack:
            return np.zeros_like(self.active_target_base)
        return self.prediction_stack[-1]

    def validate_prediction(self, pred: Array) -> Array:
        """Validate a model output before it enters the prediction stack."""
        pred = np.asarray(pred)
        if pred.shape != self.current_target.shape:
            raise ValueError(
                f"Prediction shape {pred.shape} does not match current_target "
                f"shape {self.current_target.shape}."
            )
        if not np.all(np.isfinite(pred)):
            raise ValueError("Prediction contains NaN or infinite values.")
        if not np.issubdtype(pred.dtype, np.floating):
            pred = pred.astype(np.float32)
        return pred

    def push_prediction(self, pred: Array, token_name: str) -> Array:
        """Add model output and update the residual in the active target scale."""
        pred = self.validate_prediction(pred)
        self.prediction_stack.append(pred)
        self.prediction_names.append(token_name)

        cumulative = self.cumulative_prediction()
        self.current_target = self.active_target_base - cumulative

        self.metadata["last_prediction"] = pred
        self.metadata["cumulative_prediction"] = cumulative
        self.metadata["current_residual"] = self.current_target
        self.features["last_prediction"] = pred
        self.features["cumulative_prediction"] = cumulative
        self.features["current_residual"] = self.current_target
        return pred

    # ------------------------------------------------------------------
    # Bookkeeping and logs
    # ------------------------------------------------------------------

    def record_token(
        self,
        token_name: str,
        token_class: str,
        reads: dict[str, Any] | None = None,
        writes: dict[str, Any] | None = None,
    ) -> None:
        """Record token order, class counts, and execution log together."""
        self.token_sequence.append(token_name)
        self.class_counts[token_class] = self.class_counts.get(token_class, 0) + 1
        self.log_step(token_name, reads or {}, writes or {})

    def log_step(self, token_name: str, reads: dict, writes: dict) -> None:
        """Record what a token did. reads/writes usually map names to shapes."""
        self.log.append(
            {
                "step": len(self.log),
                "token": token_name,
                "reads": reads,
                "writes": writes,
            }
        )

    def print_log(self) -> None:
        """Pretty-print the transformation log."""
        print(f"{'Step':>4}  {'Token':<20}  {'Reads':<35}  {'Writes':<35}")
        print("-" * 96)
        for entry in self.log:
            reads = ", ".join(f"{k}{v}" for k, v in entry["reads"].items())
            writes = ", ".join(f"{k}{v}" for k, v in entry["writes"].items())
            print(f"{entry['step']:4d}  {entry['token']:<20}  {reads:<35}  {writes:<35}")

    # ------------------------------------------------------------------
    # Copy support for MCTS rollouts
    # ------------------------------------------------------------------

    def copy(self) -> "State":
        """Deep copy mutable rollout state while sharing immutable inputs."""
        s = State.__new__(State)
        s.original_history = self.original_history
        s.original_future = self.original_future

        s.historical_features = self._copy_mapping(self.historical_features)
        s.future_features = self._copy_mapping(self.future_features)
        s.features = self._copy_mapping(self.features)

        s.metadata = deepcopy(self.metadata)
        s.flags = deepcopy(self.flags)
        s.artifacts = deepcopy(self.artifacts)
        s.input_bundles = deepcopy(self.input_bundles)
        s.active_input_bundle_name = self.active_input_bundle_name

        s.token_sequence = list(self.token_sequence)
        s.class_counts = dict(self.class_counts)

        s.active_target_base = self.active_target_base.copy()
        s.current_target = self.current_target.copy()

        s.prediction_stack = [p.copy() for p in self.prediction_stack]
        s.prediction_names = list(self.prediction_names)

        # Transform callables are shared by reference. Their params are copied
        # so logs can be edited/debugged without affecting sibling rollouts.
        s.transform_stack = [
            TransformRecord(
                name=t.name,
                inverse_fn=t.inverse_fn,
                params=deepcopy(t.params),
                affects=t.affects,
            )
            for t in self.transform_stack
        ]
        s.pending_conditions = list(self.pending_conditions)

        s.log = deepcopy(self.log)
        s.terminated = self.terminated
        s.mase = self.mase
        return s

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _copy_mapping(values: dict[str, Any]) -> dict[str, Any]:
        copied = {}
        for key, value in values.items():
            copied[key] = value.copy() if hasattr(value, "copy") else deepcopy(value)
        return copied

    def _store_by_name(self, store: str) -> dict[str, Any]:
        stores = {
            "historical_features": self.historical_features,
            "future_features": self.future_features,
            "features": self.features,
            "metadata": self.metadata,
            "flags": self.flags,
        }
        if store not in stores:
            raise ValueError(f"Unknown state store {store!r}.")
        return stores[store]

    @staticmethod
    def _as_2d_numeric_array(value: Array, name: str) -> Array:
        arr = np.asarray(value)
        if arr.ndim != 2:
            raise ValueError(f"{name} must be a 2D array, got shape {arr.shape}.")
        if not np.issubdtype(arr.dtype, np.number):
            raise TypeError(f"{name} must contain numeric values, got {arr.dtype}.")
        if not np.issubdtype(arr.dtype, np.floating):
            arr = arr.astype(np.float32)
        return arr

    def _validate_target_like(self, value: Array, name: str) -> Array:
        target = np.asarray(value)
        if target.shape != self.original_future.shape:
            raise ValueError(
                f"{name} shape {target.shape} must match original_future "
                f"shape {self.original_future.shape}."
            )
        if not np.all(np.isfinite(target)):
            raise ValueError(f"{name} contains NaN or infinite values.")
        if not np.issubdtype(target.dtype, np.floating):
            target = target.astype(np.float32)
        return target

    def __repr__(self) -> str:
        return (
            f"State(tokens={self.token_sequence}, "
            f"hist_feats={list(self.historical_features.keys())}, "
            f"future_feats={list(self.future_features.keys())}, "
            f"active_bundle={self.active_input_bundle_name!r}, "
            f"preds={len(self.prediction_stack)}, "
            f"transforms={[t.name for t in self.transform_stack]}, "
            f"term={self.terminated})"
        )
