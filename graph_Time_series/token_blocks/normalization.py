import numpy as np
from typing import TYPE_CHECKING

from ..token import TransformToken

if TYPE_CHECKING:
    from ..state import State


class MeanAbsScalingToken(TransformToken):
    """Scale each series by mean absolute history value; no centering."""

    name = "MeanAbsScaling"
    reads = ("raw_history",)
    writes = ("scaled_history", "active_target_base")
    description = "Scale by mean absolute value of history; no centering."

    def check_specific_conditions(self, state: "State") -> bool:
        if state.flags.get("is_mean_abs_scaled", False):
            return False
        return super().check_specific_conditions(state)

    def apply(self, state: "State") -> "State":
        hist = np.asarray(state.features["raw_history"], dtype=np.float32)
        scale = np.mean(np.abs(hist), axis=-1, keepdims=True) + 1e-8

        scaled_history = hist / scale
        scaled_future = state.target_base() / scale
        state.add_historical_feature("scaled_history", scaled_history)

        def inverse_fn(predictions: np.ndarray) -> np.ndarray:
            return predictions * scale

        state.register_transform(
            name=self.name,
            inverse_fn=inverse_fn,
            transformed_target=scaled_future,
            params={"scale_shape": scale.shape},
            affects="target",
        )
        state.flags["is_mean_abs_scaled"] = True

        self._log_execution(
            state,
            reads={"raw_history": hist.shape},
            writes={
                "scaled_history": scaled_history.shape,
                "active_target_base": state.active_target_base.shape,
            },
        )
        return state


class ZNormalizationToken(TransformToken):
    name = "ZNormalization"
    reads = ("raw_history",)
    writes = ("scaled_history", "active_target_base")
    description = "Instance Z-Normalization: centers and scales each sample by history mean/std."

    def check_specific_conditions(self, state: "State") -> bool:
        if state.flags.get("is_z_normalized", False):
            return False
        return super().check_specific_conditions(state)

    def apply(self, state: "State") -> "State":
        hist = np.asarray(state.features["raw_history"], dtype=np.float32)
        mu = np.mean(hist, axis=-1, keepdims=True)
        sigma = np.std(hist, axis=-1, keepdims=True) + 1e-8

        norm_history = (hist - mu) / sigma
        norm_future = (state.target_base() - mu) / sigma
        state.add_historical_feature("scaled_history", norm_history)

        def inverse_fn(predictions: np.ndarray) -> np.ndarray:
            return (predictions * sigma) + mu

        state.register_transform(
            name=self.name,
            inverse_fn=inverse_fn,
            transformed_target=norm_future,
            params={"mu_shape": mu.shape, "sigma_shape": sigma.shape},
            affects="target",
        )
        state.flags["is_z_normalized"] = True

        self._log_execution(
            state,
            reads={"raw_history": hist.shape},
            writes={
                "scaled_history": norm_history.shape,
                "active_target_base": state.active_target_base.shape,
            },
        )
        return state
