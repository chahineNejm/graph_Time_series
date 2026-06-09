import numpy as np
from typing import TYPE_CHECKING

from ..token import TransformToken

if TYPE_CHECKING:
    from ..state import State


class MeanAbsScalingToken(TransformToken):
    """Scale each series by its mean absolute value; no centering.

    m = 0,  s = (1/C) Σ|xᵢ|   →   x̃ = x / s
    Inverse: pred_original = pred_scaled * s
    """

    name = "MeanAbsScaling"
    reads = ("raw_history",)
    writes = ("scaled_history", "active_target_base")
    description = "Scale by mean absolute value of history; no centering (m=0, s=(1/C)Σ|xᵢ|)."

    def check_specific_conditions(self, state: "State") -> bool:
        if state.flags.get("is_mean_abs_scaled", False):
            return False
        return super().check_specific_conditions(state)

    def apply(self, state: "State") -> "State":
        s = np.mean(np.abs(state.original_history), axis=-1, keepdims=True) + 1e-8

        scaled_history = state.original_history / s
        scaled_future = state.target_base() / s
        state.add_historical_feature("scaled_history", scaled_history)

        def inverse_fn(predictions: np.ndarray) -> np.ndarray:
            return predictions * s

        state.register_transform(
            name=self.name,
            inverse_fn=inverse_fn,
            transformed_target=scaled_future,
            params={"s_shape": s.shape},
            affects="target",
        )
        state.flags["is_mean_abs_scaled"] = True

        self._log_execution(
            state,
            reads={"raw_history": state.original_history.shape},
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
    description = "Instance Z-Normalization: Centers and scales data per-sample based on historical mean/std."
    
    def check_specific_conditions(self, state: 'State') -> bool:
        # Prevent the RL agent from applying Z-Norm multiple times in a row
        if state.flags.get("is_z_normalized", False):
            return False
        return super().check_specific_conditions(state)

    def apply(self, state: 'State') -> 'State':
        mu = np.mean(state.original_history, axis=-1, keepdims=True)
        sigma = np.std(state.original_history, axis=-1, keepdims=True) + 1e-8

        norm_history = (state.original_history - mu) / sigma
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
            reads={"raw_history": state.original_history.shape},
            writes={
                "scaled_history": norm_history.shape,
                "active_target_base": state.active_target_base.shape,
            },
        )
        return state
