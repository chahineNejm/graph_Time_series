import numpy as np
from typing import TYPE_CHECKING, Any

from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import pairwise_distances

from ..token import ModelToken

if TYPE_CHECKING:
    from ..state import State


class KernelRBFToken(ModelToken):
    name = "kernel_rbf"
    reads = ()
    writes = ("prediction_stack",)
    accepted_input_kinds = {"sequence_flat", "tabular"}
    description = "RBF KernelRidge with median-heuristic lengthscale and leave-one-out predictions."

    def __init__(self, alpha: float = 1e-2, median_subset: int = 24, seed: int = 0):
        self.alpha = alpha
        self.median_subset = median_subset
        self.seed = seed
        self.last_lengthscale = None
        self.last_gamma = None

    def get_model(self) -> Any:
        return {
            "model": "KernelRidge",
            "kernel": "rbf",
            "alpha": self.alpha,
            "median_subset": self.median_subset,
        }

    def apply(self, state: 'State') -> 'State':
        X, input_name, bundle = self._resolve_model_input(state)
        X = self._flatten_samples(X)
        Y = np.asarray(state.current_target, dtype=np.float32)

        lengthscale = self._median_lengthscale(X)
        gamma = 1.0 / (2.0 * lengthscale ** 2)
        self.last_lengthscale = lengthscale
        self.last_gamma = gamma

        predictions = np.zeros_like(Y)
        for i in range(X.shape[0]):
            mask = np.ones(X.shape[0], dtype=bool)
            mask[i] = False
            if not np.any(mask):
                continue

            model = KernelRidge(kernel="rbf", alpha=self.alpha, gamma=gamma)
            model.fit(X[mask], Y[mask])
            predictions[i] = model.predict(X[i:i + 1])[0]

        state.push_prediction(predictions, self.name)
        bundle_info = bundle.to_dict() if bundle is not None else None
        self._log_execution(
            state,
            reads={
                input_name: X.shape,
                "active_input_bundle": bundle_info,
                "current_target": Y.shape,
            },
            writes={
                "prediction_stack[-1]": predictions.shape,
                "lengthscale": lengthscale,
                "gamma": gamma,
            },
        )
        return state

    def check_specific_conditions(self, state: 'State') -> bool:
        if not super().check_specific_conditions(state):
            return False
        try:
            _, _, bundle = self._resolve_model_input(state)
        except (KeyError, ValueError):
            return False
        if bundle is None:
            return True
        return bundle.kind in self.accepted_input_kinds

    def _resolve_model_input(self, state: 'State'):
        if "model_input" in state.historical_features:
            bundle = state.active_input_bundle()
            if bundle is not None and bundle.kind not in self.accepted_input_kinds:
                raise ValueError(
                    f"{self.name} accepts {sorted(self.accepted_input_kinds)}, "
                    f"got bundle kind {bundle.kind!r}."
                )
            return (
                np.asarray(state.historical_features["model_input"], dtype=np.float32),
                "model_input",
                bundle,
            )

        # No binder: fuse available features via the Signal-board auto-bundle.
        try:
            fb = state.feature_bundle()
            if fb.matrix.shape[1] > 0:
                return np.asarray(fb.matrix, dtype=np.float32), "feature_bundle", None
        except Exception:
            pass

        # Backward-compatible path for existing notebooks and direct token use.
        if "scaled_history" in state.historical_features:
            return (
                np.asarray(state.historical_features["scaled_history"], dtype=np.float32),
                "scaled_history",
                None,
            )

        raise KeyError("kernel_rbf requires model_input, features, or scaled_history.")

    def _median_lengthscale(self, X: np.ndarray) -> float:
        if X.shape[0] <= 1:
            return 1.0

        rng = np.random.default_rng(self.seed)
        subset_size = min(self.median_subset, X.shape[0])
        subset_idx = rng.choice(X.shape[0], size=subset_size, replace=False)
        subset = X[subset_idx]

        sqdist = pairwise_distances(subset, metric="sqeuclidean")
        dists = np.sqrt(sqdist[np.triu_indices(subset_size, k=1)])
        dists = dists[np.isfinite(dists) & (dists > 0.0)]
        if dists.size == 0:
            return 1.0
        return float(np.median(dists))

    @staticmethod
    def _flatten_samples(X: np.ndarray) -> np.ndarray:
        if X.ndim == 2:
            return X
        return X.reshape(X.shape[0], -1)
