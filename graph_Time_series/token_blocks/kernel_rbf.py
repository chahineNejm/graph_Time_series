import numpy as np
from typing import TYPE_CHECKING, Any

from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import pairwise_distances

from ..token import ModelToken

if TYPE_CHECKING:
    from ..state import State


class KernelRBFToken(ModelToken):
    name = "kernel_rbf"
    reads = ("scaled_history",)
    writes = ("prediction_stack",)
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
        X = np.asarray(state.historical_features["scaled_history"], dtype=np.float32)
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
        self._log_execution(
            state,
            reads={
                "scaled_history": X.shape,
                "current_target": Y.shape,
            },
            writes={
                "prediction_stack[-1]": predictions.shape,
                "lengthscale": lengthscale,
                "gamma": gamma,
            },
        )
        return state

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
