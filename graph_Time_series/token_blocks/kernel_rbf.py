import numpy as np
from typing import TYPE_CHECKING, Any

from ..token import ModelToken

if TYPE_CHECKING:
    from ..state import State


class KernelRBFToken(ModelToken):
    name = "kernel_rbf"
    reads = ("scaled_history",)
    writes = ("prediction_stack",)
    description = "RBF kernel ridge regression with explicit leave-one-out predictions."

    def __init__(self, gamma: float | None = None, ridge: float = 1e-3):
        self.gamma = gamma
        self.ridge = ridge

    def get_model(self) -> Any:
        return {"kernel": "rbf", "gamma": self.gamma, "ridge": self.ridge}

    def apply(self, state: 'State') -> 'State':
        X = np.asarray(state.historical_features["scaled_history"], dtype=np.float32)
        X = self._flatten_samples(X)
        Y = np.asarray(state.current_target, dtype=np.float32)

        n_samples = X.shape[0]
        predictions = np.zeros_like(Y)
        gamma = self.gamma if self.gamma is not None else self._median_gamma(X)

        for i in range(n_samples):
            train_mask = np.ones(n_samples, dtype=bool)
            train_mask[i] = False

            if not np.any(train_mask):
                continue

            X_train = X[train_mask]
            Y_train = Y[train_mask]

            K_train = self._rbf_kernel(X_train, X_train, gamma)
            K_train += self.ridge * np.eye(K_train.shape[0], dtype=K_train.dtype)
            alpha = np.linalg.solve(K_train, Y_train)

            K_test = self._rbf_kernel(X[i:i + 1], X_train, gamma)
            predictions[i] = K_test @ alpha

        state.push_prediction(predictions, self.name)
        self._log_execution(
            state,
            reads={
                "scaled_history": X.shape,
                "current_target": Y.shape,
            },
            writes={"prediction_stack[-1]": predictions.shape},
        )
        return state

    @staticmethod
    def _flatten_samples(X: np.ndarray) -> np.ndarray:
        if X.ndim == 2:
            return X
        return X.reshape(X.shape[0], -1)

    @staticmethod
    def _median_gamma(X: np.ndarray) -> float:
        if X.shape[0] < 2:
            return 1.0
        diffs = X[:, None, :] - X[None, :, :]
        sq_dists = np.sum(diffs * diffs, axis=-1)
        positive = sq_dists[sq_dists > 0]
        if positive.size == 0:
            return 1.0
        return 1.0 / (2.0 * float(np.median(positive)))

    @staticmethod
    def _rbf_kernel(A: np.ndarray, B: np.ndarray, gamma: float) -> np.ndarray:
        sq_norms = (
            np.sum(A * A, axis=1, keepdims=True)
            + np.sum(B * B, axis=1)[None, :]
            - 2.0 * (A @ B.T)
        )
        return np.exp(-gamma * np.maximum(sq_norms, 0.0)).astype(np.float32)
