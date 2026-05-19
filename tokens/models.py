"""Model tokens and STOP token."""

import gc
import numpy as np
from tqdm.auto import tqdm

from ..token import Token


# ── Kernel helpers ──────────────────────────────────────────────

def squared_distance_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise squared L2 distances between rows of a and b."""
    a2 = np.sum(a ** 2, axis=1, keepdims=True)
    b2 = np.sum(b ** 2, axis=1, keepdims=True).T
    return np.maximum(a2 + b2 - 2.0 * (a @ b.T), 0.0)


def median_heuristic_lengthscale(X: np.ndarray, max_pts: int = 24) -> float:
    """Estimate RBF lengthscale via median heuristic."""
    rng = np.random.default_rng(0)
    n = min(max_pts, X.shape[0])
    idx = rng.choice(X.shape[0], size=n, replace=False)
    sub = X[idx]
    d = np.sqrt(squared_distance_matrix(sub, sub))
    tri = d[np.triu_indices(n, k=1)]
    return float(np.median(tri[tri > 0])) if np.any(tri > 0) else 1.0


# ── Evaluation metric ──────────────────────────────────────────

def compute_mase(actual: np.ndarray, forecast: np.ndarray,
                 history: np.ndarray) -> float:
    """MASE: per-sample MAE / naive-baseline scale, then averaged."""
    mae = np.mean(np.abs(actual - forecast), axis=1)
    scale = np.mean(np.abs(np.diff(history, axis=1)), axis=1)
    scale = np.where(scale < 1e-8, 1.0, scale)
    return float(np.mean(mae / scale))


# ── Model tokens ───────────────────────────────────────────────

class ModelKernelRBF(Token):
    name = "kernel_rbf"
    token_class = "model"
    reads = ["model_input"]
    writes = []
    description = "RBF kernel regression — analytic LOO"

    def apply(self, state):
        X = state.features["model_input"].astype(np.float32)
        Y = state.current_target.astype(np.float32)
        n = X.shape[0]

        ls = median_heuristic_lengthscale(X)
        G = np.exp(-squared_distance_matrix(X, X) / (2.0 * ls ** 2))
        A = G + 1e-2 * np.eye(n, dtype=np.float32)
        A_inv = np.linalg.inv(A)
        alpha = A_inv @ Y
        diag = np.diag(A_inv).reshape(-1, 1)
        Y_loo = Y - alpha / diag

        state.push_prediction(Y_loo, self.name)
        state.log_step(self.name,
                       {"model_input": X.shape, "current_target": Y.shape},
                       {"prediction_stack[-1]": Y_loo.shape,
                        "current_target": state.current_target.shape})
        state.token_sequence.append(self.name)
        del A, A_inv, alpha, G
        gc.collect()
        return state


class ModelRandomForest(Token):
    name = "random_forest"
    token_class = "model"
    reads = ["model_input"]
    writes = []
    description = "Random forest — explicit LOO"

    def apply(self, state):
        from sklearn.ensemble import RandomForestRegressor

        X = state.features["model_input"].astype(np.float32)
        Y = state.current_target.astype(np.float32)
        n = X.shape[0]
        Y_loo = np.zeros_like(Y)

        for i in tqdm(range(n), desc="  RF LOO", leave=False):
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            rf = RandomForestRegressor(n_estimators=30, max_depth=6,
                                       random_state=0, n_jobs=1)
            rf.fit(X[mask], Y[mask])
            Y_loo[i] = rf.predict(X[i:i + 1])[0]
            del rf

        state.push_prediction(Y_loo, self.name)
        state.log_step(self.name,
                       {"model_input": X.shape, "current_target": Y.shape},
                       {"prediction_stack[-1]": Y_loo.shape,
                        "current_target": state.current_target.shape})
        state.token_sequence.append(self.name)
        gc.collect()
        return state


class ModelXGBoost(Token):
    name = "xgboost"
    token_class = "model"
    reads = ["model_input"]
    writes = []
    description = "XGBoost — explicit LOO"

    def apply(self, state):
        import xgboost as xgb

        X = state.features["model_input"].astype(np.float32)
        Y = state.current_target.astype(np.float32)
        n = X.shape[0]
        Y_loo = np.zeros_like(Y)

        for i in tqdm(range(n), desc="  XGB LOO", leave=False):
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            m = xgb.XGBRegressor(
                n_estimators=50, max_depth=4, learning_rate=0.1,
                tree_method="hist", verbosity=0, random_state=0)
            m.fit(X[mask], Y[mask])
            Y_loo[i] = m.predict(X[i:i + 1])[0]
            del m

        state.push_prediction(Y_loo, self.name)
        state.log_step(self.name,
                       {"model_input": X.shape, "current_target": Y.shape},
                       {"prediction_stack[-1]": Y_loo.shape,
                        "current_target": state.current_target.shape})
        state.token_sequence.append(self.name)
        gc.collect()
        return state


# ── STOP token ──────────────────────────────────────────────────

class StopToken(Token):
    name = "STOP"
    token_class = "control"
    reads = []
    writes = ["final_forecast"]
    description = "End sequence: compute evaluation"

    def apply(self, state):
        forecast = state.cumulative_prediction()

        # Clip extremes
        h = state.original_history
        lo = h.min(axis=1, keepdims=True)
        hi = h.max(axis=1, keepdims=True)
        rng = np.where(hi > lo, hi - lo, 1.0)
        forecast = np.clip(forecast, lo - 2 * rng, hi + 2 * rng)

        state.features["final_forecast"] = forecast
        state.mase = compute_mase(
            state.original_future, forecast, state.original_history)

        state.log_step("STOP",
                       reads={"prediction_stack": (len(state.prediction_stack),)},
                       writes={"final_forecast": forecast.shape,
                               "mase": (state.mase,)})
        state.token_sequence.append("STOP")
        state.terminated = True
        return state
