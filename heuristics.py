"""MI-based heuristic for PUCT priors."""

import numpy as np
from sklearn.feature_selection import mutual_info_regression

from .state import State


def compute_mi_score(state: State,
                     n_feat_cols: int = 10,
                     n_target_cols: int = 3) -> float:
    """
    Estimate mutual information between model_input and current_target.
    Uses subsampled columns for speed. Returns scalar MI score.
    """
    if "model_input" not in state.features:
        return 0.0

    X = state.features["model_input"]
    Y = state.current_target

    if X.shape[0] < 5:
        return 0.0

    feat_idx = np.linspace(0, X.shape[1] - 1,
                           min(n_feat_cols, X.shape[1]), dtype=int)
    tgt_idx = np.linspace(0, Y.shape[1] - 1,
                          min(n_target_cols, Y.shape[1]), dtype=int)
    X_sub = X[:, feat_idx]

    mi_scores = []
    for ti in tgt_idx:
        y_col = Y[:, ti]
        if np.std(y_col) < 1e-8:
            continue
        try:
            mi = mutual_info_regression(X_sub, y_col,
                                        random_state=0, n_neighbors=3)
            mi_scores.append(np.mean(mi))
        except Exception:
            continue

    return float(np.mean(mi_scores)) if mi_scores else 0.0


def compute_action_priors(mi_score: float,
                          actions: list[str]) -> dict[str, float]:
    """
    Convert MI score into per-action priors for PUCT (sum to 1).
    Override this to inject domain-specific heuristics.
    """
    n = len(actions)
    if n == 0:
        return {}

    priors = {a: 1.0 / n for a in actions}

    if mi_score > 0.5:
        boosts = {"kernel_rbf": 1.5, "xgboost": 1.2,
                  "random_forest": 0.8, "STOP": 0.5}
    elif mi_score > 0.1:
        boosts = {"kernel_rbf": 1.0, "xgboost": 1.3,
                  "random_forest": 1.0, "STOP": 0.8}
    else:
        boosts = {"STOP": 1.2}

    for a in actions:
        priors[a] *= boosts.get(a, 1.0)

    total = sum(priors.values())
    return {a: v / total for a, v in priors.items()}
