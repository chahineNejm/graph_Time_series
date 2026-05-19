"""Cleaning tokens - operate on raw_history, produce cleaned."""

import numpy as np
from ..token import Token, _shapes


class CleanIdentity(Token):
    name = "identity"
    token_class = "cleaning"
    reads = ["raw_history"]
    writes = ["cleaned"]
    description = "Pass-through"

    def apply(self, state):
        state.features["cleaned"] = state.features["raw_history"].copy()
        state.log_step(self.name,
                       _shapes(state, ["raw_history"]),
                       {"cleaned": state.features["cleaned"].shape})
        state.token_sequence.append(self.name)
        return state


class CleanDetrend(Token):
    name = "detrend"
    token_class = "cleaning"
    reads = ["raw_history"]
    writes = ["cleaned", "trend_slope", "trend_intercept"]
    description = "Remove per-sample linear trend"

    def apply(self, state):
        X = state.features["raw_history"]
        n, d = X.shape
        t = np.arange(d, dtype=np.float32)
        t_mean = t.mean()

        slopes = np.zeros(n, dtype=np.float32)
        intercepts = np.zeros(n, dtype=np.float32)
        detrended = np.zeros_like(X)

        for i in range(n):
            h_mean = X[i].mean()
            slope = (np.sum((t - t_mean) * (X[i] - h_mean)) /
                     (np.sum((t - t_mean) ** 2) + 1e-8))
            intercepts[i] = h_mean - slope * t_mean
            slopes[i] = slope
            detrended[i] = X[i] - (slope * t + intercepts[i])

        state.features["cleaned"] = detrended
        state.features["trend_slope"] = slopes
        state.features["trend_intercept"] = intercepts
        state.log_step(self.name,
                       _shapes(state, ["raw_history"]),
                       {k: state.features[k].shape for k in self.writes})
        state.token_sequence.append(self.name)
        return state


class CleanMovingAvg(Token):
    name = "moving_avg"
    token_class = "cleaning"
    reads = ["raw_history"]
    writes = ["cleaned"]
    description = "Centred moving average (window=5)"

    def apply(self, state):
        X = state.features["raw_history"]
        kernel = np.ones(5) / 5.0
        smoothed = np.zeros_like(X)
        for i in range(X.shape[0]):
            smoothed[i] = np.convolve(X[i], kernel, mode="same")
        state.features["cleaned"] = smoothed
        state.log_step(self.name,
                       _shapes(state, ["raw_history"]),
                       {"cleaned": smoothed.shape})
        state.token_sequence.append(self.name)
        return state


class CleanNormalize(Token):
    """Z-score normalization per sample - matches pipeline.py prepare_examples."""
    name = "normalize"
    token_class = "cleaning"
    reads = ["raw_history"]
    writes = ["cleaned", "norm_mu", "norm_sigma"]
    description = "Per-sample z-score normalization (zero mean, unit variance)"

    def apply(self, state):
        X = state.features["raw_history"]
        mu = X.mean(axis=1, keepdims=True)
        sigma = X.std(axis=1, keepdims=True)
        sigma = np.where(sigma < 1e-8, 1.0, sigma)

        state.features["cleaned"] = (X - mu) / sigma
        state.features["norm_mu"] = mu
        state.features["norm_sigma"] = sigma

        # Also normalize the target so model predictions are on the
        # same scale - this is crucial for good performance.
        state.current_target = (state.original_future - mu) / sigma
        state.metadata["normalized"] = True

        state.log_step(self.name,
                       _shapes(state, ["raw_history"]),
                       {k: state.features[k].shape for k in
                        ["cleaned", "norm_mu", "norm_sigma"]})
        state.token_sequence.append(self.name)
        return state


class CleanDetrendNorm(Token):
    """Detrend + normalize combo - removes linear trend then z-scores."""
    name = "detrend_norm"
    token_class = "cleaning"
    reads = ["raw_history"]
    writes = ["cleaned", "trend_slope", "trend_intercept", "norm_mu", "norm_sigma"]
    description = "Remove linear trend then z-score normalize"

    def apply(self, state):
        X = state.features["raw_history"]
        n, d = X.shape
        t = np.arange(d, dtype=np.float32)
        t_mean = t.mean()

        # Detrend
        detrended = np.zeros_like(X)
        slopes = np.zeros(n, dtype=np.float32)
        intercepts = np.zeros(n, dtype=np.float32)
        for i in range(n):
            h_mean = X[i].mean()
            slope = (np.sum((t - t_mean) * (X[i] - h_mean)) /
                     (np.sum((t - t_mean) ** 2) + 1e-8))
            intercepts[i] = h_mean - slope * t_mean
            slopes[i] = slope
            detrended[i] = X[i] - (slope * t + intercepts[i])

        # Normalize
        mu = detrended.mean(axis=1, keepdims=True)
        sigma = detrended.std(axis=1, keepdims=True)
        sigma = np.where(sigma < 1e-8, 1.0, sigma)

        state.features["cleaned"] = (detrended - mu) / sigma
        state.features["trend_slope"] = slopes
        state.features["trend_intercept"] = intercepts
        state.features["norm_mu"] = mu
        state.features["norm_sigma"] = sigma

        # Normalize the target too
        state.current_target = (state.original_future - mu) / sigma
        state.metadata["normalized"] = True

        state.log_step(self.name,
                       _shapes(state, ["raw_history"]),
                       {k: state.features[k].shape for k in self.writes})
        state.token_sequence.append(self.name)
        return state
