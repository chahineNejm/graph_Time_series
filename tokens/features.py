"""Feature extraction tokens."""

import numpy as np
from ..token import Token, _shapes


class FeatRaw(Token):
    name = "feat_raw"
    token_class = "feature"
    reads = ["cleaned"]
    writes = ["model_input"]
    description = "Use cleaned histories as model input"

    def apply(self, state):
        state.features["model_input"] = state.features["cleaned"].copy()
        state.log_step(self.name,
                       _shapes(state, ["cleaned"]),
                       {"model_input": state.features["model_input"].shape})
        state.token_sequence.append(self.name)
        return state


class FeatFFTEncode(Token):
    """Spectral feature extractor - top-K FFT magnitudes + dominant frequencies."""
    name = "fft_encode"
    token_class = "feature"
    reads = ["cleaned"]
    writes = ["model_input"]
    description = "Spectral features: top-K FFT magnitudes and dominant frequency indices"

    def apply(self, state):
        X = state.features["cleaned"]
        n, d = X.shape
        F = np.fft.rfft(X, axis=1)
        magnitudes = np.abs(F).astype(np.float32)

        # Keep top-K frequency components as compact features
        K = min(20, magnitudes.shape[1])
        top_k_idx = np.argsort(magnitudes, axis=1)[:, -K:]

        # Build feature vector: [top-K magnitudes, top-K freq indices (normalized)]
        top_mags = np.take_along_axis(magnitudes, top_k_idx, axis=1)
        top_freqs = top_k_idx.astype(np.float32) / magnitudes.shape[1]

        # Also include basic spectral stats
        spectral_energy = np.sum(magnitudes ** 2, axis=1, keepdims=True)
        spectral_entropy = np.zeros((n, 1), dtype=np.float32)
        for i in range(n):
            p = magnitudes[i] / (magnitudes[i].sum() + 1e-8)
            spectral_entropy[i] = -np.sum(p * np.log(p + 1e-10))

        model_input = np.hstack([top_mags, top_freqs,
                                 spectral_energy, spectral_entropy])
        state.features["model_input"] = model_input
        state.log_step(self.name,
                       _shapes(state, ["cleaned"]),
                       {"model_input": model_input.shape})
        state.token_sequence.append(self.name)
        return state


class FeatLagFeatures(Token):
    """Lag-based features - autocorrelation structure as model input."""
    name = "feat_lag"
    token_class = "feature"
    reads = ["cleaned"]
    writes = ["model_input"]
    description = "Lag features: last L values + autocorrelation coefficients"

    def apply(self, state):
        X = state.features["cleaned"]
        n, d = X.shape
        horizon = state.horizon

        # Last L values (most recent history - strongest signal)
        L = min(3 * horizon, d)
        recent = X[:, -L:]

        # Autocorrelation at key lags
        lags = [1, 2, 3, 5, 7, 10, 14, 21]
        lags = [l for l in lags if l < d]
        acf = np.zeros((n, len(lags)), dtype=np.float32)
        for j, lag in enumerate(lags):
            x1, x2 = X[:, lag:], X[:, :-lag]
            minlen = min(x1.shape[1], x2.shape[1])
            x1, x2 = x1[:, :minlen], x2[:, :minlen]
            std1 = x1.std(axis=1, keepdims=True)
            std2 = x2.std(axis=1, keepdims=True)
            std1 = np.where(std1 < 1e-8, 1.0, std1)
            std2 = np.where(std2 < 1e-8, 1.0, std2)
            acf[:, j:j+1] = np.mean(
                (x1 - x1.mean(axis=1, keepdims=True)) *
                (x2 - x2.mean(axis=1, keepdims=True)),
                axis=1, keepdims=True
            ) / (std1 * std2)

        model_input = np.hstack([recent, acf])
        state.features["model_input"] = model_input.astype(np.float32)
        state.log_step(self.name,
                       _shapes(state, ["cleaned"]),
                       {"model_input": model_input.shape})
        state.token_sequence.append(self.name)
        return state
