"""Fourier feature tokens."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..token import FeatureToken

if TYPE_CHECKING:
    from ..state import State


class FourierFeaturesToken(FeatureToken):
    """Create fixed-width tabular Fourier features from the history."""

    name = "FourierFeatures"
    reads = ()
    writes = ("fourier_features",)
    description = "Fixed low-frequency FFT features for tabular models."
    max_uses = 1

    def __init__(
        self,
        *,
        n_harmonics: int = 16,
        source_feature: str = "scaled_history",
        fallback_to_raw: bool = True,
        include_summary: bool = True,
        center: bool = True,
    ):
        if n_harmonics <= 0:
            raise ValueError("n_harmonics must be positive.")
        self.n_harmonics = int(n_harmonics)
        self.source_feature = source_feature
        self.fallback_to_raw = fallback_to_raw
        self.include_summary = include_summary
        self.center = center

    def check_specific_conditions(self, state: "State") -> bool:
        if not super().check_specific_conditions(state):
            return False
        if "fourier_features" in state.historical_features:
            return False
        try:
            self._resolve_history(state)
        except KeyError:
            return False
        return True

    def apply(self, state: "State") -> "State":
        hist, input_name = self._resolve_history(state)
        hist = np.asarray(hist, dtype=np.float32)
        if hist.ndim != 2:
            hist = hist.reshape(hist.shape[0], -1)

        features, feature_names, bins = self._features_from_history(hist)
        state.add_historical_feature("fourier_features", features)
        state.register_artifact(
            "fourier_features",
            store="historical_features",
            kind="tabular",
            role="feature",
            source_token=self.name,
            tags=("fourier", "tabular"),
        )
        state.metadata["fourier_features"] = {
            "source": input_name,
            "n_harmonics_requested": self.n_harmonics,
            "bins": tuple(int(b) for b in bins),
            "feature_names": tuple(feature_names),
            "shape": tuple(features.shape),
            "center": self.center,
            "include_summary": self.include_summary,
        }

        self._log_execution(
            state,
            reads={input_name: hist.shape},
            writes={"fourier_features": features.shape},
        )
        return state

    def _resolve_history(self, state: "State") -> tuple[np.ndarray, str]:
        for store_name in ("historical_features", "features"):
            store = getattr(state, store_name)
            if self.source_feature in store:
                return np.asarray(store[self.source_feature]), self.source_feature

        if self.fallback_to_raw and "raw_history" in state.features:
            return np.asarray(state.features["raw_history"]), "raw_history"

        raise KeyError(
            f"{self.name} requires {self.source_feature!r}"
            + (" or raw_history." if self.fallback_to_raw else ".")
        )

    def _features_from_history(
        self, hist: np.ndarray
    ) -> tuple[np.ndarray, list[str], np.ndarray]:
        n_samples, history_length = hist.shape
        if history_length < 3:
            raise ValueError("FourierFeatures requires at least 3 history points.")

        mean = hist.mean(axis=1, keepdims=True)
        std = hist.std(axis=1, keepdims=True) + 1e-8
        working = hist - mean if self.center else hist

        fft = np.fft.rfft(working, axis=1)
        max_bin = min(self.n_harmonics, fft.shape[1] - 1)
        bins = np.arange(1, max_bin + 1)
        coeff = fft[:, bins] / float(history_length)

        real = coeff.real.astype(np.float32)
        imag = coeff.imag.astype(np.float32)
        amplitude = (2.0 * np.abs(coeff)).astype(np.float32)
        total_power = (
            np.sum(np.abs(fft[:, 1:]) ** 2, axis=1, keepdims=True).astype(np.float32)
            + 1e-8
        )
        rel_power = (np.abs(fft[:, bins]) ** 2 / total_power).astype(np.float32)

        blocks = [real, imag, amplitude, rel_power]
        names = []
        for prefix in ("fft_real", "fft_imag", "fft_amp", "fft_rel_power"):
            names.extend(f"{prefix}_{int(bin_idx)}" for bin_idx in bins)

        if self.include_summary:
            last = hist[:, -1:]
            first = hist[:, :1]
            slope = (last - first) / max(history_length - 1, 1)
            summary = np.concatenate(
                [
                    mean,
                    std,
                    hist.min(axis=1, keepdims=True),
                    hist.max(axis=1, keepdims=True),
                    last,
                    slope,
                ],
                axis=1,
            ).astype(np.float32)
            blocks.append(summary)
            names.extend(
                [
                    "history_mean",
                    "history_std",
                    "history_min",
                    "history_max",
                    "history_last",
                    "history_slope",
                ]
            )

        features = np.concatenate(blocks, axis=1).astype(np.float32)
        if features.shape[0] != n_samples:
            raise RuntimeError("Fourier feature construction changed sample count.")
        if not np.all(np.isfinite(features)):
            raise ValueError("Fourier features contain NaN or infinite values.")
        return features, names, bins
