"""Period detection + seasonal sinusoid features.

Two cooperating tokens:

* ``PeriodDetect`` (precursor) finds the dominant period(s) in the history via
  the power spectrum and publishes them as a ``param:periods`` Signal.
* ``SeasonalFeatures`` requires that output and builds sin/cos features (with
  harmonics) for each detected period, on both the history and the forecast
  horizon, so later models get explicit periodicity.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..token import FeatureToken
from ..signal import Port

if TYPE_CHECKING:
    from ..state import State


class PeriodDetectToken(FeatureToken):
    """Detect periods using FLAIR's selection logic, published as param:periods.

    Mirrors ``PeriodSelectionToken``: candidate periods come from a fixed
    frequency table (``FREQ_PERIODS``), each is scored with FLAIR's rank-1
    SVD/BIC fold score (``_period_bic``), the min-BIC candidate is the primary
    period, and its integer multiples (that still have enough complete cycles)
    are the secondary periods. ``periods = [primary] + secondary``.
    """

    name = "PeriodDetect"
    token_class = "feature"
    reads = ("raw_history",)
    writes = ("periods",)
    description = "FLAIR-style BIC period selection (freq candidates) -> param:periods."
    max_uses = 1
    provides = (Port(sem="param:periods", alignment="static", space="any"),)

    def __init__(self, freq: str = "H", max_series: int = 8,
                 min_complete: int = 3, source_feature: str = "scaled_history"):
        self.freq = freq
        self.max_series = int(max_series)
        self.min_complete = int(min_complete)
        self.source_feature = source_feature

    def check_specific_conditions(self, state: "State") -> bool:
        return "periods" not in state.metadata

    def _history(self, state: "State") -> np.ndarray:
        # Prefer a FLAIR-cleaned history if present, else the configured source.
        for name in ("flair_history", self.source_feature):
            for store in ("historical_features", "features"):
                d = getattr(state, store)
                if name in d:
                    return np.asarray(d[name], dtype=np.float64)
        return np.asarray(state.features["raw_history"], dtype=np.float64)

    def apply(self, state: "State") -> "State":
        from .flair import FREQ_PERIODS, _period_bic

        hist = self._history(state)
        n_steps = hist.shape[1]

        candidates = [
            p for p in FREQ_PERIODS.get(self.freq.upper(), [1])
            if p > 0 and n_steps // p >= self.min_complete
        ]
        candidates = sorted(set([1] + candidates))
        subset = hist[: min(self.max_series, hist.shape[0])]

        scores = {int(p): float(np.mean([_period_bic(row, p) for row in subset]))
                  for p in candidates}
        primary = min(scores, key=scores.get)
        secondary = [int(p) for p in candidates
                     if p > primary and primary > 1 and p % primary == 0]
        periods = [int(primary)] + secondary

        state.put_param("periods", periods, source_token=self.name)
        state.metadata["period"] = int(primary)              # back-compat single period
        state.metadata["periods"] = periods
        state.metadata["period_scores"] = scores
        state.metadata["flair_secondary_periods"] = secondary
        state.flags["periods_detected"] = True
        self._log_execution(
            state,
            reads={"history": hist.shape},
            writes={"periods": periods, "period_scores": scores},
        )
        return state


class SeasonalFeaturesToken(FeatureToken):
    """Build sin/cos features (with harmonics) for each detected period."""

    name = "SeasonalFeatures"
    token_class = "feature"
    reads = ("raw_history",)
    writes = ("seasonal_features", "future_seasonal_features")
    description = ("Per-detected-period sin/cos channels (history + known future, no_bundle) "
                   "plus a bundle-safe per-series projection (Fourier amplitudes at h/p + seasonal_strength).")
    max_uses = 1
    requires = (Port(sem="param:periods", alignment="static", space="any"),)
    provides = (
        Port(sem="features", alignment="history", space="any"),
        Port(sem="features", alignment="future", space="any"),
    )

    def __init__(self, n_harmonics: int = 2):
        if n_harmonics < 1:
            raise ValueError("n_harmonics must be >= 1.")
        self.n_harmonics = int(n_harmonics)

    def check_specific_conditions(self, state: "State") -> bool:
        if state.flags.get("seasonal_features_built", False):
            return False
        return "periods" in state.metadata

    def _channel_names(self, periods):
        names = []
        for p in periods:
            for h in range(1, self.n_harmonics + 1):
                names.append(f"sin_p{p}_h{h}")
                names.append(f"cos_p{p}_h{h}")
        return names

    def _history_series(self, state: "State") -> np.ndarray:
        for name in ("scaled_history",):
            if name in state.historical_features:
                return np.asarray(state.historical_features[name], dtype=np.float64)
        return np.asarray(state.features["raw_history"], dtype=np.float64)

    def _tabular_projection(self, series: np.ndarray, periods):
        """Per-series spectral summary at the DETECTED periods (bundle-safe).

        For each (period, harmonic) it projects the centered series onto
        sin/cos at that frequency and returns the amplitude sqrt(a^2 + b^2);
        also a 'seasonal_strength' = seasonal power / total variance. These
        vary per series/window, so tree models can actually use them. Distinct
        from FourierFeatures (fixed low-freq bins) -- these sit at the detected
        periods only.
        """
        n, T = series.shape
        x = series - series.mean(axis=1, keepdims=True)
        t = np.arange(T)
        cols, names = [], []
        seasonal_power = np.zeros(n, dtype=np.float64)
        for p in periods:
            for h in range(1, self.n_harmonics + 1):
                ang = 2.0 * np.pi * h * t / float(p)
                a = (2.0 / T) * (x * np.cos(ang)).sum(axis=1)
                b = (2.0 / T) * (x * np.sin(ang)).sum(axis=1)
                amp = np.sqrt(a * a + b * b)
                cols.append(amp)
                names.append(f"amp_p{p}_h{h}")
                seasonal_power += a * a + b * b
        total_var = x.var(axis=1) + 1e-12
        strength = np.clip(0.5 * seasonal_power / total_var, 0.0, 1.0)
        cols.append(strength)
        names.append("seasonal_strength")
        proj = np.stack(cols, axis=1).astype(np.float32)         # (n, K)
        return proj, names

    def _encode(self, positions: np.ndarray, periods, n_samples: int) -> np.ndarray:
        """Return a structured (n_samples, len, n_channels) tensor (not flat)."""
        chans = []
        for p in periods:
            for h in range(1, self.n_harmonics + 1):
                angle = 2.0 * np.pi * h * positions / float(p)
                chans.append(np.sin(angle))
                chans.append(np.cos(angle))
        block = np.stack(chans, axis=1).astype(np.float32)           # (len, n_channels)
        return np.broadcast_to(
            block[None], (n_samples,) + block.shape
        ).astype(np.float32)                                         # (n, len, n_channels)

    def apply(self, state: "State") -> "State":
        periods = [int(p) for p in state.metadata["periods"]]
        n = state.n_samples
        T = state.original_history.shape[1]
        H = state.horizon
        n_channels = 2 * self.n_harmonics * len(periods)

        # Structured channel tensors: (sample, time/horizon, channel). Kept on
        # their own channel axis -- NOT flattened -- and tagged ``no_bundle`` so
        # they do not blow up the tabular feature bundle. Sequence/per-step
        # models consume them via their (sample, time, feature) port.
        hist_feat = self._encode(np.arange(T), periods, n)
        fut_feat = self._encode(np.arange(T, T + H), periods, n)

        state.add_historical_feature("seasonal_features", hist_feat)
        state.register_artifact(
            "seasonal_features", store="historical_features",
            kind="seasonal_history", role="feature", source_token=self.name,
            tags=("seasonal", "sinusoid", "channels", "no_bundle"),
        )
        state.add_future_feature("future_seasonal_features", fut_feat)
        state.register_artifact(
            "future_seasonal_features", store="future_features",
            kind="seasonal_future", role="known_future_feature",
            source_token=self.name,
            tags=("seasonal", "sinusoid", "channels", "no_bundle"),
        )

        # Bundle-safe per-series projection so tabular models (rf/lightgbm/
        # kernel) actually consume seasonal information: amplitude at each
        # detected period/harmonic + overall seasonal strength.
        proj, proj_names = self._tabular_projection(self._history_series(state), periods)
        state.add_historical_feature("seasonal_projection", proj)
        state.register_artifact(
            "seasonal_projection", store="historical_features", kind="tabular",
            role="feature", source_token=self.name, tags=("seasonal", "projection"),
        )

        state.metadata["seasonal_features"] = {
            "periods": periods,
            "n_harmonics": self.n_harmonics,
            "channels_per_period": 2 * self.n_harmonics,
            "n_channels": n_channels,
            "channel_names": self._channel_names(periods),
            "history_shape": tuple(hist_feat.shape),
            "future_shape": tuple(fut_feat.shape),
            "projection_names": proj_names,
            "projection_shape": tuple(proj.shape),
        }
        state.flags["seasonal_features_built"] = True
        self._log_execution(
            state,
            reads={"periods": periods},
            writes={"seasonal_features": hist_feat.shape,
                    "future_seasonal_features": fut_feat.shape,
                    "seasonal_projection": proj.shape},
        )
        return state
