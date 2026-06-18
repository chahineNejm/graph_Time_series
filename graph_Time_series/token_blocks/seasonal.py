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


class _PeriodDetectBase(FeatureToken):
    """Shared machinery for period-chain detectors (PeriodDetect / PeriodDetectBIC).

    Scores candidate periods from a fixed frequency table, selects the distinct
    seasonal scales, and reduces them to an ascending divisibility chain -- the
    queue that ``SeasonalFold`` consumes one fold at a time. Subclasses differ
    only in the scoring/selection (``_select``): periodogram power vs rank-1 BIC.
    Both guard on ``"periods" not in metadata``, so they are mutually exclusive:
    the grammar can offer both and only the first to run takes effect.
    """

    token_class = "feature"
    reads = ("raw_history",)
    writes = ("periods",)
    max_uses = 1
    provides = (Port(sem="param:periods", alignment="static", space="any"),)
    freq = "H"
    max_series = 8
    min_complete = 3
    source_feature = "scaled_history"
    max_periods = 4

    def check_specific_conditions(self, state: "State") -> bool:
        return "periods" not in state.metadata

    def _history(self, state: "State") -> np.ndarray:
        for name in ("clean_history", "flair_history", self.source_feature):
            for store in ("historical_features", "features"):
                d = getattr(state, store)
                if name in d:
                    return np.asarray(d[name], dtype=np.float64)
        return np.asarray(state.features["raw_history"], dtype=np.float64)

    def _select(self, cands, subset):
        """Return (dominant_period, strong_periods, diag_metadata)."""
        raise NotImplementedError

    def apply(self, state: "State") -> "State":
        from .flair import FREQ_PERIODS

        hist = self._history(state)
        n_steps = hist.shape[1]
        subset = hist[: min(self.max_series, hist.shape[0])]
        cal = sorted(set(FREQ_PERIODS.get(self.freq.upper(), [])))
        cands = [p for p in cal if p > 1 and n_steps // p >= self.min_complete]
        if not cands:
            dominant, periods, diag = 0, [], {}
        else:
            dominant, significant, diag = self._select(cands, subset)
            periods = self._flair_periods(dominant, significant, cal, n_steps)

        state.put_param("periods", periods, source_token=self.name)
        state.metadata["period"] = int(dominant)            # primary period (full shape)
        state.metadata["periods"] = periods                 # [primary] + coarser secondaries
        state.metadata.update(diag)
        state.flags["periods_detected"] = True
        self._log_execution(
            state,
            reads={"history": hist.shape},
            writes={"periods": periods, "dominant": dominant},
        )
        return state

    def _flair_periods(self, primary, significant, cal, n_steps):
        """FLAIR ordering: primary first, then COARSER multiples of it that the
        detector found significant (each modulates the Level via Shape_k). The
        harmonic gate in SeasonalFold neutralises a spurious secondary, so this
        stays safe even if an extra period slips through."""
        P = int(primary)
        nc = n_steps // P if P >= 1 else 0
        # Secondaries = COARSER exact multiples of the primary with enough Level
        # cycles. We do NOT filter them by the raw-series score (a coarse period
        # like an annual cycle is undetectable on the raw fold -- too few cycles).
        # SeasonalFold's harmonic gate decides per-secondary ON THE LEVEL, so a
        # multiple with no real Level pattern produces a flat (no-op) Shape_k.
        secondary = sorted(
            c for c in cal
            if c > P and P >= 1 and c % P == 0 and nc // (c // P) >= 2
        )
        # No count cap: the pool of exact coarser multiples is naturally small,
        # and SeasonalFold picks adaptively from it (only real Level patterns
        # fold; spurious multiples are skipped), so 'max_periods' would only risk
        # dropping the meaningful coarse period (e.g. annual).
        return [P] + secondary

    @staticmethod
    def _extrema(cands, scores, maximize, pad, keep):
        """Local extrema of the score curve (baseline-padded ends) passing ``keep``."""
        cands = sorted(cands)
        out = []
        for i, p in enumerate(cands):
            left = scores[cands[i - 1]] if i > 0 else pad
            right = scores[cands[i + 1]] if i + 1 < len(cands) else pad
            is_ext = (scores[p] >= left and scores[p] >= right) if maximize \
                else (scores[p] <= left and scores[p] <= right)
            if is_ext and keep(p):
                out.append(int(p))
        return out

class PeriodDetectToken(_PeriodDetectBase):
    """FLAIR-faithful primary + coarser secondaries via rank-1 SVD/BIC.

    BIC on the SVD spectrum picks the primary period P -- the one whose fold is
    most rank-1, typically the LARGE period that captures the full repeating
    unit (e.g. a whole week), giving a rich full Shape. Publishes
    ``periods = [P] + [coarser multiples]`` that beat the no-period baseline by
    ``margin``. ``SeasonalFold`` folds the primary fully (full Shape over P),
    then peels each coarser period off the Level as Shape2, Shape3, ... This is
    the default detector and matches the FLAIR paper's MDL period selection.
    """

    name = "PeriodDetect"
    description = "FLAIR BIC: primary (rank-1) + coarser secondaries -> param:periods."

    def __init__(self, freq: str = "H", max_series: int = 8, min_complete: int = 3,
                 source_feature: str = "scaled_history", margin: float = 0.15,
                 max_periods: int = 4):
        self.freq = freq
        self.max_series = int(max_series)
        self.min_complete = int(min_complete)
        self.source_feature = source_feature
        self.margin = float(margin)              # required improvement, frac of |baseline|
        self.max_periods = int(max_periods)

    def _select(self, cands, subset):
        from .flair import _period_bic
        base = float(np.mean([_period_bic(row, 1) for row in subset]))
        scores = {int(p): float(np.mean([_period_bic(row, p) for row in subset])) for p in cands}
        dominant = int(min(scores, key=scores.get))
        thresh = self.margin * abs(base)
        significant = self._extrema(cands, scores, maximize=False, pad=base,
                                    keep=lambda p: (base - scores[p]) > thresh)
        return dominant, significant, {"period_scores": {**scores, 1: base},
                                       "period_baseline_bic": base, "period_selector": "bic"}


class PeriodDetectBICToken(PeriodDetectToken):
    """Explicit-name BIC/FLAIR detector. Identical to ``PeriodDetect`` (which is
    already the BIC selector); kept as a distinct name for back-compatibility and
    for sequences that want to name the selector explicitly."""

    name = "PeriodDetectBIC"
    description = "FLAIR BIC primary + coarser secondaries (explicit name) -> param:periods."


class PeriodDetectSpectralToken(_PeriodDetectBase):
    """Periodogram alternative: primary = fundamental, + coarser secondaries.

    Each candidate ``p`` is scored by the fraction of spectral power at its
    fundamental frequency ``1/p``. The primary is the period with the most
    power (the genuine fundamental, since a multiple ``k*p`` is a subharmonic
    with ~no power); coarser multiples above ``power_threshold`` become the
    secondary Level-modulation periods. NOTE: for FLAIR-style forecasting the
    BIC detector (``PeriodDetect``) is usually preferable -- it picks the large
    rank-1 period for a richer full Shape, whereas the fundamental can be small.
    """

    name = "PeriodDetectSpectral"
    description = "Periodogram: fundamental primary + coarser secondaries -> param:periods."

    def __init__(self, freq: str = "H", max_series: int = 8, min_complete: int = 3,
                 source_feature: str = "scaled_history", power_threshold: float = 0.05,
                 band: int = 1, max_periods: int = 4):
        self.freq = freq
        self.max_series = int(max_series)
        self.min_complete = int(min_complete)
        self.source_feature = source_feature
        self.power_threshold = float(power_threshold)
        self.band = int(band)
        self.max_periods = int(max_periods)

    def _select(self, cands, subset):
        from .flair import _period_power
        scores = {int(p): float(np.mean([_period_power(row, p, self.band) for row in subset]))
                  for p in cands}
        dominant = int(max(scores, key=scores.get))
        significant = [int(p) for p in cands if scores[p] > self.power_threshold]
        return dominant, significant, {"period_power": scores, "period_selector": "periodogram"}


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
