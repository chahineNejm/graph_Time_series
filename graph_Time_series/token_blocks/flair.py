"""Experimental FLAIR-style token family.

The tokens in this module keep the FLAIR decomposition visible in State instead
of hiding it behind one monolithic forecaster. They are intentionally
self-contained and do not depend on private ``flaircast`` internals.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ..token import FeatureToken, ModelToken
from ..signal import Port
from .imputation import LinearFillToken, linear_fill_1d

if TYPE_CHECKING:
    from ..state import State


FREQ_PERIODS = {
    "H": [6, 8, 12, 24, 72, 168, 336, 720],
    "HOURLY": [6, 8, 12, 24, 72, 168, 336, 720],

    "D": [2, 3, 4, 5, 7, 14, 28, 30, 31, 90, 182, 364],
    "DAILY": [2, 3, 4, 5, 7, 14, 28, 30, 31, 90, 182, 364],

    "W": [4, 13, 26, 52],
    "WEEKLY": [4, 13, 26, 52],

    "M": [3,4, 6, 12],
    "MONTHLY": [3, 6, 12],

    "15T": [4, 8, 16, 48, 96, 192, 672],
    "5T": [12, 36, 72, 144, 288, 576, 2016],
}
_EPS = 1e-8


class FlairPreprocessToken(LinearFillToken):
    """Backward-compatible alias for generic linear filling."""

    name = "FlairPreprocess"
    description = "Compatibility alias for LinearFill; no positivity shift."


class SeasonalFoldToken(FeatureToken):
    """FLAIR-faithful seasonal decomposition, extended past FLAIR's single Shape2.

    Call once for a plain FLAIR fold; call again per coarser period for
    ``Shape3, Shape4, ...`` -- as many as there are secondary periods.

    * call 0 (primary fold): fold the active clean/raw history by the primary period ``P``
      (``periods[0]``, the BIC rank-1 period) into a FULL within-period shape
      (``P`` proportions, summing to 1) plus a per-P-cycle ``level`` (totals).
      The full shape captures all structure inside one P-cycle -- it is NOT a
      separable product, so nothing is approximated by step functions.
    * call k >= 1 (Shape_{k+1}): estimate a secondary periodic pattern in the
      (already-deseasonalized) LEVEL at ``cp = periods[k] // P`` -- FLAIR's
      ``_compute_shape2``: per-position mean, a BIC-gated first-harmonic-vs-flat
      prior, and empirical-Bayes shrinkage ``w = nc2/(nc2+cp)``. Divide it out of
      the level and accumulate it into the forecast-side level correction
      ``flair_level_future_shape2``. The primary period and full shape are
      unchanged.

    Reconstruction (in the model heads):
        y_hat[t] = Level_reseasonalized[t // P] * Shape[t % P]
    where ``Level_reseasonalized`` re-applies every Shape_k. The Ridge forecasts
    only the smooth, fully-deseasonalized level (one value per P-cycle).

    ``periods = [P_primary] + [coarser secondaries]`` comes from PeriodDetect.
    Optional rank-1 SVD level shrinkage (``shrink_level``) denoises the level.
    """

    name = "SeasonalFold"
    token_class = "feature"
    reads = ("raw_history",)
    writes = ("period_matrix", "level_series", "shape_vector")
    description = (
        "FLAIR fold: call 0 = full shape over the primary period + level; each "
        "later call = one more level modulation (Shape2, Shape3, ...) at a "
        "coarser period. Replaces PeriodFold/ShapeLevel/SecondaryLevelSeasonality."
    )
    max_uses = 6
    requires = (Port(sem="param:periods", alignment="static", space="any"),)

    def __init__(self, shrink_level: bool = True, min_factor: float = 0.05,
                 shape_k: int = 2, min_complete: int = 2, harmonic_prior: bool = True,
                 secondary_min_strength: float = 0.05):
        self.shrink_level = bool(shrink_level)
        self.min_factor = float(min_factor)
        self.shape_k = int(shape_k)
        self.min_complete = int(min_complete)
        self.harmonic_prior = bool(harmonic_prior)
        self.secondary_min_strength = float(secondary_min_strength)

    def _depth(self, state: "State") -> int:
        return state.token_sequence.count(self.name)

    def _periods(self, state: "State") -> list:
        periods = state.metadata.get("periods")
        if not periods:
            p = int(state.metadata.get("period", 0))
            periods = [p] if p > 0 else []
        return [int(p) for p in periods]

    def check_specific_conditions(self, state: "State") -> bool:
        if not super().check_specific_conditions(state):
            return False
        periods = self._periods(state)
        depth = self._depth(state)
        if depth >= len(periods):
            return False
        if depth == 0:
            hist = _flair_history(state)
            return periods[0] > 0 and hist.shape[1] // periods[0] >= self.min_complete
        if "flair_level_work" not in state.historical_features:
            return False
        # A further fold applies only if some UNUSED coarser period shows a real
        # (harmonic-gated) modulation of the current Level above the strength
        # floor. Spurious multiples (flat Shape) are skipped, so no no-op folds.
        return self._best_secondary(state) is not None

    def apply(self, state: "State") -> "State":
        depth = self._depth(state)
        periods = self._periods(state)
        if depth == 0:
            return self._fold_primary(state, int(periods[0]))
        P = int(periods[0])
        sel = self._best_secondary(state)
        if sel is None:                    # guarded by can_apply, defensive only
            return state
        cp = sel[1]
        return self._shape_k(state, P, cp, depth + 1)

    def _best_secondary(self, state: "State"):
        """Among UNUSED coarser candidate periods, return (period, cp) whose
        harmonic-gated Level modulation is strongest, or None if none clears
        ``secondary_min_strength``. The Level is tested directly (a coarse period
        like an annual cycle is undetectable on the raw fold, but shows clearly
        in the per-cycle Level)."""
        periods = self._periods(state)
        if len(periods) < 2:
            return None
        P = int(periods[0])
        if "flair_level_work" not in state.historical_features:
            return None
        level = np.asarray(state.historical_features["flair_level_work"], dtype=np.float64)
        n_complete = level.shape[1]
        Lmean = level.mean(axis=0)
        used = set(int(c) for c in state.metadata.get("flair_cross_periods", []))
        best, best_strength = None, self.secondary_min_strength
        for sec in periods[1:]:
            sec = int(sec)
            if sec <= P:
                continue
            cp = sec // P
            if cp < 2 or cp in used or n_complete // cp < 2:
                continue
            S2 = self._estimate_shape2(Lmean, cp, n_complete)
            if S2 is None:
                continue
            strength = float(np.max(np.abs(S2 - 1.0)))
            if strength > best_strength:
                best, best_strength = (sec, cp), strength
        return best

    # -- primary fold: full within-period shape + level ------------------
    def _level_shrink_factors(self, matrix: np.ndarray) -> np.ndarray:
        factors = np.ones(matrix.shape[0], dtype=np.float64)
        for i in range(matrix.shape[0]):
            s = np.linalg.svd(matrix[i], compute_uv=False)
            if s.size <= 1 or s[0] <= _EPS:
                continue
            residual_noise = float(np.mean(s[1:] ** 2))
            signal = max(float(s[0] ** 2) - residual_noise, 0.0)
            factors[i] = np.clip(signal / float(s[0] ** 2), self.min_factor, 1.0)
        return factors

    def _fold_primary(self, state: "State", P: int) -> "State":
        hist = _flair_history(state)
        n_s = hist.shape[0]
        n_complete = hist.shape[1] // P
        usable = n_complete * P
        matrix = hist[:, -usable:].reshape(n_s, n_complete, P).transpose(0, 2, 1).astype(np.float32)  # (n,P,nc)
        level = matrix.sum(axis=1).astype(np.float32)                                                  # (n,nc)

        mat64 = matrix.astype(np.float64)
        K = min(max(self.shape_k, 1), n_complete)
        recent = mat64[:, :, -K:]
        totals = recent.sum(axis=1, keepdims=True)
        props = np.where(totals > _EPS, recent / np.maximum(totals, _EPS), 1.0 / max(P, 1))
        shape = props.mean(axis=2)
        shape = np.maximum(shape, _EPS)
        shape = (shape / np.maximum(shape.sum(axis=1, keepdims=True), _EPS)).astype(np.float32)        # (n,P)

        state.add_historical_feature("period_matrix", matrix)
        state.add_historical_feature("flair_period_matrix", matrix)
        state.add_historical_feature("level_series", level)
        state.add_historical_feature("flair_level_raw", level)
        state.add_historical_feature("shape_vector", shape)
        state.add_historical_feature("flair_shape", shape)
        state.metadata["period"] = int(P)
        state.metadata["n_complete_periods"] = int(n_complete)
        state.metadata["shape_k"] = int(K)
        state.register_artifact("period_matrix", store="historical_features",
                                kind="period_matrix", source_token=self.name, tags=("flair", "periodic"))
        state.register_artifact("level_series", store="historical_features",
                                kind="level_series", source_token=self.name, tags=("flair", "level"))
        state.register_artifact("shape_vector", store="historical_features",
                                kind="period_shape", source_token=self.name, tags=("flair", "shape"))

        work = level
        if self.shrink_level:
            factors = self._level_shrink_factors(mat64)
            denoised = (level.astype(np.float64) * factors[:, None]).astype(np.float32)
            state.add_historical_feature("flair_level_denoised", denoised)
            state.metadata["flair_level_shrinkage"] = factors.astype(np.float32)
            state.register_artifact("flair_level_denoised", store="historical_features",
                                    kind="level_series", source_token=self.name,
                                    tags=("flair", "level", "denoised"))
            work = denoised

        state.add_historical_feature("flair_level_work", work.astype(np.float32))
        state.register_artifact("flair_level_work", store="historical_features",
                                kind="level_series", source_token=self.name, tags=("flair", "level", "work"))
        m = int(np.ceil(state.horizon / max(P, 1)))
        state.features["flair_level_future_shape2"] = np.ones((n_s, m), dtype=np.float32)
        state.metadata["flair_cross_period"] = 1
        state.metadata["flair_cross_periods"] = []
        self._log_execution(
            state, reads={"flair_history": hist.shape, "period": P},
            writes={"period_matrix": matrix.shape, "level_series": level.shape, "shape_vector": shape.shape})
        return state

    # -- Shape_{k}: secondary level modulation (FLAIR _compute_shape2) ----
    def _estimate_shape2(self, L: np.ndarray, cp: int, n_complete: int) -> np.ndarray | None:
        nc2 = n_complete // cp
        if nc2 < 2:
            return None
        pos = np.arange(n_complete) % cp
        raw = np.array([L[pos == d].mean() if np.any(pos == d) else 1.0 for d in range(cp)], dtype=np.float64)
        rmean = raw.mean()
        if rmean < _EPS:
            return None
        raw = raw / rmean
        if self.harmonic_prior:
            t = np.arange(cp, dtype=np.float64)
            cos_b = np.cos(2 * np.pi * t / cp)
            sin_b = np.sin(2 * np.pi * t / cp)
            c = raw - 1.0
            a = 2.0 * np.mean(c * cos_b)
            b = 2.0 * np.mean(c * sin_b)
            harm = 1.0 + a * cos_b + b * sin_b
            rss_flat = float(np.sum(c ** 2))
            rss_harm = float(np.sum((raw - harm) ** 2))
            _LOG = 1e-12
            bic_flat = cp * np.log(max(rss_flat / cp, _LOG))
            bic_harm = cp * np.log(max(rss_harm / cp, _LOG)) + 2 * np.log(cp)
            prior = harm if bic_harm < bic_flat else np.ones(cp)
        else:
            prior = np.ones(cp)
        w = nc2 / (nc2 + cp)
        S2 = w * raw + (1.0 - w) * prior
        S2 = np.maximum(S2, _EPS)
        return S2 / S2.mean()

    def _shape_k(self, state: "State", P: int, cp: int, k: int) -> "State":
        level = np.asarray(state.historical_features["flair_level_work"], dtype=np.float64)
        n_s, n_complete = level.shape
        m = int(np.ceil(state.horizon / max(P, 1)))
        pos = np.arange(n_complete) % cp
        fut_pos = (n_complete + np.arange(m)) % cp

        shape2 = np.ones((n_s, cp), dtype=np.float64)
        level_work = level.copy()
        for i in range(n_s):
            S2 = self._estimate_shape2(level[i], cp, n_complete)
            if S2 is None:
                continue
            shape2[i] = S2
            level_work[i] = level[i] / S2[pos]

        future_factor = shape2[:, fut_pos]
        prev = np.asarray(state.features.get("flair_level_future_shape2", np.ones((n_s, m))), dtype=np.float64)
        if prev.shape != future_factor.shape:
            prev = np.ones_like(future_factor)
        future_shape2 = (prev * future_factor).astype(np.float32)

        state.add_historical_feature("flair_level_work", level_work.astype(np.float32))
        state.add_historical_feature("flair_shape%d" % k, shape2.astype(np.float32))
        if k == 2:
            state.add_historical_feature("flair_shape2", shape2.astype(np.float32))
        state.features["flair_level_future_shape2"] = future_shape2
        cps = list(state.metadata.get("flair_cross_periods", []))
        cps.append(int(cp))
        state.metadata["flair_cross_periods"] = cps
        state.metadata["flair_cross_period"] = int(max(cps))
        state.register_artifact("flair_level_work", store="historical_features",
                                kind="level_series", source_token=self.name,
                                tags=("flair", "level", "deseasonalized"))
        self._log_execution(
            state, reads={"flair_level_work": level.shape, "cp": cp, "shape_index": k},
            writes={"flair_shape%d" % k: shape2.shape, "flair_level_future_shape2": future_shape2.shape})
        return state


class LevelShapeRidgeToken(ModelToken):
    learning_scope = "within_series"
    """Forecast period-level values with internal Box-Cox ridge, then expand."""

    name = "level_shape_ridge"
    reads = ("level_series", "shape_vector", "period")
    writes = ("prediction_stack",)
    description = (
        "Compact Box-Cox soft-averaged ridge over period levels + shape expansion."
    )

    def __init__(
        self,
        alpha_log_min: float = -6.0,
        alpha_log_max: float = 3.0,
        n_alphas: int = 25,
        n_lambda_grid: int = 21,
        eps: float = 1e-6,
        level_sources: tuple[str, ...] = (
            "flair_level_work",
            "flair_level_denoised",
            "level_series",
        ),
        show_progress: bool = True,
        progress_min_samples: int = 32,
    ):
        self.alpha_log_min = alpha_log_min
        self.alpha_log_max = alpha_log_max
        self.n_alphas = int(n_alphas)
        self.n_lambda_grid = int(n_lambda_grid)
        self.eps = float(eps)
        self.level_sources = tuple(level_sources)
        self.show_progress = show_progress
        self.progress_min_samples = int(progress_min_samples)

    def get_model(self) -> dict[str, Any]:
        return {
            "model": "boxcox_soft_averaged_ridge",
            "target": "period_level",
            "n_alphas": self.n_alphas,
            "n_lambda_grid": self.n_lambda_grid,
        }

    def check_specific_conditions(self, state: "State") -> bool:
        if not super().check_specific_conditions(state):
            return False
        return (
            state.n_models_applied == 0
            and "period" in state.metadata
            and "shape_vector" in state.historical_features
            and self._level_name(state) is not None
        )

    def _level_name(self, state: "State") -> str | None:
        for name in self.level_sources:
            if name in state.historical_features:
                return name
        return None

    def apply(self, state: "State") -> "State":
        level_name = self._level_name(state)
        if level_name is None:
            raise KeyError(
                f"{self.name} requires one of {self.level_sources!r} in historical_features."
            )

        level = np.asarray(state.historical_features[level_name], dtype=np.float64)
        shape = np.asarray(state.historical_features["shape_vector"], dtype=np.float64)
        period = int(state.metadata["period"])
        horizon = state.horizon
        m = int(np.ceil(horizon / max(period, 1)))
        cp = int(state.metadata.get("flair_cross_period", 1))
        future_shape2 = np.asarray(
            state.features.get("flair_level_future_shape2", np.ones((state.n_samples, m))),
            dtype=np.float64,
        )

        pred = np.zeros((state.n_samples, horizon), dtype=np.float32)
        alphas = np.logspace(self.alpha_log_min, self.alpha_log_max, self.n_alphas)
        iterator = _maybe_tqdm(
            range(state.n_samples),
            self.show_progress and state.n_samples >= self.progress_min_samples,
            "level_shape_ridge",
        )
        for i in iterator:
            offset = max(0.0, self.eps - float(np.min(level[i])))
            positive = level[i] + offset
            lam = _boxcox_lambda_grid(positive, self.n_lambda_grid)
            transformed = _boxcox(positive, lam)
            last = float(transformed[-1])
            innov = transformed - last

            beta, _, phi = _fit_level_ridge(innov, cp, alphas)
            innov_hat = _forecast_level_innov(innov, beta, cp, phi, m)
            level_hat = _boxcox_inv(innov_hat + last, lam) - offset
            level_hat = level_hat * future_shape2[i, :m]
            pred[i] = _expand_levels(level_hat, shape[i], horizon).astype(np.float32)

        state.push_prediction(pred, self.name)
        self._log_execution(
            state,
            reads={
                level_name: level.shape,
                "shape_vector": shape.shape,
                "period": period,
                "cross_period": cp,
            },
            writes={"prediction_stack[-1]": pred.shape},
        )
        return state


class FlairRidgeLevelToken(ModelToken):
    learning_scope = "within_series"  # per-series ridge AR on its own level
    """Forecast internally transformed FLAIR levels, expand through shape, and push a point forecast."""

    name = "FlairRidgeLevel"
    reads = ("flair_level_work", "shape_vector", "period")
    writes = ("prediction_stack", "flair_point_forecast")
    description = "FLAIR-style soft-averaged Ridge with internal Box-Cox level centering."

    def __init__(
        self,
        alpha_log_min: float = -6.0,
        alpha_log_max: float = 3.0,
        n_alphas: int = 25,
        n_lambda_grid: int = 21,
        eps: float = 1e-6,
        level_sources: tuple[str, ...] = (
            "flair_level_work",
            "flair_level_denoised",
            "level_series",
        ),
        show_progress: bool = True,
        progress_min_samples: int = 32,
    ):
        self.alpha_log_min = alpha_log_min
        self.alpha_log_max = alpha_log_max
        self.n_alphas = int(n_alphas)
        self.n_lambda_grid = int(n_lambda_grid)
        self.eps = float(eps)
        self.level_sources = tuple(level_sources)
        self.show_progress = show_progress
        self.progress_min_samples = int(progress_min_samples)

    def get_model(self) -> dict[str, Any]:
        return {
            "model": "internal_boxcox_soft_averaged_ridge",
            "space": "period_level",
            "n_alphas": self.n_alphas,
            "n_lambda_grid": self.n_lambda_grid,
        }

    def check_specific_conditions(self, state: "State") -> bool:
        if not super().check_specific_conditions(state):
            return False
        non_flair = [t for t in state.transform_stack if t.name != "flair_shift"]
        return (
            state.n_models_applied == 0
            and not non_flair
            and "shape_vector" in state.historical_features
            and self._level_name(state) is not None
        )

    def _level_name(self, state: "State") -> str | None:
        for name in self.level_sources:
            if name in state.historical_features:
                return name
        return None

    def apply(self, state: "State") -> "State":
        level_name = self._level_name(state)
        if level_name is None:
            raise KeyError(
                f"{self.name} requires one of {self.level_sources!r} in historical_features."
            )

        level_raw = np.asarray(state.historical_features[level_name], dtype=np.float64)
        shape = np.asarray(state.historical_features["shape_vector"], dtype=np.float64)
        shift = np.asarray(state.metadata.get("flair_shift", 0.0), dtype=np.float64)
        if shift.ndim == 0:
            shift = np.full(state.n_samples, float(shift), dtype=np.float64)

        period = int(state.metadata["period"])
        horizon = state.horizon
        m = int(np.ceil(horizon / max(period, 1)))
        cp = int(state.metadata.get("flair_cross_period", 1))
        future_shape2 = np.asarray(
            state.features.get("flair_level_future_shape2", np.ones((state.n_samples, m))),
            dtype=np.float64,
        )

        pred = np.zeros((state.n_samples, horizon), dtype=np.float32)
        level_point = np.zeros((state.n_samples, m), dtype=np.float32)
        level_innov_point = np.zeros((state.n_samples, m), dtype=np.float32)
        betas = np.zeros((state.n_samples, 4), dtype=np.float32)
        phis = np.zeros(state.n_samples, dtype=np.float32)
        lambdas = np.ones(state.n_samples, dtype=np.float64)
        offsets = np.zeros(state.n_samples, dtype=np.float64)
        last_values = np.zeros(state.n_samples, dtype=np.float64)
        residuals: list[np.ndarray] = []
        alphas = np.logspace(self.alpha_log_min, self.alpha_log_max, self.n_alphas)

        iterator = _maybe_tqdm(
            range(state.n_samples),
            self.show_progress and state.n_samples >= self.progress_min_samples,
            "FLAIR Ridge levels",
        )
        for i in iterator:
            offsets[i] = max(0.0, self.eps - float(np.min(level_raw[i])))
            positive = level_raw[i] + offsets[i]
            lambdas[i] = _boxcox_lambda_grid(positive, self.n_lambda_grid)
            transformed = _boxcox(positive, float(lambdas[i]))
            last_values[i] = float(transformed[-1])
            level_innov = transformed - last_values[i]

            beta, resid, phi = _fit_level_ridge(level_innov, cp, alphas)
            innov_hat = _forecast_level_innov(level_innov, beta, cp, phi, m)
            level_hat = _boxcox_inv(innov_hat + last_values[i], lambdas[i]) - offsets[i]
            level_hat = level_hat * future_shape2[i, :m]
            expanded = _expand_levels(level_hat, shape[i], horizon)

            betas[i, : beta.size] = beta.astype(np.float32)
            phis[i] = np.float32(phi)
            residuals.append(resid.astype(np.float32))
            level_innov_point[i] = innov_hat.astype(np.float32)
            level_point[i] = level_hat.astype(np.float32)
            pred[i] = expanded.astype(np.float32)

        state.features["flair_point_forecast"] = pred
        state.features["flair_level_point"] = level_point
        state.features["flair_level_innov_point"] = level_innov_point
        state.features["flair_ridge_beta"] = betas
        state.features["flair_ridge_residuals"] = residuals
        state.features["flair_ridge_phi"] = phis
        state.metadata[self.name] = {
            **self.get_model(),
            "period": period,
            "cross_period": cp,
            "level_steps": m,
            "level_source": level_name,
            "mode": "history_only_point_forecast",
            "level_transform": {
                "lambda": lambdas.astype(np.float32),
                "offset": offsets.astype(np.float32),
                "last": last_values.astype(np.float32),
            },
        }
        state.register_artifact(
            "flair_point_forecast",
            store="features",
            kind="forecast",
            role="point_prediction",
            source_token=self.name,
            tags=("flair", "point"),
        )
        state.push_prediction(pred, self.name)
        self._log_execution(
            state,
            reads={level_name: level_raw.shape, "shape_vector": shape.shape},
            writes={
                "prediction_stack[-1]": pred.shape,
                "flair_point_forecast": pred.shape,
                "flair_ridge_beta": betas.shape,
            },
        )
        return state


class FlairSamplePathsToken(FeatureToken):
    learning_scope = "within_series"
    """Generate FLAIR-style stochastic samples around the point forecast."""

    name = "FlairSamplePaths"
    reads = ("flair_level_work", "shape_vector", "flair_ridge_beta", "FlairRidgeLevel")
    writes = ("flair_samples", "flair_sample_mean")
    description = "Bootstrap level residuals and coherent phase noise into forecast samples."

    def __init__(
        self,
        n_paths: int = 100,
        seed: int = 0,
        phase_noise_scale: float = 1.0,
        clip_to_history: bool = True,
        max_sample_values: int = 5_000_000,
        show_progress: bool = True,
        progress_min_samples: int = 16,
    ):
        self.n_paths = n_paths
        self.seed = seed
        self.phase_noise_scale = phase_noise_scale
        self.clip_to_history = clip_to_history
        self.max_sample_values = max_sample_values
        self.show_progress = show_progress
        self.progress_min_samples = progress_min_samples

    def check_specific_conditions(self, state: "State") -> bool:
        return (
            super().check_specific_conditions(state)
            and "flair_ridge_beta" in state.features
            and "FlairRidgeLevel" in state.token_sequence
        )

    def apply(self, state: "State") -> "State":
        ridge_meta = state.metadata["FlairRidgeLevel"]
        level_name = ridge_meta.get("level_source", "flair_level_work")
        level_raw = np.asarray(state.historical_features[level_name], dtype=np.float64)
        shape = np.asarray(state.historical_features["shape_vector"], dtype=np.float64)
        matrix = np.asarray(state.historical_features["period_matrix"], dtype=np.float64)
        raw_level = np.asarray(state.historical_features["level_series"], dtype=np.float64)
        level_transform = ridge_meta["level_transform"]
        lambdas = np.asarray(level_transform["lambda"], dtype=np.float64)
        offsets = np.asarray(level_transform["offset"], dtype=np.float64)
        last_values = np.asarray(level_transform["last"], dtype=np.float64)
        beta = np.asarray(state.features["flair_ridge_beta"], dtype=np.float64)
        phi = np.asarray(state.features["flair_ridge_phi"], dtype=np.float64)
        residuals = state.features["flair_ridge_residuals"]
        shift = np.asarray(state.metadata.get("flair_shift", 0.0), dtype=np.float64)
        if shift.ndim == 0:
            shift = np.full(state.n_samples, float(shift), dtype=np.float64)

        period = int(state.metadata["period"])
        horizon = state.horizon
        m = int(np.ceil(horizon / max(period, 1)))
        cp = int(state.metadata.get("flair_cross_period", 1))
        future_shape2 = np.asarray(
            state.features.get("flair_level_future_shape2", np.ones((state.n_samples, m))),
            dtype=np.float64,
        )
        n_paths = _bounded_paths(
            state.n_samples, self.n_paths, horizon, self.max_sample_values
        )
        rng = np.random.default_rng(self.seed)
        samples = np.zeros((state.n_samples, n_paths, horizon), dtype=np.float32)

        iterator = _maybe_tqdm(
            range(state.n_samples),
            self.show_progress and state.n_samples >= self.progress_min_samples,
            "FLAIR sample paths",
        )
        phases = np.arange(horizon) % max(period, 1)
        steps = np.arange(horizon) // max(period, 1)
        for i in iterator:
            level = _boxcox(level_raw[i] + offsets[i], float(lambdas[i])) - last_values[i]
            rel_noise = _relative_phase_noise(matrix[i], raw_level[i], shape[i])
            hist_min = float(np.min(state.original_history[i]))
            hist_max = float(np.max(state.original_history[i]))
            for k in range(n_paths):
                innov = _forecast_level_innov(
                    level,
                    beta[i, :],
                    cp,
                    float(phi[i]),
                    m,
                    residual_pool=np.asarray(residuals[i], dtype=np.float64),
                    rng=rng,
                )
                level_hat = _boxcox_inv(innov + last_values[i], lambdas[i]) - offsets[i]
                level_hat = level_hat * future_shape2[i, :m]
                values = level_hat[steps] * shape[i, phases]
                if rel_noise.shape[1] > 0:
                    col = int(rng.integers(0, rel_noise.shape[1]))
                    values = values * (
                        1.0 + self.phase_noise_scale * rel_noise[phases, col]
                    )
                values = values - shift[i]
                if self.clip_to_history and np.isfinite(hist_min) and np.isfinite(hist_max):
                    values = np.clip(values, hist_min, hist_max)
                samples[i, k] = values.astype(np.float32)

        sample_mean = samples.mean(axis=1).astype(np.float32)
        state.features["flair_samples"] = samples
        state.features["flair_sample_mean"] = sample_mean
        state.metadata[self.name] = {
            "n_paths_requested": int(self.n_paths),
            "n_paths": int(n_paths),
            "seed": int(self.seed),
            "phase_noise_scale": float(self.phase_noise_scale),
        }
        state.register_artifact(
            "flair_samples",
            store="features",
            kind="forecast_samples",
            role="sample_prediction",
            source_token=self.name,
            tags=("flair", "samples"),
        )
        self._log_execution(
            state,
            reads={
                level_name: level_raw.shape,
                "shape_vector": shape.shape,
                "flair_ridge_beta": beta.shape,
            },
            writes={"flair_samples": samples.shape, "flair_sample_mean": sample_mean.shape},
        )
        return state


def _flair_history(state: "State") -> np.ndarray:
    if "clean_history" in state.historical_features:
        return np.asarray(state.historical_features["clean_history"], dtype=np.float64)
    if "flair_history" in state.historical_features:
        return np.asarray(state.historical_features["flair_history"], dtype=np.float64)
    return np.asarray(state.features["raw_history"], dtype=np.float64)


def _level_for_flair(state: "State") -> np.ndarray:
    if "flair_level_denoised" in state.historical_features:
        return np.asarray(state.historical_features["flair_level_denoised"], dtype=np.float64)
    return np.asarray(state.historical_features["level_series"], dtype=np.float64)


def _interp_finite_1d(y: np.ndarray) -> np.ndarray:
    return linear_fill_1d(y)


def _period_bic(y: np.ndarray, period: int) -> float:
    y = _interp_finite_1d(y)
    n = y.size
    if period <= 1:
        rss = float(np.var(y) * n)
        return n * np.log(max(rss / max(n, 1), 1e-30)) + np.log(max(n, 2))
    n_complete = n // period
    if n_complete <= 0:
        return float("inf")
    usable = n_complete * period
    matrix = y[-usable:].reshape(n_complete, period).T
    s = np.linalg.svd(matrix, compute_uv=False)
    rss = float(np.sum(s[1:] ** 2)) if s.size > 1 else 0.0
    return usable * np.log(max(rss / max(usable, 1), 1e-30)) + (
        period + n_complete - 1
    ) * np.log(max(usable, 2))



def _period_power(y: np.ndarray, period: int, band: int = 1) -> float:
    """Fraction of total spectral power at the fundamental frequency 1/period.

    Periodogram-based period score: high only when ``period`` is a true period.
    A multiple k*period maps to a subharmonic frequency 1/(k*period) with no
    power, so -- unlike the rank-1 BIC fold -- it does not reward multiples of
    the fundamental. Higher is better.
    """
    y = _interp_finite_1d(y)
    y = y - float(np.mean(y))
    T = y.size
    if T < 4 or period <= 1:
        return 0.0
    F = np.abs(np.fft.rfft(y)) ** 2
    total = float(F[1:].sum())                       # drop DC bin
    if total <= 0.0:
        return 0.0
    k = int(round(T / period))
    if k <= 0 or k >= F.size:
        return 0.0
    lo = max(1, k - band)
    hi = min(F.size, k + band + 1)
    return float(F[lo:hi].sum() / total)

def _boxcox(y: np.ndarray, lam: float) -> np.ndarray:
    y = np.maximum(np.asarray(y, dtype=np.float64), _EPS)
    if abs(lam) < 1e-8:
        return np.log(y)
    return (np.power(y, lam) - 1.0) / lam


def _boxcox_inv(z: np.ndarray, lam: float) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    if abs(lam) < 1e-8:
        return np.exp(np.clip(z, -50.0, 50.0))
    base = np.maximum(lam * z + 1.0, _EPS)
    return np.power(base, 1.0 / lam)


def _boxcox_lambda_grid(y: np.ndarray, n_grid: int) -> float:
    y = np.maximum(np.asarray(y, dtype=np.float64), _EPS)
    if y.size < 3 or np.std(y) <= _EPS:
        return 1.0
    lambdas = np.linspace(0.0, 1.0, max(int(n_grid), 2))
    log_y = np.log(y)
    best_lam = 1.0
    best_ll = -np.inf
    for lam in lambdas:
        z = _boxcox(y, float(lam))
        var = float(np.var(z))
        if var <= _EPS:
            continue
        ll = -0.5 * y.size * np.log(var) + (lam - 1.0) * float(np.sum(log_y))
        if ll > best_ll:
            best_ll = ll
            best_lam = float(lam)
    return best_lam


def _fit_level_ridge(
    y: np.ndarray, cp: int, alphas: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    y = np.asarray(y, dtype=np.float64)
    n = y.size
    start = max(1, cp) if cp >= 2 else 1
    if n - start < 3 or np.std(y) <= _EPS:
        beta = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float64)
        return beta, np.zeros(1, dtype=np.float64), 0.0

    rows = []
    target = []
    for t in range(start, n):
        rows.append(_level_design_row(y, t, n, cp, 0.0))
        target.append(y[t])
    x = np.asarray(rows, dtype=np.float64)
    target_arr = np.asarray(target, dtype=np.float64)
    beta, fitted, leverages = _ridge_soft_average(x, target_arr, alphas)
    resid = target_arr - fitted
    loo = resid / np.sqrt(np.maximum(1.0 - leverages, _EPS))
    phi = _estimate_phi(y)
    return beta, loo, phi


def _ridge_soft_average(
    x: np.ndarray, y: np.ndarray, alphas: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u, s, vt = np.linalg.svd(x, full_matrices=False)
    uty = u.T @ y
    s2 = s ** 2
    betas = []
    errors = []
    fitted_values = []
    leverage_values = []
    for alpha in alphas:
        shrink = s2 / (s2 + alpha)
        coef = vt.T @ ((s / (s2 + alpha)) * uty)
        fitted = u @ (shrink * uty)
        lev = np.sum((u ** 2) * shrink[None, :], axis=1)
        gcv = np.mean(((y - fitted) / np.maximum(1.0 - lev, _EPS)) ** 2)
        betas.append(coef)
        errors.append(float(gcv))
        fitted_values.append(fitted)
        leverage_values.append(lev)
    errors_arr = np.asarray(errors)
    scale = max(float(np.min(errors_arr)), _EPS)
    weights = np.exp(-(errors_arr - float(np.min(errors_arr))) / scale)
    weights = weights / max(float(np.sum(weights)), _EPS)
    beta = np.sum(np.asarray(betas) * weights[:, None], axis=0)
    fitted = np.sum(np.asarray(fitted_values) * weights[:, None], axis=0)
    leverages = np.sum(np.asarray(leverage_values) * weights[:, None], axis=0)
    return beta, fitted, leverages


def _level_design_row(values: np.ndarray, idx: int, n: int, cp: int, phi: float) -> np.ndarray:
    trend = _damped_trend(idx, n, phi)
    lag1 = values[idx - 1]
    lag_cp = values[idx - cp] if cp >= 2 and idx - cp >= 0 else 0.0
    return np.array([1.0, trend, lag1, lag_cp], dtype=np.float64)


def _forecast_level_innov(
    history: np.ndarray,
    beta: np.ndarray,
    cp: int,
    phi: float,
    m: int,
    *,
    residual_pool: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    values = [float(v) for v in np.asarray(history, dtype=np.float64)]
    n = len(values)
    beta = np.asarray(beta, dtype=np.float64)
    out = np.zeros(m, dtype=np.float64)
    pool = np.asarray(residual_pool, dtype=np.float64) if residual_pool is not None else None
    if pool is not None:
        pool = pool[np.isfinite(pool)]
    for j in range(m):
        idx = n + j
        row = _level_design_row(np.asarray(values, dtype=np.float64), idx, n, cp, phi)
        pred = float(row @ beta[:4])
        if pool is not None and pool.size and rng is not None:
            pred += float(rng.choice(pool))
        values.append(pred)
        out[j] = pred
    return out


def _damped_trend(idx: int, n: int, phi: float) -> float:
    if idx < n:
        return idx / max(n, 1)
    steps = idx - n + 1
    if phi <= 0:
        extra = 0.0
    elif abs(phi - 1.0) < 1e-6:
        extra = float(steps)
    else:
        extra = float(phi * (1.0 - phi ** steps) / (1.0 - phi))
    return (n - 1 + extra) / max(n, 1)


def _estimate_phi(y: np.ndarray) -> float:
    diff = np.diff(np.asarray(y, dtype=np.float64))
    if diff.size < 3 or np.std(diff) <= _EPS:
        return 0.0
    a = diff[:-1]
    b = diff[1:]
    denom = float(np.dot(a, a))
    if denom <= _EPS:
        return 0.0
    return float(np.clip(np.dot(a, b) / denom, 0.0, 0.98))


def _expand_levels(level_hat: np.ndarray, shape: np.ndarray, horizon: int) -> np.ndarray:
    period = shape.size
    phases = np.arange(horizon) % max(period, 1)
    steps = np.arange(horizon) // max(period, 1)
    return level_hat[steps] * shape[phases]


def _relative_phase_noise(
    matrix: np.ndarray, level: np.ndarray, shape: np.ndarray
) -> np.ndarray:
    fitted = shape[:, None] * level[None, :]
    denom = np.maximum(np.abs(fitted), np.maximum(np.median(np.abs(fitted)), 1.0) * 1e-3)
    rel = (matrix - fitted) / denom
    rel = np.clip(rel, -0.8, 2.0)
    rel = rel - np.mean(rel, axis=1, keepdims=True)
    return rel


def _bounded_paths(
    n_samples: int, requested: int, horizon: int, max_values: int
) -> int:
    requested = max(1, int(requested))
    max_paths = max(1, int(max_values) // max(n_samples * horizon, 1))
    return min(requested, max_paths)


def _maybe_tqdm(iterable, enabled: bool, desc: str):
    if not enabled:
        return iterable
    try:
        from tqdm import tqdm

        return tqdm(iterable, desc=desc)
    except Exception:
        return iterable
