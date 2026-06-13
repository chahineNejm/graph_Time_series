"""Experimental FLAIR-style token family.

The tokens in this module keep the FLAIR decomposition visible in State instead
of hiding it behind one monolithic forecaster. They are intentionally
self-contained and do not depend on private ``flaircast`` internals.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ..token import CleaningToken, FeatureToken, ModelToken

if TYPE_CHECKING:
    from ..state import State


FREQ_PERIODS = {
    "H": [24, 168],
    "HOURLY": [24, 168],
    "D": [7, 365],
    "DAILY": [7, 365],
    "W": [52],
    "WEEKLY": [52],
    "M": [12],
    "MONTHLY": [12],
    "15T": [4, 96],
    "5T": [12, 288],
}

_EPS = 1e-8


class FlairPreprocessToken(CleaningToken):
    """Interpolate missing history values and shift each series positive."""

    name = "FlairPreprocess"
    reads = ("raw_history",)
    writes = ("flair_history", "flair_shift")
    description = "FLAIR preprocessing: finite interpolation plus positivity shift."

    def __init__(self, eps: float = 1e-6):
        self.eps = eps

    def apply(self, state: "State") -> "State":
        raw = np.asarray(state.features["raw_history"], dtype=np.float64)
        cleaned = np.empty_like(raw, dtype=np.float64)
        shifts = np.zeros(raw.shape[0], dtype=np.float64)

        for i, row in enumerate(raw):
            y = _interp_finite_1d(row)
            shift = max(0.0, self.eps - float(np.min(y)))
            cleaned[i] = y + shift
            shifts[i] = shift

        cleaned32 = cleaned.astype(np.float32)
        state.add_historical_feature("flair_history", cleaned32)
        state.metadata["flair_shift"] = shifts.astype(np.float32)
        state.flags["flair_preprocessed"] = True
        state.register_artifact(
            "flair_history",
            store="historical_features",
            kind="sequence",
            role="clean_history",
            source_token=self.name,
            tags=("flair", "preprocess"),
        )

        # Register the positivity shift as an invertible transform. FLAIR builds
        # its forecast in this shifted space; registering the inverse here means
        # State.get_final_prediction() subtracts the shift automatically, so no
        # downstream model token has to remember to undo it.
        shift_col = shifts.reshape(-1, 1).astype(np.float64)

        def _unshift(pred, _s=shift_col):
            return np.asarray(pred, dtype=np.float64) - _s

        state.register_transform(
            name="flair_shift",
            inverse_fn=_unshift,
            params={"shift_shape": shifts.shape, "kind": "additive_positivity_shift"},
            affects="target",
        )
        self._log_execution(
            state,
            reads={"raw_history": raw.shape},
            writes={"flair_history": cleaned32.shape, "flair_shift": shifts.shape},
        )
        return state


class PeriodSelectionToken(FeatureToken):
    """Choose a global period with a rank-1 SVD/BIC score."""

    name = "PeriodSelection"
    reads = ("flair_history",)
    writes = ("period", "period_scores")
    description = "Select a FLAIR-style MDL/BIC period from frequency candidates."

    def __init__(
        self,
        freq: str = "H",
        max_series: int = 8,
        min_complete: int = 3,
    ):
        self.freq = freq
        self.max_series = max_series
        self.min_complete = min_complete

    def check_specific_conditions(self, state: "State") -> bool:
        return super().check_specific_conditions(state) and "period" not in state.metadata

    def apply(self, state: "State") -> "State":
        hist = _flair_history(state)
        n_steps = hist.shape[1]
        candidates = [
            p
            for p in FREQ_PERIODS.get(self.freq.upper(), [1])
            if p > 0 and n_steps // p >= self.min_complete
        ]
        candidates = sorted(set([1] + candidates))
        subset = hist[: min(self.max_series, hist.shape[0])]

        scores: dict[int, float] = {}
        for period in candidates:
            vals = [_period_bic(row, period) for row in subset]
            scores[int(period)] = float(np.mean(vals))

        period = min(scores, key=scores.get)
        secondary = [
            int(p)
            for p in candidates
            if p > period and period > 1 and p % period == 0
        ]
        state.metadata["period"] = int(period)
        state.metadata["period_scores"] = scores
        state.metadata["flair_secondary_periods"] = secondary
        state.flags["period_selected"] = True
        self._log_execution(
            state,
            reads={"flair_history": hist.shape},
            writes={
                "period": int(period),
                "period_scores": scores,
                "flair_secondary_periods": secondary,
            },
        )
        return state


class PeriodFoldToken(FeatureToken):
    """Fold preprocessed history into phase x complete-period matrices."""

    name = "PeriodFold"
    reads = ("flair_history", "period")
    writes = ("period_matrix", "level_series")
    description = "Fold FLAIR history into period phases and level totals."

    def check_specific_conditions(self, state: "State") -> bool:
        if not super().check_specific_conditions(state):
            return False
        period = int(state.metadata.get("period", 0))
        hist = _flair_history(state)
        return period > 0 and hist.shape[1] // period >= 1

    def apply(self, state: "State") -> "State":
        hist = _flair_history(state)
        period = int(state.metadata["period"])
        n_complete = hist.shape[1] // period
        usable = n_complete * period
        matrix = hist[:, -usable:].reshape(hist.shape[0], n_complete, period)
        matrix = matrix.transpose(0, 2, 1).astype(np.float32)
        level = matrix.sum(axis=1).astype(np.float32)

        state.add_historical_feature("period_matrix", matrix)
        state.add_historical_feature("level_series", level)
        state.add_historical_feature("flair_period_matrix", matrix)
        state.add_historical_feature("flair_level_raw", level)
        state.metadata["n_complete_periods"] = int(n_complete)
        state.register_artifact(
            "period_matrix",
            store="historical_features",
            kind="period_matrix",
            source_token=self.name,
            tags=("flair", "periodic"),
        )
        state.register_artifact(
            "level_series",
            store="historical_features",
            kind="level_series",
            source_token=self.name,
            tags=("flair", "level"),
        )
        self._log_execution(
            state,
            reads={"flair_history": hist.shape, "period": period},
            writes={"period_matrix": matrix.shape, "level_series": level.shape},
        )
        return state


class LevelShrinkageToken(FeatureToken):
    """Denoise period totals with a rank-1 energy shrinkage factor."""

    name = "LevelShrinkage"
    reads = ("period_matrix", "level_series")
    writes = ("flair_level_denoised", "flair_level_shrinkage")
    description = "Shrink FLAIR level totals using the folded matrix SVD spectrum."

    def __init__(self, min_factor: float = 0.05):
        self.min_factor = min_factor

    def apply(self, state: "State") -> "State":
        matrix = np.asarray(state.historical_features["period_matrix"], dtype=np.float64)
        level = np.asarray(state.historical_features["level_series"], dtype=np.float64)
        factors = np.ones(matrix.shape[0], dtype=np.float64)

        for i in range(matrix.shape[0]):
            s = np.linalg.svd(matrix[i], compute_uv=False)
            if s.size <= 1 or s[0] <= _EPS:
                factors[i] = 1.0
                continue
            residual_noise = float(np.mean(s[1:] ** 2))
            signal = max(float(s[0] ** 2) - residual_noise, 0.0)
            factors[i] = np.clip(signal / float(s[0] ** 2), self.min_factor, 1.0)

        denoised = (level * factors[:, None]).astype(np.float32)
        state.add_historical_feature("flair_level_denoised", denoised)
        state.metadata["flair_level_shrinkage"] = factors.astype(np.float32)
        state.register_artifact(
            "flair_level_denoised",
            store="historical_features",
            kind="level_series",
            source_token=self.name,
            tags=("flair", "level", "denoised"),
        )
        self._log_execution(
            state,
            reads={"period_matrix": matrix.shape, "level_series": level.shape},
            writes={
                "flair_level_denoised": denoised.shape,
                "flair_level_shrinkage": factors.shape,
            },
        )
        return state


class ShapeLevelToken(FeatureToken):
    """Estimate a frozen within-period shape from recent periods."""

    name = "ShapeLevel"
    reads = ("period_matrix", "level_series")
    writes = ("shape_vector", "flair_shape_history")
    description = "Estimate FLAIR within-period proportions from recent periods."

    def __init__(self, shape_k: int = 2):
        self.shape_k = shape_k

    def apply(self, state: "State") -> "State":
        matrix = np.asarray(state.historical_features["period_matrix"], dtype=np.float64)
        period = matrix.shape[1]
        n_complete = matrix.shape[2]
        k = min(max(int(self.shape_k), 1), n_complete)
        totals = matrix.sum(axis=1, keepdims=True)
        uniform = np.full_like(matrix, 1.0 / max(period, 1), dtype=np.float64)
        props = np.where(np.abs(totals) > _EPS, matrix / np.maximum(totals, _EPS), uniform)
        shape = props[:, :, -k:].mean(axis=2)
        shape = np.maximum(shape, _EPS)
        shape = shape / np.maximum(shape.sum(axis=1, keepdims=True), _EPS)

        shape32 = shape.astype(np.float32)
        props32 = props.astype(np.float32)
        state.add_historical_feature("shape_vector", shape32)
        state.add_historical_feature("flair_shape", shape32)
        state.add_historical_feature("flair_shape_history", props32)
        state.metadata["shape_k"] = int(k)
        state.register_artifact(
            "shape_vector",
            store="historical_features",
            kind="period_shape",
            source_token=self.name,
            tags=("flair", "shape"),
        )
        self._log_execution(
            state,
            reads={"period_matrix": matrix.shape},
            writes={"shape_vector": shape32.shape, "flair_shape_history": props32.shape},
        )
        return state


class SecondaryLevelSeasonalityToken(FeatureToken):
    """Estimate secondary seasonality on the compressed level series."""

    name = "SecondaryLevelSeasonality"
    reads = ("level_series", "period")
    writes = ("flair_shape2", "flair_level_work")
    description = "Estimate secondary FLAIR seasonality over period-level totals."

    def __init__(self, min_complete: int = 2):
        self.min_complete = min_complete

    def apply(self, state: "State") -> "State":
        level = _level_for_flair(state)
        period = int(state.metadata["period"])
        n_samples, n_complete = level.shape
        horizon = state.horizon
        m = int(np.ceil(horizon / max(period, 1)))

        cross_periods = _cross_periods(state, period, n_complete, self.min_complete)
        cp = max(cross_periods) if cross_periods else 1
        shape2 = np.ones((n_samples, cp), dtype=np.float64)
        level_work = level.astype(np.float64).copy()

        if cp > 1:
            positions = np.arange(n_complete) % cp
            for i in range(n_samples):
                mean_level = max(float(np.mean(level[i])), _EPS)
                raw = np.ones(cp, dtype=np.float64)
                for r in range(cp):
                    vals = level[i, positions == r]
                    if vals.size:
                        raw[r] = float(np.mean(vals)) / mean_level
                raw = raw / max(float(np.mean(raw)), _EPS)
                n_groups = max(n_complete // cp, 1)
                weight = n_groups / (n_groups + cp)
                shape2[i] = weight * raw + (1.0 - weight)
                level_work[i] = level[i] / np.maximum(shape2[i, positions], _EPS)

        future_positions = (n_complete + np.arange(m)) % cp
        future_shape2 = shape2[:, future_positions].astype(np.float32)
        state.add_historical_feature("flair_shape2", shape2.astype(np.float32))
        state.add_historical_feature("flair_level_work", level_work.astype(np.float32))
        state.features["flair_level_future_shape2"] = future_shape2
        state.metadata["flair_cross_period"] = int(cp)
        state.metadata["flair_cross_periods"] = [int(x) for x in cross_periods]
        state.register_artifact(
            "flair_level_work",
            store="historical_features",
            kind="level_series",
            source_token=self.name,
            tags=("flair", "level", "secondary"),
        )
        self._log_execution(
            state,
            reads={"level_series": level.shape, "period": period},
            writes={
                "flair_shape2": shape2.shape,
                "flair_level_work": level_work.shape,
                "flair_level_future_shape2": future_shape2.shape,
            },
        )
        return state


class LevelBoxCoxCenterToken(FeatureToken):
    """Apply a per-series Box-Cox transform to level totals and center at now."""

    name = "LevelBoxCoxCenter"
    reads = ("flair_level_work",)
    writes = ("flair_level_innov", "flair_boxcox")
    description = "Box-Cox transform FLAIR levels and center by the last level."

    def __init__(self, n_lambda_grid: int = 21, eps: float = 1e-6):
        self.n_lambda_grid = n_lambda_grid
        self.eps = eps

    def apply(self, state: "State") -> "State":
        level = np.asarray(state.historical_features["flair_level_work"], dtype=np.float64)
        n_samples, _ = level.shape
        lambdas = np.ones(n_samples, dtype=np.float64)
        offsets = np.zeros(n_samples, dtype=np.float64)
        last_values = np.zeros(n_samples, dtype=np.float64)
        bc = np.zeros_like(level, dtype=np.float64)

        for i in range(n_samples):
            offset = max(0.0, self.eps - float(np.min(level[i])))
            y = level[i] + offset
            lam = _boxcox_lambda_grid(y, self.n_lambda_grid)
            z = _boxcox(y, lam)
            offsets[i] = offset
            lambdas[i] = lam
            last_values[i] = z[-1]
            bc[i] = z

        innov = (bc - last_values[:, None]).astype(np.float32)
        state.add_historical_feature("flair_level_bc", bc.astype(np.float32))
        state.add_historical_feature("flair_level_innov", innov)
        state.metadata["flair_boxcox"] = {
            "lambda": lambdas.astype(np.float32),
            "offset": offsets.astype(np.float32),
            "last": last_values.astype(np.float32),
        }
        self._log_execution(
            state,
            reads={"flair_level_work": level.shape},
            writes={"flair_level_innov": innov.shape, "flair_boxcox": n_samples},
        )
        return state


class FlairRidgeLevelToken(ModelToken):
    learning_scope = "within_series"  # per-series ridge AR on its own level
    """Forecast compressed FLAIR levels, expand through shape, and push a point forecast."""

    name = "FlairRidgeLevel"
    reads = ("flair_level_innov", "shape_vector")
    writes = ("prediction_stack", "flair_point_forecast")
    description = "FLAIR-style soft-averaged Ridge over compressed level innovations."

    def __init__(
        self,
        alpha_log_min: float = -6.0,
        alpha_log_max: float = 3.0,
        n_alphas: int = 25,
        show_progress: bool = True,
        progress_min_samples: int = 32,
    ):
        self.alpha_log_min = alpha_log_min
        self.alpha_log_max = alpha_log_max
        self.n_alphas = n_alphas
        self.show_progress = show_progress
        self.progress_min_samples = progress_min_samples

    def get_model(self) -> dict[str, Any]:
        return {
            "model": "soft_averaged_ridge",
            "space": "flair_level",
            "n_alphas": self.n_alphas,
        }

    def check_specific_conditions(self, state: "State") -> bool:
        if not super().check_specific_conditions(state):
            return False
        non_flair = [t for t in state.transform_stack if t.name != "flair_shift"]
        return state.n_models_applied == 0 and not non_flair

    def apply(self, state: "State") -> "State":
        level = np.asarray(state.historical_features["flair_level_innov"], dtype=np.float64)
        shape = np.asarray(state.historical_features["shape_vector"], dtype=np.float64)
        boxcox = state.metadata["flair_boxcox"]
        lambdas = np.asarray(boxcox["lambda"], dtype=np.float64)
        offsets = np.asarray(boxcox["offset"], dtype=np.float64)
        last_values = np.asarray(boxcox["last"], dtype=np.float64)
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
        residuals: list[np.ndarray] = []
        alphas = np.logspace(self.alpha_log_min, self.alpha_log_max, self.n_alphas)

        iterator = _maybe_tqdm(
            range(state.n_samples),
            self.show_progress and state.n_samples >= self.progress_min_samples,
            "FLAIR Ridge levels",
        )
        for i in iterator:
            beta, resid, phi = _fit_level_ridge(level[i], cp, alphas)
            innov_hat = _forecast_level_innov(level[i], beta, cp, phi, m)
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
            "mode": "history_only_point_forecast",
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
            reads={"flair_level_innov": level.shape, "shape_vector": shape.shape},
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
    reads = ("flair_level_innov", "shape_vector")
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
        level = np.asarray(state.historical_features["flair_level_innov"], dtype=np.float64)
        shape = np.asarray(state.historical_features["shape_vector"], dtype=np.float64)
        matrix = np.asarray(state.historical_features["period_matrix"], dtype=np.float64)
        raw_level = np.asarray(state.historical_features["level_series"], dtype=np.float64)
        boxcox = state.metadata["flair_boxcox"]
        lambdas = np.asarray(boxcox["lambda"], dtype=np.float64)
        offsets = np.asarray(boxcox["offset"], dtype=np.float64)
        last_values = np.asarray(boxcox["last"], dtype=np.float64)
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
            rel_noise = _relative_phase_noise(matrix[i], raw_level[i], shape[i])
            hist_min = float(np.min(state.original_history[i]))
            hist_max = float(np.max(state.original_history[i]))
            for k in range(n_paths):
                innov = _forecast_level_innov(
                    level[i],
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
                "flair_level_innov": level.shape,
                "shape_vector": shape.shape,
                "flair_ridge_beta": beta.shape,
            },
            writes={"flair_samples": samples.shape, "flair_sample_mean": sample_mean.shape},
        )
        return state


def _flair_history(state: "State") -> np.ndarray:
    if "flair_history" in state.historical_features:
        return np.asarray(state.historical_features["flair_history"], dtype=np.float64)
    return np.asarray(state.features["raw_history"], dtype=np.float64)


def _level_for_flair(state: "State") -> np.ndarray:
    if "flair_level_denoised" in state.historical_features:
        return np.asarray(state.historical_features["flair_level_denoised"], dtype=np.float64)
    return np.asarray(state.historical_features["level_series"], dtype=np.float64)


def _interp_finite_1d(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    good = np.isfinite(y)
    if np.all(good):
        return y.copy()
    if not np.any(good):
        return np.zeros_like(y, dtype=np.float64)
    x = np.arange(y.size)
    return np.interp(x, x[good], y[good]).astype(np.float64)


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


def _cross_periods(
    state: "State", period: int, n_complete: int, min_complete: int
) -> list[int]:
    values = []
    for secondary in state.metadata.get("flair_secondary_periods", []):
        if secondary > period and secondary % period == 0:
            cp = secondary // period
            if cp >= 2 and n_complete // cp >= min_complete:
                values.append(int(cp))
    return sorted(set(values))


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

