"""Built-in token block registry."""

from __future__ import annotations

from .flair import (
    FlairPreprocessToken,
    FlairRidgeLevelToken,
    FlairSamplePathsToken,
    LevelBoxCoxCenterToken,
    LevelShapeRidgeToken,
    SeasonalFoldToken,
)
from .fourier import FourierFeaturesToken
from .imputation import ForwardFillToken, LinearFillToken
from .kernel_rbf import KernelRBFToken
from .normalization import MeanAbsScalingToken, ZNormalizationToken
from .parrot import ParrotToken, ParrotDatasetToken
from .periodic import PeriodPhaseOneHotToken
from .seasonal import PeriodDetectToken, PeriodDetectBICToken, PeriodDetectSpectralToken, SeasonalFeaturesToken
from .step_regression import StepRegressionToken, collect_step_covariates
from .tabular_models import LightGBMTabularToken, RandomForestTabularToken
from .versatile import (
    GBLevelForecastToken,
    VersatileGradientBoostingToken,
    VersatileTabularToken,
)


def register_default_tokens(grammar):
    """Register the built-in scaling/feature/model blocks (binder-free).

    Models consume the Signal-board auto-bundle, so no binder token is needed:
    a model may follow any scaling/feature chain directly.
    """
    scalers = ["ZNormalization", "MeanAbsScaling"]
    cleaners = ["LinearFill", "ForwardFill"]
    features = ["FourierFeatures"]
    models = ["kernel_rbf", "rf_tabular", "lightgbm_tabular", "parrot", "parrot_dataset"]
    non_parrot_models = ["kernel_rbf", "rf_tabular", "lightgbm_tabular"]
    parrot_sources = scalers + features + non_parrot_models

    grammar.register(LinearFillToken(), follows=["START"],
                     leads_to=scalers + features + models)
    grammar.register(ForwardFillToken(), follows=["START"],
                     leads_to=scalers + features + models)
    grammar.register(ZNormalizationToken(), follows=["START"] + cleaners,
                     leads_to=features + models)
    grammar.register(MeanAbsScalingToken(), follows=["START"] + cleaners,
                     leads_to=features + models)
    grammar.register(FourierFeaturesToken(),
                     follows=["START"] + cleaners + scalers,
                     leads_to=features + models)
    grammar.register(KernelRBFToken(),
                     follows=scalers + features + ["kernel_rbf", "parrot"],
                     leads_to=["kernel_rbf", "parrot", "STOP"])
    grammar.register(RandomForestTabularToken(),
                     follows=scalers + features + ["parrot"],
                     leads_to=["parrot", "STOP"])
    grammar.register(LightGBMTabularToken(),
                     follows=scalers + features + ["parrot"],
                     leads_to=["parrot", "STOP"])
    grammar.register(ParrotToken(),
                     follows=parrot_sources,
                     leads_to=non_parrot_models + ["STOP"])
    grammar.register(ParrotDatasetToken(),
                     follows=parrot_sources,
                     leads_to=non_parrot_models + ["STOP"])
    return grammar


def register_flair_tokens(grammar):
    """Register the experimental FLAIR-style branch.

    The seasonal decomposition is now a single repeatable ``SeasonalFold``
    token (it replaced PeriodFold + ShapeLevel + SecondaryLevelSeasonality):
    call it once per detected period. Two model heads consume its output -- the
    compact ``level_shape_ridge`` (Box-Cox ridge folded into one token) and the
    explicit ``LevelBoxCoxCenter`` -> ``FlairRidgeLevel`` -> ``FlairSamplePaths``
    path. Kept separate from ``register_default_tokens`` so routine searches do
    not widen unless the caller opts in.
    """

    grammar.register(
        FlairPreprocessToken(),
        follows=["START"],
        leads_to=["PeriodDetect"],
    )
    grammar.register(
        PeriodDetectToken(),
        follows=["FlairPreprocess", "LinearFill", "ForwardFill"],
        leads_to=["PeriodPhaseOneHot", "SeasonalFold"],
    )
    grammar.register(
        PeriodDetectSpectralToken(),
        follows=["FlairPreprocess", "LinearFill", "ForwardFill"],
        leads_to=["PeriodPhaseOneHot", "SeasonalFold"],
    )
    grammar.register(
        PeriodDetectBICToken(),
        follows=["FlairPreprocess", "LinearFill", "ForwardFill"],
        leads_to=["PeriodPhaseOneHot", "SeasonalFold"],
    )
    grammar.register(
        PeriodPhaseOneHotToken(),
        follows=["PeriodDetect", "PeriodDetectSpectral", "PeriodDetectBIC"],
        leads_to=["SeasonalFold"],
    )
    grammar.register(
        SeasonalFoldToken(),
        follows=["PeriodDetect", "PeriodDetectSpectral", "PeriodDetectBIC", "PeriodPhaseOneHot", "SeasonalFold"],
        leads_to=["SeasonalFold", "level_shape_ridge", "LevelBoxCoxCenter"],
    )
    grammar.register(
        LevelShapeRidgeToken(),
        follows=["SeasonalFold"],
        leads_to=["kernel_rbf", "parrot", "STOP"],
    )
    grammar.register(
        LevelBoxCoxCenterToken(),
        follows=["SeasonalFold"],
        leads_to=["FlairRidgeLevel"],
    )
    grammar.register(
        FlairRidgeLevelToken(),
        follows=["LevelBoxCoxCenter"],
        leads_to=["FlairSamplePaths", "parrot", "STOP"],
    )
    grammar.register(
        FlairSamplePathsToken(),
        follows=["FlairRidgeLevel"],
        leads_to=["STOP"],
    )
    return grammar


def register_versatile_tokens(grammar):
    """Register binder-free versatile models + exogenous calendar feature.

    These rely on the Signal board's auto-bundle, so they need no binder token
    and can follow any cleaning/scaling/feature chain.
    """
    feature_sources = [
        "START", "LinearFill", "ForwardFill",
        "ZNormalization", "MeanAbsScaling", "FourierFeatures",
    ]
    grammar.register(
        VersatileGradientBoostingToken(),
        follows=feature_sources + ["parrot", "versatile_gb"],
        leads_to=["STOP", "parrot", "versatile_gb"],
    )
    return grammar


def register_flair_gb_swap(grammar):
    """Register the gradient-boosted period-level forecaster as a FLAIR step.

    Slots in after ``SeasonalFold`` exactly where ``level_shape_ridge`` would,
    showing a FLAIR sub-step (predicting the period scale) served by a
    different learner.
    """
    grammar.register(
        GBLevelForecastToken(),
        follows=["SeasonalFold"],
        leads_to=["parrot", "STOP"],
    )
    return grammar


def register_seasonal_tokens(grammar):
    """Period detector + seasonal sinusoid features (auto-bundle friendly)."""
    grammar.register(
        PeriodDetectToken(),
        follows=["START", "LinearFill", "ForwardFill", "ZNormalization", "MeanAbsScaling"],
        leads_to=["SeasonalFeatures"],
    )
    grammar.register(
        PeriodDetectSpectralToken(),
        follows=["START", "LinearFill", "ForwardFill", "ZNormalization", "MeanAbsScaling"],
        leads_to=["SeasonalFeatures"],
    )
    grammar.register(
        PeriodDetectBICToken(),
        follows=["START", "LinearFill", "ForwardFill", "ZNormalization", "MeanAbsScaling"],
        leads_to=["SeasonalFeatures"],
    )
    grammar.register(
        SeasonalFeaturesToken(),
        follows=["PeriodDetect", "PeriodDetectSpectral", "PeriodDetectBIC"],
        leads_to=["kernel_rbf", "rf_tabular", "lightgbm_tabular", "parrot",
                  "versatile_gb", "step_regression"],
    )
    grammar.register(
        StepRegressionToken(),
        follows=["ZNormalization", "MeanAbsScaling", "FourierFeatures",
                 "SeasonalFeatures", "PeriodDetect", "PeriodDetectSpectral", "PeriodDetectBIC"],
        leads_to=["parrot", "STOP"],
    )
    return grammar

__all__ = [
    "PeriodDetectToken",
    "PeriodDetectSpectralToken",
    "PeriodDetectBICToken",
    "SeasonalFeaturesToken",
    "StepRegressionToken",
    "collect_step_covariates",
    "GBLevelForecastToken",
    "VersatileGradientBoostingToken",
    "VersatileTabularToken",
    "register_versatile_tokens",
    "register_flair_gb_swap",
    "FlairPreprocessToken",
    "FlairRidgeLevelToken",
    "FlairSamplePathsToken",
    "FourierFeaturesToken",
    "KernelRBFToken",
    "LevelBoxCoxCenterToken",
    "LevelShapeRidgeToken",
    "SeasonalFoldToken",
    "LightGBMTabularToken",
    "MeanAbsScalingToken",
    "PeriodPhaseOneHotToken",
    "RandomForestTabularToken",
    "ParrotToken",
    "ParrotDatasetToken",
    "ZNormalizationToken",
    "register_default_tokens",
    "register_flair_tokens",
    "register_seasonal_tokens",
]
