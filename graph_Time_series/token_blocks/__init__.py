"""Built-in token block registry."""

from __future__ import annotations

from .flair import (
    FlairPreprocessToken,
    FlairRidgeLevelToken,
    FlairSamplePathsToken,
    LevelBoxCoxCenterToken,
    PeriodFoldToken,
    SecondaryLevelSeasonalityToken,
    ShapeLevelToken,
)
from .fourier import FourierFeaturesToken
from .kernel_rbf import KernelRBFToken
from .normalization import MeanAbsScalingToken, ZNormalizationToken
from .periodic import PeriodPhaseOneHotToken
from .seasonal import PeriodDetectToken, SeasonalFeaturesToken
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
    features = ["FourierFeatures"]
    models = ["kernel_rbf", "rf_tabular", "lightgbm_tabular"]

    grammar.register(ZNormalizationToken(), follows=["START"],
                     leads_to=features + models)
    grammar.register(MeanAbsScalingToken(), follows=["START"],
                     leads_to=features + models)
    grammar.register(FourierFeaturesToken(),
                     follows=["START"] + scalers,
                     leads_to=features + models)
    grammar.register(KernelRBFToken(),
                     follows=scalers + features + ["kernel_rbf"],
                     leads_to=["kernel_rbf", "STOP"])
    grammar.register(RandomForestTabularToken(),
                     follows=scalers + features,
                     leads_to=["STOP"])
    grammar.register(LightGBMTabularToken(),
                     follows=scalers + features,
                     leads_to=["STOP"])
    return grammar


def register_flair_tokens(grammar):
    """Register the experimental FLAIR-style branch.

    This is separate from ``register_default_tokens`` so routine searches do not
    become wider unless the caller explicitly opts in.
    """

    grammar.register(
        FlairPreprocessToken(),
        follows=["START"],
        leads_to=["PeriodDetect"],
    )
    grammar.register(
        PeriodDetectToken(),
        follows=["FlairPreprocess"],
        leads_to=["PeriodPhaseOneHot", "PeriodFold"],
    )
    grammar.register(
        PeriodPhaseOneHotToken(),
        follows=["PeriodDetect"],
        leads_to=["PeriodFold"],
    )
    grammar.register(
        PeriodFoldToken(),
        follows=["PeriodDetect", "PeriodPhaseOneHot"],
        leads_to=["ShapeLevel"],
    )
    grammar.register(
        ShapeLevelToken(),
        follows=["PeriodFold"],
        leads_to=["SecondaryLevelSeasonality"],
    )
    grammar.register(
        SecondaryLevelSeasonalityToken(),
        follows=["ShapeLevel"],
        leads_to=["LevelBoxCoxCenter"],
    )
    grammar.register(
        LevelBoxCoxCenterToken(),
        follows=["SecondaryLevelSeasonality"],
        leads_to=["FlairRidgeLevel"],
    )
    grammar.register(
        FlairRidgeLevelToken(),
        follows=["LevelBoxCoxCenter"],
        leads_to=["FlairSamplePaths", "STOP"],
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
        "START", "ZNormalization", "MeanAbsScaling", "FourierFeatures",
    ]
    grammar.register(
        VersatileGradientBoostingToken(),
        follows=feature_sources + ["versatile_gb"],
        leads_to=["STOP", "versatile_gb"],
    )
    return grammar


def register_flair_gb_swap(grammar):
    """Register the gradient-boosted period-level forecaster as a FLAIR step.

    Slots in after ShapeLevel exactly where FlairRidgeLevel would, showing a
    FLAIR sub-step (predicting the period scale) being served by a different
    learner.
    """
    grammar.register(
        GBLevelForecastToken(),
        follows=["ShapeLevel"],
        leads_to=["STOP"],
    )
    return grammar


def register_seasonal_tokens(grammar):
    """Period detector + seasonal sinusoid features (auto-bundle friendly)."""
    grammar.register(
        PeriodDetectToken(),
        follows=["START", "ZNormalization", "MeanAbsScaling"],
        leads_to=["SeasonalFeatures"],
    )
    grammar.register(
        SeasonalFeaturesToken(),
        follows=["PeriodDetect"],
        leads_to=["kernel_rbf", "rf_tabular", "lightgbm_tabular",
                  "versatile_gb", "step_regression"],
    )
    grammar.register(
        StepRegressionToken(),
        follows=["ZNormalization", "MeanAbsScaling", "FourierFeatures",
                 "SeasonalFeatures", "PeriodDetect"],
        leads_to=["STOP"],
    )
    return grammar


__all__ = [
    "PeriodDetectToken",
    "SeasonalFeaturesToken",
    "StepRegressionToken",
    "register_seasonal_tokens",
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
    "LightGBMTabularToken",
    "MeanAbsScalingToken",
    "PeriodFoldToken",
    "PeriodPhaseOneHotToken",
    "RandomForestTabularToken",
    "SecondaryLevelSeasonalityToken",
    "ShapeLevelToken",
    "ZNormalizationToken",
    "register_default_tokens",
    "register_flair_tokens",
]