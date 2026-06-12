"""Built-in token block registry."""

from __future__ import annotations

from .flair import (
    FlairPreprocessToken,
    FlairRidgeLevelToken,
    FlairSamplePathsToken,
    LevelBoxCoxCenterToken,
    LevelShrinkageToken,
    PeriodFoldToken,
    PeriodSelectionToken,
    SecondaryLevelSeasonalityToken,
    ShapeLevelToken,
)
from .fourier import FourierFeaturesToken
from .kernel_rbf import KernelRBFToken
from .normalization import MeanAbsScalingToken, ZNormalizationToken
from .periodic import PeriodPhaseOneHotToken
from .tabular_models import LightGBMTabularToken, RandomForestTabularToken
from .versatile import (
    DayOfWeekFeatureToken,
    GBLevelForecastToken,
    VersatileGradientBoostingToken,
    VersatileRandomForestToken,
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
        leads_to=["PeriodSelection"],
    )
    grammar.register(
        PeriodSelectionToken(),
        follows=["FlairPreprocess"],
        leads_to=["PeriodPhaseOneHot", "PeriodFold"],
    )
    grammar.register(
        PeriodPhaseOneHotToken(),
        follows=["PeriodSelection"],
        leads_to=["PeriodFold"],
    )
    grammar.register(
        PeriodFoldToken(),
        follows=["PeriodSelection", "PeriodPhaseOneHot"],
        leads_to=["LevelShrinkage", "ShapeLevel"],
    )
    grammar.register(
        LevelShrinkageToken(),
        follows=["PeriodFold"],
        leads_to=["ShapeLevel"],
    )
    grammar.register(
        ShapeLevelToken(),
        follows=["PeriodFold", "LevelShrinkage"],
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
        "START", "ZNormalization", "MeanAbsScaling",
        "FourierFeatures", "DayOfWeekFeature",
    ]
    grammar.register(
        DayOfWeekFeatureToken(period=24),
        follows=feature_sources,
        leads_to=["FourierFeatures", "versatile_rf", "versatile_gb"],
    )
    grammar.register(
        VersatileRandomForestToken(),
        follows=feature_sources + ["versatile_rf", "versatile_gb"],
        leads_to=["STOP", "versatile_rf", "versatile_gb"],
    )
    grammar.register(
        VersatileGradientBoostingToken(),
        follows=feature_sources + ["versatile_rf", "versatile_gb"],
        leads_to=["STOP", "versatile_rf", "versatile_gb"],
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
        follows=["ShapeLevel", "LevelShrinkage"],
        leads_to=["STOP"],
    )
    return grammar


__all__ = [
    "DayOfWeekFeatureToken",
    "GBLevelForecastToken",
    "VersatileGradientBoostingToken",
    "VersatileRandomForestToken",
    "VersatileTabularToken",
    "register_versatile_tokens",
    "register_flair_gb_swap",
    "FlairPreprocessToken",
    "FlairRidgeLevelToken",
    "FlairSamplePathsToken",
    "FourierFeaturesToken",
    "KernelRBFToken",
    "LevelBoxCoxCenterToken",
    "LevelShrinkageToken",
    "LightGBMTabularToken",
    "MeanAbsScalingToken",
    "PeriodFoldToken",
    "PeriodPhaseOneHotToken",
    "PeriodSelectionToken",
    "RandomForestTabularToken",
    "SecondaryLevelSeasonalityToken",
    "ShapeLevelToken",
    "ZNormalizationToken",
    "register_default_tokens",
    "register_flair_tokens",
]