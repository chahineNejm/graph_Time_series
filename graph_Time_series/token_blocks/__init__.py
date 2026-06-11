"""Built-in token block registry."""

from __future__ import annotations

from .bindings import (
    BindAllSafeTabularToken,
    BindFeatureToken,
    BindScaledHistoryToken,
    StackFeatureBundleToken,
)
from .fourier import FourierFeaturesToken
from .kernel_rbf import KernelRBFToken
from .normalization import MeanAbsScalingToken, ZNormalizationToken
from .periodic import PeriodPhaseOneHotToken
from .tabular_models import LightGBMTabularToken, RandomForestTabularToken


def register_default_tokens(grammar):
    """Register the current built-in blocks without changing their behavior."""

    grammar.register(
        ZNormalizationToken(),
        follows=["START"],
        leads_to=["BindScaledHistory", "FourierFeatures", "kernel_rbf"],
    )
    grammar.register(
        MeanAbsScalingToken(),
        follows=["START"],
        leads_to=["BindScaledHistory", "FourierFeatures", "kernel_rbf"],
    )
    grammar.register(
        FourierFeaturesToken(),
        follows=["START", "ZNormalization", "MeanAbsScaling"],
        leads_to=["BindAllSafeTabular"],
    )
    grammar.register(
        BindScaledHistoryToken(),
        follows=["ZNormalization", "MeanAbsScaling", "kernel_rbf"],
        leads_to=["kernel_rbf"],
    )
    grammar.register(
        BindAllSafeTabularToken(),
        follows=[
            "ZNormalization",
            "MeanAbsScaling",
            "FourierFeatures",
            "BindScaledHistory",
            "kernel_rbf",
        ],
        leads_to=["kernel_rbf", "rf_tabular", "lightgbm_tabular"],
    )
    grammar.register(
        KernelRBFToken(),
        follows=[
            "ZNormalization",
            "MeanAbsScaling",
            "BindScaledHistory",
            "BindAllSafeTabular",
            "kernel_rbf",
        ],
        leads_to=["BindScaledHistory", "BindAllSafeTabular", "kernel_rbf", "STOP"],
    )
    grammar.register(
        RandomForestTabularToken(),
        follows=["BindAllSafeTabular"],
        leads_to=["STOP"],
    )
    grammar.register(
        LightGBMTabularToken(),
        follows=["BindAllSafeTabular"],
        leads_to=["STOP"],
    )
    return grammar


__all__ = [
    "BindAllSafeTabularToken",
    "BindFeatureToken",
    "BindScaledHistoryToken",
    "FourierFeaturesToken",
    "KernelRBFToken",
    "LightGBMTabularToken",
    "MeanAbsScalingToken",
    "PeriodPhaseOneHotToken",
    "RandomForestTabularToken",
    "StackFeatureBundleToken",
    "ZNormalizationToken",
    "register_default_tokens",
]
