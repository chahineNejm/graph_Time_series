"""Built-in token block registry."""

from __future__ import annotations

from .bindings import (
    BindAllSafeTabularToken,
    BindFeatureToken,
    BindScaledHistoryToken,
    StackFeatureBundleToken,
)
from .kernel_rbf import KernelRBFToken
from .normalization import ZNormalizationToken
from .periodic import PeriodPhaseOneHotToken


def register_default_tokens(grammar):
    """Register the current built-in blocks without changing their behavior."""

    grammar.register(
        ZNormalizationToken(),
        follows=["START"],
        leads_to=["BindScaledHistory", "kernel_rbf"],
    )
    grammar.register(
        BindScaledHistoryToken(),
        follows=["ZNormalization", "kernel_rbf"],
        leads_to=["kernel_rbf"],
    )
    grammar.register(
        BindAllSafeTabularToken(),
        follows=["ZNormalization", "BindScaledHistory", "kernel_rbf"],
        leads_to=["kernel_rbf"],
    )
    grammar.register(
        KernelRBFToken(),
        follows=["ZNormalization", "BindScaledHistory", "BindAllSafeTabular", "kernel_rbf"],
        leads_to=["BindScaledHistory", "BindAllSafeTabular", "kernel_rbf", "STOP"],
    )
    return grammar


__all__ = [
    "BindAllSafeTabularToken",
    "BindFeatureToken",
    "BindScaledHistoryToken",
    "KernelRBFToken",
    "PeriodPhaseOneHotToken",
    "StackFeatureBundleToken",
    "ZNormalizationToken",
    "register_default_tokens",
]
