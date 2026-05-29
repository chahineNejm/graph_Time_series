"""Built-in token block registry."""

from __future__ import annotations

from .kernel_rbf import KernelRBFToken
from .normalization import ZNormalizationToken


def register_default_tokens(grammar):
    """Register the current built-in blocks without changing their behavior."""

    grammar.register(
        ZNormalizationToken(),
        follows=["START"],
        leads_to=["kernel_rbf"],
    )
    grammar.register(
        KernelRBFToken(),
        follows=["ZNormalization", "kernel_rbf"],
        leads_to=["kernel_rbf", "STOP"],
    )
    return grammar


__all__ = [
    "KernelRBFToken",
    "ZNormalizationToken",
    "register_default_tokens",
]
