"""Token library — import and register into a Grammar."""

from .cleaning import CleanIdentity, CleanDetrend, CleanMovingAvg
from .features import FeatRaw, FeatFFTEncode, FeatFFTDecode
from .models import (ModelKernelRBF, ModelRandomForest, ModelXGBoost,
                     StopToken, compute_mase)


def register_all(grammar):
    """Register every token into the grammar with default edges."""

    # ── Cleaning: follows START ──
    grammar.register(CleanIdentity(),  follows=["START"])
    grammar.register(CleanDetrend(),   follows=["START"])
    grammar.register(CleanMovingAvg(), follows=["START"])

    # ── Feature extraction: follows cleaning ──
    cleaning_names = ["identity", "detrend", "moving_avg"]
    for cl in cleaning_names:
        grammar.register(FeatRaw(),       follows=[cl], leads_to=[])
        grammar.register(FeatFFTEncode(), follows=[cl], leads_to=[])

    # ── Decoder: follows models, leads to STOP ──
    grammar.register(FeatFFTDecode(), follows=[], leads_to=["STOP"])

    # ── Models: follow features and other models (residual chaining) ──
    model_follows = ["feat_raw", "fft_encode",
                     "kernel_rbf", "random_forest", "xgboost"]
    model_leads   = ["kernel_rbf", "random_forest", "xgboost",
                     "fft_decode", "STOP"]

    grammar.register(ModelKernelRBF(),    follows=model_follows, leads_to=model_leads)
    grammar.register(ModelRandomForest(), follows=model_follows, leads_to=model_leads)
    grammar.register(ModelXGBoost(),      follows=model_follows, leads_to=model_leads)

    # ── STOP ──
    grammar.register(StopToken(), follows=[], leads_to=[])

    return grammar
