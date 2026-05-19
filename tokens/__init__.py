"""Token library - import and register into a Grammar."""

from .cleaning import (CleanIdentity, CleanDetrend, CleanMovingAvg,
                       CleanNormalize, CleanDetrendNorm)
from .features import FeatRaw, FeatFFTEncode, FeatLagFeatures
from .models import (ModelKernelRBF, ModelRandomForest, ModelXGBoost,
                     StopToken, compute_mase)


def register_all(grammar):
    """Register every token into the grammar with default edges."""

    # Cleaning: follows START
    grammar.register(CleanIdentity(),    follows=["START"])
    grammar.register(CleanDetrend(),     follows=["START"])
    grammar.register(CleanMovingAvg(),   follows=["START"])
    grammar.register(CleanNormalize(),   follows=["START"])
    grammar.register(CleanDetrendNorm(), follows=["START"])

    # Feature extraction: follows cleaning
    cleaning_names = ["identity", "detrend", "moving_avg",
                      "normalize", "detrend_norm"]
    for cl in cleaning_names:
        grammar.register(FeatRaw(),         follows=[cl], leads_to=[])
        grammar.register(FeatFFTEncode(),   follows=[cl], leads_to=[])
        grammar.register(FeatLagFeatures(), follows=[cl], leads_to=[])

    # Models: follow features and other models (residual chaining)
    model_follows = ["feat_raw", "fft_encode", "feat_lag",
                     "kernel_rbf", "random_forest", "xgboost"]
    model_leads   = ["kernel_rbf", "random_forest", "xgboost", "STOP"]

    grammar.register(ModelKernelRBF(),    follows=model_follows, leads_to=model_leads)
    grammar.register(ModelRandomForest(), follows=model_follows, leads_to=model_leads)
    grammar.register(ModelXGBoost(),      follows=model_follows, leads_to=model_leads)

    # STOP
    grammar.register(StopToken(), follows=[], leads_to=[])

    return grammar
