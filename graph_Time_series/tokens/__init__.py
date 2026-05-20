"""Token library - import and register into a Grammar."""

from .cleaning import (CleanIdentity, CleanDetrend, CleanMovingAvg,
                       CleanNormalize, CleanDetrendNorm)
from .features import FeatRaw, FeatFFTEncode, FeatLagFeatures
from .models import (ModelKernelRBF, ModelRandomForest, ModelXGBoost,
                     StopToken, compute_mase)


def register_all(grammar):
    """Register every token into the grammar with default edges."""

    # Normalization is mandatory: it is the only cleaning token after START.
    grammar.register(CleanNormalize(),   follows=["START"])
    grammar.register(CleanIdentity(),    follows=["normalize", "detrend", "moving_avg"])
    grammar.register(CleanDetrend(),     follows=["normalize", "identity", "moving_avg"])
    grammar.register(CleanMovingAvg(),   follows=["normalize", "identity", "detrend"])

    # Feature extraction: follows cleaning and can refresh model_input after a model.
    cleaning_names = ["identity", "detrend", "moving_avg", "normalize"]
    feature_names = ["feat_raw", "fft_encode", "feat_lag"]
    model_names = ["kernel_rbf", "random_forest", "xgboost"]
    feature_follows = cleaning_names + model_names
    for prev in feature_follows:
        grammar.register(FeatRaw(),         follows=[prev], leads_to=[])
        grammar.register(FeatFFTEncode(),   follows=[prev], leads_to=[])
        grammar.register(FeatLagFeatures(), follows=[prev], leads_to=[])

    # Models: follow features and other models. Repeated models fit residuals.
    model_follows = feature_names + model_names
    model_leads   = feature_names + model_names + ["STOP"]

    grammar.register(ModelKernelRBF(),    follows=model_follows, leads_to=model_leads)
    grammar.register(ModelRandomForest(), follows=model_follows, leads_to=model_leads)
    grammar.register(ModelXGBoost(),      follows=model_follows, leads_to=model_leads)

    # STOP
    grammar.register(StopToken(), follows=[], leads_to=[])

    return grammar
