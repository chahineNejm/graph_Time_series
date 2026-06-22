"""Pipeline inspection toolkit for the troubleshooting notebook.

Keeps the notebook compact: all the machinery (grammar building, token factory,
state snapshots, per-step diff reports, plotting) lives here so the notebook can
read like a pipeline -- set a SEQUENCE, call ``inspect_pipeline``, watch the
effect on ``State`` at each step.

Typical use in a notebook:

    from pipeline_inspect import inspect_pipeline, token_table
    state, states = inspect_pipeline(SEQUENCE, H, F, seasonal_period=48)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

try:
    from IPython.display import display
except Exception:  # pragma: no cover
    display = print

from graph_Time_series.state import State
from graph_Time_series.grammar import Grammar
from graph_Time_series.token_blocks import (
    FlairPreprocessToken, FlairRidgeLevelToken,
    FlairSamplePathsToken, FourierFeaturesToken, GBLevelForecastToken,
    KernelRBFToken, LevelBoxCoxCenterToken, LevelShapeRidgeToken,
    LightGBMTabularToken, MeanAbsScalingToken,
    PeriodPhaseOneHotToken, RandomForestTabularToken,
    SeasonalFoldToken, ParrotToken, ParrotDatasetToken,
    VersatileGradientBoostingToken,
    ZNormalizationToken, register_default_tokens, register_flair_gb_swap,
    register_flair_tokens, register_versatile_tokens,
    PeriodDetectToken, PeriodDetectBICToken, PeriodDetectSpectralToken,
    ForwardFillToken, LinearFillToken,
    SeasonalFeaturesToken, StepRegressionToken,
    register_seasonal_tokens,
)
from graph_Time_series.token_blocks.affine_fold import (
    AffineSeasonalFoldToken, AffineLevelForecastToken, register_affine_tokens,
)
from graph_Time_series.token_blocks.window_kernel import (
    WindowKernelToken, register_window_kernel,
)

__all__ = [
    "build_grammar", "make_token_factory", "instantiate_tokens", "token_table",
    "shape_of", "snapshot", "describe_inputs", "print_step_report",
    "summarize_state", "signal_board_df", "feature_bundle_df", "artifact_df",
    "plot_raw_data", "plot_array_artifact", "plot_intermediate_artifacts",
    "plot_prediction_state", "plot_decode", "score_prediction", "valid_next_from_grammar",
    "run_sequence", "inspect_pipeline",
]


# --------------------------------------------------------------------------
# Grammar + tokens
# --------------------------------------------------------------------------

def build_grammar(include_flair=True, include_versatile=True, include_flair_gb=True,
                  include_seasonal=True, include_affine=True, include_window_kernel=True):
    grammar = register_default_tokens(Grammar())
    if include_flair:
        grammar = register_flair_tokens(grammar)
    if include_versatile:
        grammar = register_versatile_tokens(grammar)
    if include_flair_gb:
        grammar = register_flair_gb_swap(grammar)
    if include_seasonal:
        grammar = register_seasonal_tokens(grammar)
    if include_affine:
        grammar = register_affine_tokens(grammar)
    if include_window_kernel:
        grammar = register_window_kernel(grammar)
    return grammar


def make_token_factory(seasonal_period=48):
    """Fresh token instances per run. Edit hyperparameters here."""
    return {
        "ZNormalization": lambda: ZNormalizationToken(),
        "MeanAbsScaling": lambda: MeanAbsScalingToken(),
        "FourierFeatures": lambda: FourierFeaturesToken(n_harmonics=8),
        "kernel_rbf": lambda: KernelRBFToken(alpha=1e-2, median_subset=24, seed=0),
        "parrot": lambda: ParrotToken(),
        "parrot_dataset": lambda: ParrotDatasetToken(
            min_corr=0.97,
            max_candidates=10000,
            show_progress=True,
            progress_min_samples=16,
        ),
        "affine_fold": lambda: AffineSeasonalFoldToken(),
        "affine_forecast": lambda: AffineLevelForecastToken(),
        "window_kernel": lambda: WindowKernelToken(
            gamma="auto", target_meff=8.0, top_k_per_series=1,
            max_series=32, space="value", seed=0,
        ),
        "rf_tabular": lambda: RandomForestTabularToken(n_estimators=80, max_depth=10, min_samples_leaf=2, seed=0, n_jobs=-1),
        "lightgbm_tabular": lambda: LightGBMTabularToken(n_estimators=160, learning_rate=0.05, num_leaves=31, seed=0, n_jobs=-1),
        "FlairPreprocess": lambda: FlairPreprocessToken(),
        "PeriodPhaseOneHot": lambda: PeriodPhaseOneHotToken(),
        "SeasonalFold": lambda: SeasonalFoldToken(),
        "LevelBoxCoxCenter": lambda: LevelBoxCoxCenterToken(),
        "level_shape_ridge": lambda: LevelShapeRidgeToken(show_progress=True, progress_min_samples=16),
        "FlairRidgeLevel": lambda: FlairRidgeLevelToken(show_progress=True, progress_min_samples=16),
        "FlairSamplePaths": lambda: FlairSamplePathsToken(n_paths=50, show_progress=True, progress_min_samples=8),
        "PeriodDetect": lambda: PeriodDetectToken(freq="H"),
        "PeriodDetectBIC": lambda: PeriodDetectBICToken(freq="H"),
        "PeriodDetectSpectral": lambda: PeriodDetectSpectralToken(freq="H"),
        "LinearFill": lambda: LinearFillToken(),
        "ForwardFill": lambda: ForwardFillToken(),
        "SeasonalFeatures": lambda: SeasonalFeaturesToken(n_harmonics=2),
        "step_regression": lambda: StepRegressionToken(per_series=True, n_estimators=60),
        "versatile_gb": lambda: VersatileGradientBoostingToken(max_iter=80, learning_rate=0.06, seed=0),
        "gb_level_forecast": lambda: GBLevelForecastToken(n_lags=3, max_iter=80, learning_rate=0.08, seed=0),
    }


def instantiate_tokens(sequence, factory=None, seasonal_period=48):
    factory = factory or make_token_factory(seasonal_period)
    missing = [name for name in sequence if name not in factory]
    if missing:
        raise KeyError(f"Unknown token(s): {missing}. Available: {sorted(factory)}")
    return {name: factory[name]() for name in sorted(set(sequence))}


def token_table(seasonal_period=48):
    rows = []
    for name, make in make_token_factory(seasonal_period).items():
        tok = make()
        rows.append({
            "token": name, "class": getattr(tok, "token_class", ""),
            "learning_scope": getattr(tok, "learning_scope", "n/a"),
            "reads": ", ".join(getattr(tok, "reads", ()) or ()),
            "writes": ", ".join(getattr(tok, "writes", ()) or ()),
            "description": getattr(tok, "description", ""),
        })
    return pd.DataFrame(rows).sort_values(["class", "token"]).reset_index(drop=True)


# --------------------------------------------------------------------------
# State snapshots / diffs
# --------------------------------------------------------------------------

def shape_of(value):
    return tuple(value.shape) if hasattr(value, "shape") else type(value).__name__

def _shapes(store):
    return {k: shape_of(v) for k, v in store.items()}

def snapshot(state):
    return {
        "historical_features": _shapes(state.historical_features),
        "future_features": _shapes(state.future_features),
        "features": {k: shape_of(v) for k, v in state.features.items()},
        "board": {s.name: f"{s.sem}{tuple(s.axes)} @{s.space.id}" for s in state.board},
        "space": state.current_space.id,
        "n_models": state.n_models_applied,
        "transforms": [t.name for t in state.transforms],
        "flags": dict(state.flags),
        "metadata": sorted(k for k in state.metadata
                           if k not in {"artifacts", "cumulative_prediction",
                                        "current_residual", "last_prediction",
                                        "input_bundles", "active_input_bundle"}),
        "residual_l2": float(np.linalg.norm(np.asarray(state.current_target, float))),
    }

def _diff_keys(before, after):
    added = {k: after[k] for k in after if k not in before}
    removed = {k: before[k] for k in before if k not in after}
    changed = {k: (before[k], after[k]) for k in after
               if k in before and before[k] != after[k]}
    return added, removed, changed

def describe_inputs(token, state):
    """What the token will actually consume from the current State."""
    lines = []
    reads = getattr(token, "reads", ()) or ()
    reads = (reads,) if isinstance(reads, str) else tuple(reads)
    for r in reads:
        where = None
        for store in ("historical_features", "future_features", "features", "metadata"):
            d = getattr(state, store, {})
            if r in d:
                where = f"{r}  {shape_of(d[r])}  [{store}]"
                break
        lines.append("reads  " + (where or f"{r}  <MISSING>"))
    for p in getattr(token, "requires", ()) or ():
        sp = p.space if getattr(p, "space", None) in ("current", "raw", "any") else "any"
        align = None if getattr(p, "alignment", "history") == "any" else getattr(p, "alignment", "history")
        try:
            matches = state.query(sem=p.sem, axes=p.axes, alignment=align, space=sp)
        except Exception:
            matches = []
        tag = f"port {p.sem}{tuple(p.axes) if p.axes else ''} @{p.space}"
        if matches:
            lines.append(tag + " <- " + ", ".join(f"{m.name}{shape_of(m.value)}" for m in matches[:6]))
        elif getattr(p, "coerce", False):
            try:
                b = state.feature_bundle()
                lines.append(tag + f" <- auto-bundle {b.matrix.shape}: "
                             + ", ".join(f"{bl['name']}({bl['cols']})" for bl in b.blocks))
            except Exception as exc:
                lines.append(tag + f" <- UNSATISFIED ({exc!r})")
        else:
            lines.append(tag + ("  (optional, none)" if getattr(p, "optional", False) else "  <- UNSATISFIED"))
    return lines

def _short(v, maxlen=140):
    if isinstance(v, np.ndarray):
        return f"ndarray{tuple(v.shape)}"
    if isinstance(v, dict):
        v = {k: (f"ndarray{tuple(x.shape)}" if isinstance(x, np.ndarray) else x)
             for k, x in v.items()}
    text = repr(v)
    return text if len(text) <= maxlen else text[:maxlen] + " ..."


def print_step_report(step_idx, name, token, before, after, grammar=None):
    bs, as_ = snapshot(before), snapshot(after)
    print("\n" + "-" * 78)
    print(f"STEP {step_idx}  -  {name}   [{token.token_class}]")
    desc = getattr(token, "description", "")
    if desc:
        print("  " + desc)
    print("  > TAKES:")
    inp = describe_inputs(token, before)
    for ln in (inp or ["(no declared inputs)"]):
        print("      " + ln)
    print("  > IMPACT on state:")
    changed = False
    for store in ("historical_features", "future_features", "features"):
        add, rem, chg = _diff_keys(bs[store], as_[store])
        for k, v in add.items():
            print(f"      + {store}: {k} {v}"); changed = True
        for k, (b, a) in chg.items():
            print(f"      ~ {store}: {k} {b} -> {a}"); changed = True
        for k in rem:
            print(f"      - {store}: {k}"); changed = True
    b_add, _, _ = _diff_keys(bs["board"], as_["board"])
    after_sig = {sig.name: sig for sig in after.board}
    for k, v in b_add.items():
        sig = after_sig.get(k)
        if sig is not None and sig.sem.startswith("param"):
            print(f"      + signal: {k}  [{v}] = {_short(sig.value)}")
        else:
            print(f"      + signal: {k}  [{v}]")
        changed = True
    if bs["space"] != as_["space"]:
        print(f"      space: {bs['space']} -> {as_['space']}"); changed = True
    if bs["transforms"] != as_["transforms"]:
        new_t = [t for t in as_["transforms"] if t not in bs["transforms"]]
        print(f"      + transform(s): {new_t}  (inverse auto-applied at decode)"); changed = True
    new_flags = [k for k in as_["flags"] if k not in bs["flags"]]
    if new_flags:
        print(f"      + flags: {new_flags}"); changed = True
    new_meta = [k for k in as_["metadata"] if k not in bs["metadata"]]
    for k in new_meta:
        print(f"      + metadata: {k} = {_short(after.metadata[k])}"); changed = True
    if as_["n_models"] != bs["n_models"]:
        last = np.asarray(after.last_prediction())
        print(f"      + prediction pushed: {last.shape}   n_models {bs['n_models']} -> {as_['n_models']}")
        print(f"      residual L2: {bs['residual_l2']:.4g} -> {as_['residual_l2']:.4g}")
        changed = True
    if not changed:
        print("      (no observable change)")
    if grammar is not None:
        print("  > valid next:", valid_next_from_grammar(grammar, after))


# --------------------------------------------------------------------------
# Tables + plots
# --------------------------------------------------------------------------

def summarize_state(state):
    print("tokens:", state.token_sequence)
    print("current_space:", state.current_space.id, "| n_models:", state.n_models_applied)
    print("historical_features:", _shapes(state.historical_features))
    print("future_features:", _shapes(state.future_features))
    print("metadata keys:", sorted(k for k in state.metadata
          if k not in {"artifacts", "cumulative_prediction", "current_residual", "last_prediction"}))

def signal_board_df(state):
    return pd.DataFrame([{
        "name": s.name, "sem": s.sem, "axes": s.axes, "alignment": s.alignment,
        "space": s.space.id, "shape": shape_of(s.value), "source": s.source,
        "tags": ",".join(sorted(s.tags)),
    } for s in state.board])

def feature_bundle_df(state):
    try:
        bundle = state.feature_bundle()
    except Exception as exc:
        print("feature_bundle unavailable:", repr(exc)); return None
    print("feature_bundle matrix:", bundle.matrix.shape, "space:", bundle.space.id)
    return pd.DataFrame(bundle.blocks)

def artifact_df(state):
    rows = state.describe_artifacts()
    return pd.DataFrame(rows) if rows else pd.DataFrame()

def plot_raw_data(H, F, sample_idx=0):
    if plt is None:
        return
    hist_x = np.arange(H.shape[1]); fut_x = np.arange(H.shape[1], H.shape[1] + F.shape[1])
    plt.figure(figsize=(12, 4))
    plt.plot(hist_x, H[sample_idx], label="history")
    plt.plot(fut_x, F[sample_idx], label="holdout future", color="black", linewidth=2)
    plt.axvline(H.shape[1] - 1, color="gray", linestyle="--", linewidth=1)
    plt.title(f"Raw series sample {sample_idx}"); plt.legend(); plt.show()

def plot_array_artifact(name, value, sample_idx=0):
    if plt is None:
        return
    arr = np.asarray(value)
    if arr.ndim == 0:
        return
    plt.figure(figsize=(10, 3))
    if arr.ndim == 1:
        plt.plot(arr)
    elif arr.ndim == 2:
        plt.plot(arr[sample_idx])
    elif arr.ndim == 3:
        plt.imshow(arr[sample_idx], aspect="auto", interpolation="nearest"); plt.colorbar(label=name)
    else:
        plt.plot(arr.reshape(arr.shape[0], -1)[sample_idx])
    plt.title(f"{name} shape={arr.shape}"); plt.tight_layout(); plt.show()

def plot_intermediate_artifacts(state, sample_idx=0, max_items=6):
    items = []
    for store_name in ["historical_features", "future_features", "features"]:
        for name, value in getattr(state, store_name).items():
            if name == "raw_history" or not hasattr(value, "shape"):
                continue
            items.append((f"{store_name}.{name}", value))
    for name, value in items[-max_items:]:
        plot_array_artifact(name, value, sample_idx=sample_idx)

def plot_prediction_state(state, sample_idx=0, title=""):
    if plt is None:
        return
    hist = state.original_history[sample_idx]; true_future = state.original_future[sample_idx]
    hist_x = np.arange(hist.shape[0]); fut_x = np.arange(hist.shape[0], hist.shape[0] + true_future.shape[0])
    plt.figure(figsize=(12, 4))
    plt.plot(hist_x, hist, label="history")
    plt.plot(fut_x, true_future, label="holdout future", color="black", linewidth=2)
    if state.n_models_applied > 0:
        plt.plot(fut_x, state.get_final_prediction()[sample_idx], label="decoded prediction", linewidth=2)
    plt.axvline(hist.shape[0] - 1, color="gray", linestyle="--", linewidth=1)
    plt.title(title or "Prediction state"); plt.legend(); plt.show()

def plot_decode(state, sample_idx=0):
    """Visualise the inverse-transform (e.g. un-z-normalize) at decode time.

    Left: the cumulative prediction in the MODEL's scale (current space) vs the
    scaled target. Right: the same prediction after the inverse transform chain
    (get_final_prediction) vs the original future. The gap between panels is
    exactly the rescaling that the transform stack undoes at the end.
    """
    if plt is None or state.n_models_applied == 0:
        print("Need matplotlib and at least one model prediction.")
        return
    i = sample_idx
    scaled_pred = np.asarray(state.cumulative_prediction())[i]
    scaled_true = np.asarray(state.active_target_base)[i]
    decoded_pred = np.asarray(state.get_final_prediction())[i]
    raw_true = np.asarray(state.original_future)[i]
    chain = [t.name for t in state.transforms] or ["(none)"]
    H = np.arange(scaled_pred.shape[0])

    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    ax[0].plot(H, scaled_true, color="black", lw=1.5, label="target (scaled)")
    ax[0].plot(H, scaled_pred, "--", color="tab:blue", lw=1.5, label="prediction (scaled)")
    ax[0].set_title(f"model scale: {state.current_space.id}")
    ax[0].legend(fontsize=8)
    ax[1].plot(H, raw_true, color="black", lw=1.5, label="future (original)")
    ax[1].plot(H, decoded_pred, "--", color="tab:orange", lw=1.5, label="decoded prediction")
    ax[1].set_title("original scale  (after inverse: " + " -> ".join(reversed(chain)) + ")")
    ax[1].legend(fontsize=8)
    fig.suptitle(f"Decode / rescale - sample {i}")
    plt.tight_layout(); plt.show()


def score_prediction(state, seasonal_period=1):
    """Decoded-forecast metrics, incl. MASE.

    MASE scales each series' MAE by its in-sample naive error
    mean|y_t - y_{t-m}| on the history (m = seasonal_period, default 1 =
    random-walk naive). MASE < 1 beats that naive baseline.
    """
    if state.n_models_applied == 0:
        return None
    pred = np.asarray(state.get_final_prediction(), float)
    F = np.asarray(state.original_future, float)
    Hh = np.asarray(state.original_history, float)
    err = pred - F
    m = max(1, int(seasonal_period))
    mae_series = np.mean(np.abs(err), axis=1)
    if Hh.shape[1] > m:
        scale = np.mean(np.abs(Hh[:, m:] - Hh[:, :-m]), axis=1)
    else:
        scale = np.zeros(Hh.shape[0])
    # Series whose naive error is ~0 (flat/constant history) are EXCLUDED from
    # the average (set to NaN), so they neither blow it up nor drag it to 0.
    ok = scale > 1e-12
    per_series = np.where(ok, mae_series / np.where(ok, scale, 1.0), np.nan)
    mase = float(np.nanmean(per_series)) if ok.any() else float("nan")
    return {"mae": float(np.mean(np.abs(err))),
            "rmse": float(np.sqrt(np.mean(err ** 2))),
            "mase": mase,
            "max_abs_error": float(np.max(np.abs(err)))}


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------

def valid_next_from_grammar(grammar, state):
    try:
        return grammar.valid_actions(state)
    except Exception as exc:
        return [f"grammar.valid_actions failed: {exc!r}"]

def run_sequence(sequence, H, F, *, grammar, factory=None, seasonal_period=48,
                 sample_idx=0, plot_each_step=True, show_signal_board=False,
                 show_feature_bundle=False, show_artifacts=False,
                 max_artifacts_to_plot=6, stop_on_error=True):
    """Apply a sequence one token at a time with a readable per-step report.

    Returns (final_state, [(label, state_copy), ...]).
    """
    tokens = instantiate_tokens(sequence, factory, seasonal_period)
    state = State(H, F)
    states = [("START", state.copy())]

    print("Running:", " -> ".join(sequence) if sequence else "<empty>")
    print("Initial valid next:", valid_next_from_grammar(grammar, state))
    if plot_each_step:
        plot_prediction_state(state, sample_idx=sample_idx, title="START")

    for step_idx, name in enumerate(sequence, start=1):
        token = tokens[name]
        before = state.copy()
        try:
            ok = token.can_apply(state)
        except Exception as exc:
            print(f"\nSTEP {step_idx} - {name}: can_apply raised {exc!r}")
            if stop_on_error:
                raise
            break
        if not ok:
            print("\n" + "-" * 78)
            print(f"STEP {step_idx}  -  {name}   [{token.token_class}]   x cannot apply")
            print("  > wanted:")
            for ln in describe_inputs(token, state):
                print("      " + ln)
            print("  > valid next instead:", valid_next_from_grammar(grammar, state))
            if stop_on_error:
                raise RuntimeError(f"{name} cannot apply after {state.token_sequence}")
            break
        try:
            state = token.apply(state)
        except Exception as exc:
            print(f"\nSTEP {step_idx} - {name}: apply raised {exc!r}")
            if stop_on_error:
                raise
            break

        states.append((name, state.copy()))
        print_step_report(step_idx, name, token, before, state, grammar=grammar)
        if show_signal_board:
            display(signal_board_df(state))
        if show_feature_bundle:
            display(feature_bundle_df(state))
        if show_artifacts:
            display(artifact_df(state))
        if plot_each_step:
            plot_prediction_state(state, sample_idx=sample_idx, title=f"After {name}")
            plot_intermediate_artifacts(state, sample_idx=sample_idx, max_items=max_artifacts_to_plot)

    print("\n" + "=" * 78)
    print("FINAL STATE")
    summarize_state(state)
    metrics = score_prediction(state)
    print("Final holdout metrics:", metrics if metrics is not None else "(no model produced a prediction)")
    return state, states


def inspect_pipeline(sequence, H, F, *, grammar=None, **kwargs):
    """One-call entrypoint: builds the default grammar if needed, then runs."""
    grammar = grammar or build_grammar()
    return run_sequence(sequence, H, F, grammar=grammar, **kwargs)
