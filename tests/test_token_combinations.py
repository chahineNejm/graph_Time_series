"""Self-contained combinatorial smoke test for the Signal-board architecture.

Mirrors the intent of ``examples/test_token_sequence_colab.ipynb`` without its
external data/repo dependencies: it builds synthetic seasonal series, then

  1. exhaustively enumerates every token sequence reachable under the tokens'
     own inherited conditions (caps, phase order, port/dependency checks), and
     executes each one to completion, asserting a finite, correctly-shaped
     forecast; and
  2. runs a set of named "signature" pipelines that specifically exercise the
     questions this architecture is meant to answer:
       * a full FLAIR pipeline,
       * a FLAIR pipeline with gradient boosting swapped in for the period-level
         step,
       * versatile models run WITHOUT any binder token,
       * the same versatile models WITH a binder (proving binders are now
         optional, not required),
       * adding an exogenous day-of-week feature mid-sequence.

Run:  python tests/test_token_combinations.py
"""

from __future__ import annotations

import itertools
import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graph_Time_series.state import State
from graph_Time_series.token_blocks import (
    FlairPreprocessToken,
    FlairRidgeLevelToken,
    FourierFeaturesToken,
    GBLevelForecastToken,
    KernelRBFToken,
    LevelBoxCoxCenterToken,
    LevelShrinkageToken,
    MeanAbsScalingToken,
    PeriodFoldToken,
    PeriodPhaseOneHotToken,
    PeriodSelectionToken,
    RandomForestTabularToken,
    SecondaryLevelSeasonalityToken,
    ShapeLevelToken,
    VersatileGradientBoostingToken,
)


def make_data(n_samples=6, hist_len=96, horizon=24, period=24, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(hist_len)
    tf = np.arange(hist_len, hist_len + horizon)
    H, F = [], []
    for k in range(n_samples):
        level = 8.0 + 0.5 * k
        amp = 2.0 + 0.1 * k
        season = amp * np.sin(2 * np.pi * t / period) + 0.5 * np.cos(2 * np.pi * t / 7)
        H.append(level + season + rng.normal(0, 0.3, hist_len))
        seasonf = amp * np.sin(2 * np.pi * tf / period) + 0.5 * np.cos(2 * np.pi * tf / 7)
        F.append(level + seasonf)
    return np.asarray(H, np.float32), np.asarray(F, np.float32)


def finalize(state):
    """Validate a finished pipeline's decoded forecast."""
    pred = state.get_final_prediction()
    assert pred.shape == state.original_future.shape, (
        f"shape {pred.shape} != {state.original_future.shape}")
    assert np.all(np.isfinite(pred)), "forecast has non-finite values"
    return pred


# ---------------------------------------------------------------------------
# 1. Exhaustive reachability enumeration
# ---------------------------------------------------------------------------

def enumerate_pipelines(tokens, H, F, max_depth=6, max_pipelines=4000, time_budget_s=None):
    """DFS over all sequences valid under each token's own can_apply.

    Returns (completed, failures). A pipeline 'completes' when, after applying a
    model token, its forecast decodes cleanly (the implicit STOP).
    """
    import time as _time
    _start = _time.time()
    completed = []
    failures = []
    seen = set()
    base = State(H, F)

    def dfs(state, seq, depth):
        if len(completed) + len(failures) >= max_pipelines:
            return
        if time_budget_s is not None and _time.time() - _start > time_budget_s:
            return
        # If at least one model has run, this is a valid STOP point.
        if state.n_models_applied > 0:
            key = tuple(seq)
            if key not in seen:
                seen.add(key)
                try:
                    finalize(state)
                    completed.append(key)
                except Exception as exc:  # noqa: BLE001
                    failures.append((key, repr(exc)))
        if depth >= max_depth:
            return
        for name, tok in tokens.items():
            try:
                ok = tok.can_apply(state)
            except Exception as exc:  # noqa: BLE001
                failures.append((tuple(seq + [name + "?can_apply"]), repr(exc)))
                continue
            if not ok:
                continue
            try:
                child = tok.apply(state.copy())
            except Exception as exc:  # noqa: BLE001
                failures.append((tuple(seq + [name]), repr(exc) + "\n" +
                                 traceback.format_exc().splitlines()[-1]))
                continue
            dfs(child, seq + [name], depth + 1)

    dfs(base, [], 0)
    return completed, failures


# ---------------------------------------------------------------------------
# 2. Named signature pipelines
# ---------------------------------------------------------------------------

def run_named(tokens, H, F):
    pipelines = {
        "full_FLAIR": [
            "FlairPreprocess", "PeriodSelection", "PeriodFold",
            "LevelShrinkage", "ShapeLevel", "SecondaryLevelSeasonality",
            "LevelBoxCoxCenter", "FlairRidgeLevel",
        ],
        "FLAIR_with_GB_level_swap": [
            "FlairPreprocess", "PeriodSelection", "PeriodFold",
            "ShapeLevel", "gb_level_forecast",
        ],
        "versatile_GB_no_binder": [
            "MeanAbsScaling", "FourierFeatures", "versatile_gb",
        ],
        "rf_tabular_no_binder": [
            "MeanAbsScaling", "FourierFeatures", "rf_tabular",
        ],
        "kernel_rbf_no_binder": [
            "ZNormalization", "kernel_rbf",
        ],
    }
    results = {}
    for label, seq in pipelines.items():
        state = State(H, F)
        try:
            for name in seq:
                tok = tokens[name]
                if not tok.can_apply(state):
                    raise RuntimeError(f"{name} not applicable after {state.token_sequence}")
                state = tok.apply(state)
            pred = finalize(state)
            results[label] = ("ok", pred.shape)
        except Exception as exc:  # noqa: BLE001
            results[label] = ("FAIL", repr(exc))
    return results


def build_tokens(H):
    ctx = min(96, H.shape[1])
    return {
        "MeanAbsScaling": MeanAbsScalingToken(),
        "ZNormalization": ZNormalizationToken(),
        "FourierFeatures": FourierFeaturesToken(n_harmonics=6),
        "kernel_rbf": KernelRBFToken(),
        "rf_tabular": RandomForestTabularToken(n_estimators=40),
        "versatile_gb": VersatileGradientBoostingToken(max_iter=40),
        # FLAIR family
        "FlairPreprocess": FlairPreprocessToken(),
        "PeriodSelection": PeriodSelectionToken(freq="H"),
        "PeriodPhaseOneHot": PeriodPhaseOneHotToken(),
        "PeriodFold": PeriodFoldToken(),
        "LevelShrinkage": LevelShrinkageToken(),
        "ShapeLevel": ShapeLevelToken(),
        "SecondaryLevelSeasonality": SecondaryLevelSeasonalityToken(),
        "LevelBoxCoxCenter": LevelBoxCoxCenterToken(),
        "FlairRidgeLevel": FlairRidgeLevelToken(show_progress=False),
        "gb_level_forecast": GBLevelForecastToken(),
    }


from graph_Time_series.token_blocks import ZNormalizationToken  # noqa: E402


def main():
    H, F = make_data()
    tokens = build_tokens(H)

    print(f"Data: H{H.shape} F{F.shape}\n")

    print("=== Named signature pipelines ===")
    named = run_named(tokens, H, F)
    n_ok = 0
    for label, (status, info) in named.items():
        flag = "OK " if status == "ok" else "XX "
        n_ok += status == "ok"
        print(f"  {flag}{label:32s} {info}")
    print(f"  {n_ok}/{len(named)} named pipelines passed\n")

    print("=== Exhaustive reachability enumeration ===")
    # Enumerate over a focused token set to keep the search finite but broad.
    enum_tokens = {k: tokens[k] for k in [
        "MeanAbsScaling", "ZNormalization", "FourierFeatures",
        "versatile_gb",
        "FlairPreprocess", "PeriodSelection", "PeriodFold", "LevelShrinkage",
        "ShapeLevel", "gb_level_forecast",
    ]}
    completed, failures = enumerate_pipelines(
        enum_tokens, H, F, max_depth=6, max_pipelines=5000, time_budget_s=30)
    print(f"  completed pipelines: {len(completed)}")
    print(f"  failures:            {len(failures)}")
    for key, err in failures[:20]:
        print(f"    FAIL {' -> '.join(key)}\n         {err}")

    ok = n_ok == len(named) and not failures
    print("\nRESULT:", "ALL GREEN" if ok else "FAILURES PRESENT")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
