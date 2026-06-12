"""Print every valid token sequence (package tokens), one per line, and export to CSV.

Run from the repo root:  python examples/list_all_sequences.py

The enumeration respects three structural rules on top of each token's own
``can_apply`` so the output is meaningful rather than noisy:

  1. Phase order is non-decreasing: cleaning -> scaling -> feature -> model.
     (A feature can never precede its scaler, a model is always last, etc.)
  2. At most one token per mutually-exclusive group (e.g. only ONE normalizer:
     never ZNormalization AND MeanAbsScaling in the same pipeline).
  3. Within a phase, tokens follow a fixed canonical order, so we emit ONE
     representative per combination instead of every permutation. Real
     dependencies (the FLAIR chain PeriodSelection -> PeriodFold -> ShapeLevel
     -> ...) are encoded in that canonical order and therefore preserved.

Fast by design: model tokens are recorded as terminal steps without being
*fitted*. Set APPLY_MODELS=True to also explore residual model chains
(MAX_MODELS>1); that path actually fits models, so it is slower.
"""

import sys
import time
import csv
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graph_Time_series.state import State
from graph_Time_series.token_blocks import (
    ZNormalizationToken, MeanAbsScalingToken, FourierFeaturesToken,
    KernelRBFToken, RandomForestTabularToken,
    VersatileGradientBoostingToken,
    GBLevelForecastToken, FlairPreprocessToken, PeriodSelectionToken,
    PeriodPhaseOneHotToken, PeriodFoldToken, LevelShrinkageToken,
    ShapeLevelToken, SecondaryLevelSeasonalityToken, LevelBoxCoxCenterToken,
    FlairRidgeLevelToken, FlairSamplePathsToken,
)

# --- tokens to enumerate (edit freely) ---
TOKENS = {
    "ZNormalization": ZNormalizationToken(),
    "MeanAbsScaling": MeanAbsScalingToken(),
    "FourierFeatures": FourierFeaturesToken(n_harmonics=6),
    "kernel_rbf": KernelRBFToken(),
    "rf_tabular": RandomForestTabularToken(n_estimators=8),
    "versatile_gb": VersatileGradientBoostingToken(max_iter=15),
    "FlairPreprocess": FlairPreprocessToken(),
    "PeriodSelection": PeriodSelectionToken(freq="H"),
    "PeriodPhaseOneHot": PeriodPhaseOneHotToken(),
    "PeriodFold": PeriodFoldToken(),
    "LevelShrinkage": LevelShrinkageToken(),
    "ShapeLevel": ShapeLevelToken(),
    "SecondaryLevelSeasonality": SecondaryLevelSeasonalityToken(),
    "LevelBoxCoxCenter": LevelBoxCoxCenterToken(),
    "FlairRidgeLevel": FlairRidgeLevelToken(show_progress=False),
    "FlairSamplePaths": FlairSamplePathsToken(),
    "gb_level_forecast": GBLevelForecastToken(max_iter=15),
}

# Canonical within-phase order (encodes the FLAIR dependency chain). Any token
# not listed falls back to the end of its phase.
CANON_ORDER = [
    # cleaning
    "FlairPreprocess",
    # scaling
    "ZNormalization", "MeanAbsScaling",
    # feature (FLAIR chain first, in dependency order, then generic features)
    "PeriodSelection", "PeriodPhaseOneHot", "PeriodFold", "LevelShrinkage",
    "ShapeLevel", "SecondaryLevelSeasonality", "LevelBoxCoxCenter",
    "FourierFeatures",
    # model
    "gb_level_forecast", "FlairRidgeLevel", "kernel_rbf", "rf_tabular",
    "versatile_gb",
]

# Mutually-exclusive groups: at most one token from each per pipeline.
EXCLUSIVE_GROUPS = [
    {"ZNormalization", "MeanAbsScaling"},   # only one normalizer
]

PHASE_RANK = {"cleaning": 0, "transform": 1, "feature": 2, "binding": 2,
              "model": 3, "control": 4}

# settings
MAX_DEPTH = 9          # longest sequence to explore
MAX_MODELS = 1         # cap on residual model stacking
ALLOW_REPEATS = False
APPLY_MODELS = False   # True -> also explore post-model tokens / model chains (slower)
TIME_BUDGET_S = 60     # always return within this many seconds


def _phase(tok):
    return PHASE_RANK.get(getattr(tok, "token_class", ""), 2)


def _canon(name):
    return CANON_ORDER.index(name) if name in CANON_ORDER else len(CANON_ORDER)


def structurally_allowed(name, seq, tokens):
    """Phase order + canonical within-phase order + mutual exclusivity."""
    ph = _phase(tokens[name])
    if seq:
        if ph < _phase(tokens[seq[-1]]):           # phase must not go backwards
            return False
        same_phase = [s for s in seq if _phase(tokens[s]) == ph]
        if same_phase and _canon(name) <= max(_canon(s) for s in same_phase):
            return False                            # keep one canonical order
    for grp in EXCLUSIVE_GROUPS:
        if name in grp and any(s in grp for s in seq):
            return False
    return True


def seasonal_data(n=5, hist=72, horizon=24, period=24, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(hist); tf = np.arange(hist, hist + horizon)
    H = np.stack([8 + 0.5 * k + 2 * np.sin(2 * np.pi * t / period)
                  + rng.normal(0, 0.3, hist) for k in range(n)]).astype(np.float32)
    F = np.stack([8 + 0.5 * k + 2 * np.sin(2 * np.pi * tf / period)
                  for k in range(n)]).astype(np.float32)
    return H, F


def all_sequences(tokens, H, F):
    base = State(H, F)
    out, seen = [], set()
    start = time.time()

    def record(seq):
        if tuple(seq) not in seen:
            seen.add(tuple(seq)); out.append(list(seq))

    def dfs(state, seq, n_models):
        if len(seq) >= MAX_DEPTH or time.time() - start > TIME_BUDGET_S:
            return
        for name, tok in tokens.items():
            if not ALLOW_REPEATS and name in seq:
                continue
            if not structurally_allowed(name, seq, tokens):
                continue
            cls = getattr(tok, "token_class", "")
            if cls == "model" and n_models >= MAX_MODELS:
                continue
            try:
                if not tok.can_apply(state):
                    continue
            except Exception:
                continue
            if cls == "model":
                record(seq + [name])             # completed pipeline (no fit)
                if not APPLY_MODELS:
                    continue
            try:
                child = tok.apply(state.copy())
            except Exception:
                continue
            dfs(child, seq + [name], n_models + (cls == "model"))

    dfs(base, [], 0)
    return out


if __name__ == "__main__":
    H, F = seasonal_data()
    print("Searching for valid token sequences...")

    seqs = sorted(all_sequences(TOKENS, H, F), key=lambda s: (len(s), s))

    csv_filename = "valid_sequences.csv"
    with open(csv_filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["length", "sequence"])
        for s in seqs:
            writer.writerow([len(s), " -> ".join(s)])

    for s in seqs:
        print(" -> ".join(s))
    print(f"\n# Done! Saved {len(seqs)} sequences to '{csv_filename}'.")
