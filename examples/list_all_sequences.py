"""Print every valid token sequence (package tokens) and export readable views.

Run from the repo root:  python examples/list_all_sequences.py

The enumeration respects three structural rules on top of each token's own
``can_apply`` so the output is meaningful rather than noisy:

  1. Phase order is non-decreasing: cleaning -> scaling -> feature -> model.
     (A feature can never precede its scaler, a model is always last, etc.)
  2. At most one token per mutually-exclusive group (e.g. only ONE normalizer:
     never ZNormalization AND MeanAbsScaling in the same pipeline).
  3. Within a phase, tokens follow a fixed canonical order, so we emit ONE
     representative per combination instead of every permutation. Real
     dependencies (the FLAIR chain PeriodDetect -> SeasonalFold
     -> ...) are encoded in that canonical order and therefore preserved.

Fast by design: model tokens are recorded as terminal steps without being
*fitted*. Set APPLY_MODELS=True to also explore residual model chains
(MAX_MODELS>1); that path actually fits models, so it is slower.

Outputs:
  - valid_sequences.csv: machine-readable plain sequences.
  - valid_sequences_boxed.html: visual view with every token in a box.
"""

import sys
import time
import csv
import html
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graph_Time_series.state import State
from graph_Time_series.token_blocks import (
    ZNormalizationToken, MeanAbsScalingToken, FourierFeaturesToken,
    KernelRBFToken, RandomForestTabularToken, LightGBMTabularToken,
    ForwardFillToken, LinearFillToken, VersatileGradientBoostingToken,
    GBLevelForecastToken, FlairPreprocessToken,
    ParrotToken, ParrotDatasetToken, PeriodDetectToken, PeriodDetectBICToken,
    PeriodDetectSpectralToken, PeriodPhaseOneHotToken, SeasonalFoldToken,
    LevelShapeRidgeToken, FlairRidgeLevelToken, FlairSamplePathsToken,
)

# --- tokens to enumerate (edit freely) ---
TOKENS = {
    "ZNormalization": ZNormalizationToken(),
    "MeanAbsScaling": MeanAbsScalingToken(),
    "LinearFill": LinearFillToken(),
    "ForwardFill": ForwardFillToken(),
    "FourierFeatures": FourierFeaturesToken(n_harmonics=6),
    "kernel_rbf": KernelRBFToken(),
    "parrot": ParrotToken(),
    "parrot_dataset": ParrotDatasetToken(max_candidates=1000),
    "rf_tabular": RandomForestTabularToken(n_estimators=8),
    "lightgbm_tabular": LightGBMTabularToken(n_estimators=8),
    "versatile_gb": VersatileGradientBoostingToken(max_iter=15),
    "FlairPreprocess": FlairPreprocessToken(),
    "PeriodDetect": PeriodDetectToken(freq="H"),
    "PeriodDetectSpectral": PeriodDetectSpectralToken(freq="H"),
    "PeriodDetectBIC": PeriodDetectBICToken(freq="H"),
    "PeriodPhaseOneHot": PeriodPhaseOneHotToken(),
    "SeasonalFold": SeasonalFoldToken(),
    "level_shape_ridge": LevelShapeRidgeToken(show_progress=False),
    "FlairRidgeLevel": FlairRidgeLevelToken(show_progress=False),
    "FlairSamplePaths": FlairSamplePathsToken(),
    "gb_level_forecast": GBLevelForecastToken(max_iter=15),
}

# Canonical within-phase order (encodes the FLAIR dependency chain). Any token
# not listed falls back to the end of its phase.
CANON_ORDER = [
    # cleaning
    "LinearFill", "ForwardFill", "FlairPreprocess",
    # scaling
    "ZNormalization", "MeanAbsScaling",
    # feature (FLAIR chain first, in dependency order, then generic features)
    "PeriodDetect", "PeriodDetectSpectral", "PeriodDetectBIC",
    "PeriodPhaseOneHot", "SeasonalFold",
    "FourierFeatures",
    # model
    "gb_level_forecast", "level_shape_ridge", "FlairRidgeLevel",
    "parrot", "parrot_dataset", "kernel_rbf", "rf_tabular", "lightgbm_tabular", "versatile_gb",
]

# Mutually-exclusive groups: at most one token from each per pipeline.
EXCLUSIVE_GROUPS = [
    {"ZNormalization", "MeanAbsScaling"},   # only one normalizer
    {"LinearFill", "ForwardFill", "FlairPreprocess"},  # only one missing-value filler
    {"PeriodDetect", "PeriodDetectSpectral", "PeriodDetectBIC"},  # one detector
]

PHASE_RANK = {"cleaning": 0, "transform": 1, "feature": 2, "binding": 2,
              "model": 3, "control": 4}

# settings
MAX_DEPTH = 3        # longest sequence to explore
MAX_MODELS = 1        # cap on residual model stacking
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


def boxed_sequence_text(seq):
    """Compact terminal view: each token is visually boxed."""
    return " -> ".join(f"[ {name} ]" for name in seq)


def write_sequences_csv(seqs, path):
    with open(path, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["length", "sequence"])
        for seq in seqs:
            writer.writerow([len(seq), " -> ".join(seq)])


def write_boxed_html(seqs, tokens, path):
    rows = []
    for idx, seq in enumerate(seqs, start=1):
        boxes = []
        for pos, name in enumerate(seq):
            if pos:
                boxes.append('<span class="arrow">-></span>')
            token_class = html.escape(getattr(tokens[name], "token_class", "unknown"))
            boxes.append(
                f'<span class="token {token_class}">{html.escape(name)}</span>'
            )
        rows.append(
            f'<section class="sequence">'
            f'<span class="seq-index">{idx}</span>'
            f'<span class="seq-length">{len(seq)} tokens</span>'
            f'<div class="tokens">{"".join(boxes)}</div>'
            f'</section>'
        )

    rows_html = "\n  ".join(rows)
    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Valid Token Sequences</title>
  <style>
    :root {{
      color-scheme: light dark;
      --bg: #f8fafc;
      --fg: #0f172a;
      --muted: #64748b;
      --panel: #ffffff;
      --border: #cbd5e1;
      --arrow: #94a3b8;
      --cleaning: #dbeafe;
      --transform: #fef3c7;
      --feature: #dcfce7;
      --model: #ffedd5;
      --control: #e2e8f0;
    }}
    @media (prefers-color-scheme: dark) {{
      :root {{
        --bg: #0f172a;
        --fg: #e2e8f0;
        --muted: #94a3b8;
        --panel: #111827;
        --border: #334155;
        --arrow: #64748b;
        --cleaning: #1d4ed8;
        --transform: #92400e;
        --feature: #166534;
        --model: #9a3412;
        --control: #334155;
      }}
    }}
    body {{
      margin: 0;
      padding: 24px;
      background: var(--bg);
      color: var(--fg);
      font: 14px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    h1 {{
      margin: 0 0 4px;
      font-size: 22px;
    }}
    .summary {{
      margin: 0 0 20px;
      color: var(--muted);
    }}
    .sequence {{
      display: grid;
      grid-template-columns: 4rem 6rem minmax(0, 1fr);
      gap: 12px;
      align-items: start;
      padding: 10px 12px;
      margin-bottom: 8px;
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 8px;
    }}
    .seq-index,
    .seq-length {{
      color: var(--muted);
      white-space: nowrap;
    }}
    .tokens {{
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 6px;
      min-width: 0;
    }}
    .token {{
      display: inline-flex;
      align-items: center;
      max-width: 100%;
      padding: 4px 8px;
      border: 1px solid var(--border);
      border-radius: 6px;
      color: var(--fg);
      font-family: ui-monospace, SFMono-Regular, Consolas, "Liberation Mono", monospace;
      font-size: 12px;
      white-space: nowrap;
    }}
    .cleaning {{ background: var(--cleaning); }}
    .transform {{ background: var(--transform); }}
    .feature {{ background: var(--feature); }}
    .model {{ background: var(--model); }}
    .control {{ background: var(--control); }}
    .arrow {{
      color: var(--arrow);
      font-family: ui-monospace, SFMono-Regular, Consolas, "Liberation Mono", monospace;
    }}
  </style>
</head>
<body>
  <h1>Valid Token Sequences</h1>
  <p class="summary">{len(seqs)} sequences generated by examples/list_all_sequences.py.</p>
  {rows_html}
</body>
</html>
"""
    Path(path).write_text(page, encoding="utf-8")


if __name__ == "__main__":
    H, F = seasonal_data()
    print("Searching for valid token sequences...")

    seqs = sorted(all_sequences(TOKENS, H, F), key=lambda s: (len(s), s))

    csv_filename = "valid_sequences.csv"
    boxed_filename = "valid_sequences_boxed.html"
    write_sequences_csv(seqs, csv_filename)
    write_boxed_html(seqs, TOKENS, boxed_filename)

    for s in seqs:
        print(boxed_sequence_text(s))
    print(
        f"\n# Done! Saved {len(seqs)} sequences to '{csv_filename}' "
        f"and '{boxed_filename}'."
    )
