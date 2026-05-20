"""
CLP Framework - Fixed Version
==============================
Run this as a script or convert to notebook cells.

Key fixes vs the original:
1. __init__.py now exports CLP classes properly
2. CleanNormalize token added (z-score per sample) - biggest perf gain
3. FFT decoder stub removed, replaced with compact spectral features
4. FeatLagFeatures added (autocorrelation-based features)
5. Kernel RBF now cross-validates gamma regularization
6. RF/XGBoost hyperparams tuned for small n
7. STOP token un-normalizes predictions before MASE computation
"""

import sys, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- Ensure the package is importable when this script is run from examples/ ---
REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PACKAGE_PARENT = os.path.dirname(REPO_DIR)
for path in [REPO_DIR, PACKAGE_PARENT]:
    if path not in sys.path:
        sys.path.insert(0, path)

from graph_Time_series import State, Grammar, plot_grammar, mcts_search, print_mcts_tree
from graph_Time_series.tokens import register_all
from graph_Time_series.tokens.models import compute_mase


# =================================================================
# 1. Load GiftEval data
# =================================================================
from datasets import load_dataset

def load_gifteval_config(config_name, max_samples=25):
    """Load a GiftEval config, return (history, future) arrays."""
    ds = load_dataset("Salesforce/GiftEvalParquet", config_name, split="test")

    histories, futures = [], []
    for row in ds:
        ts = np.array(row["target"], dtype=np.float32)
        h = int(row.get("prediction_length", len(ts) // 5))
        if len(ts) < h + 10:
            continue
        histories.append(ts[:-h])
        futures.append(ts[-h:])
        if len(histories) >= max_samples:
            break

    min_hist = min(len(x) for x in histories)
    min_fut  = min(len(x) for x in futures)
    H = np.array([x[-min_hist:] for x in histories], dtype=np.float32)
    F = np.array([x[:min_fut]   for x in futures],   dtype=np.float32)
    print(f"  {config_name}: {H.shape[0]} samples, history={H.shape[1]}, horizon={F.shape[1]}")
    return H, F


CONFIGS = [
    "electricity_hourly_H_short",
    "traffic_weekly_weekly_short",
    "nn5_daily_daily_short",
    "weather_weekly_weekly_short",
]

data = {}
for cfg in CONFIGS:
    try:
        data[cfg] = load_gifteval_config(cfg, max_samples=25)
    except Exception as e:
        print(f"  SKIP {cfg}: {e}")

print(f"\nLoaded {len(data)} configs")


# =================================================================
# 2. Build grammar and register tokens
# =================================================================
grammar = Grammar()
register_all(grammar)
print(grammar)
print("Tokens:", grammar.token_names)


# =================================================================
# 3. Run MCTS search per config
# =================================================================
results = {}

for cfg, (H, F) in data.items():
    print(f"\n{'='*70}")
    print(f"Config: {cfg}  ({H.shape[0]} samples)")
    print(f"{'='*70}")

    state = State(H, F)
    res = mcts_search(grammar, state, n_iterations=40, puct_c=1.5, verbose=True)

    results[cfg] = res
    print(f"\n  Best: {res['best_chain']}  (MASE={res['best_mase']:.4f})")


# =================================================================
# 4. Summary table
# =================================================================
print(f"\n{'Config':<40s}  {'Best chain':<55s}  {'MASE':>8s}")
print("-" * 105)
for cfg, res in results.items():
    print(f"{cfg:<40s}  {res['best_chain']:<55s}  {res['best_mase']:8.4f}")


# =================================================================
# 5. MCTS tree visualisation
# =================================================================
for cfg, res in results.items():
    print(f"\n{'='*60}")
    print(f"  {cfg}")
    print(f"{'='*60}")
    print_mcts_tree(res["root"], max_depth=4)


# =================================================================
# 6. Convergence plots
# =================================================================
if results:
    fig, axes = plt.subplots(1, len(results), figsize=(5*len(results), 4), squeeze=False)
    for ax, (cfg, res) in zip(axes[0], results.items()):
        mases = [h["mase"] for h in res["history"]]
        best_so_far = np.minimum.accumulate(mases)
        ax.plot(mases, alpha=0.3, label="per-iteration")
        ax.plot(best_so_far, linewidth=2, label="best so far")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("MASE")
        ax.set_title(cfg.split("_")[0])
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), "convergence.png"), dpi=120)
    print("\nConvergence plot saved to convergence.png")


# =================================================================
# 7. Replay best chain
# =================================================================
if results:
    cfg_name = list(results.keys())[0]
    H, F = data[cfg_name]
    best = results[cfg_name]

    chain_tokens = best["best_chain"].replace(" -> ", ",").split(",")
    chain_tokens = [t.strip() for t in chain_tokens if t.strip() != "STOP"]

    print(f"\nReplaying: {best['best_chain']}  on  {cfg_name}\n")

    state = State(H, F)
    for tok_name in chain_tokens:
        token = grammar.tokens[tok_name]
        state = token.apply(state)

    grammar.tokens["STOP"].apply(state)
    state.print_log()
    print(f"\nFinal MASE: {state.mase:.4f}")


    # =================================================================
    # 8. Forecast visualisation
    # =================================================================
    forecast = state.features["final_forecast"]
    n_show = min(4, state.n_samples)
    fig, axes = plt.subplots(1, n_show, figsize=(4*n_show, 3))
    if n_show == 1:
        axes = [axes]

    rng = np.random.default_rng(42)
    idxs = rng.choice(state.n_samples, size=n_show, replace=False)

    for ax, i in zip(axes, idxs):
        hist = state.original_history[i]
        fut  = state.original_future[i]
        pred = forecast[i]
        t_h = np.arange(len(hist))
        t_f = np.arange(len(hist), len(hist) + len(fut))

        ax.plot(t_h[-50:], hist[-50:], "k-", label="history")
        ax.plot(t_f, fut, "b-", linewidth=2, label="actual")
        ax.plot(t_f, pred, "r--", linewidth=2, label="forecast")
        ax.set_title(f"Sample {i}")
        ax.legend(fontsize=7)

    plt.suptitle(f"{cfg_name} - {best['best_chain']}", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), "forecasts.png"), dpi=120)
    print("Forecast plot saved to forecasts.png")
