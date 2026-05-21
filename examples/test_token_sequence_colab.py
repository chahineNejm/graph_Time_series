"""Minimal manual token test file for Colab/local runs.

Edit TOKEN_SEQUENCE to test tokens one by one or in the exact order you want.
This file avoids the package-level __init__ so it can run while the framework is
still being assembled.
"""

from pathlib import Path
import importlib.util
import sys
import types

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = ROOT / "graph_Time_series"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Lightweight package shell so relative imports inside token_blocks work.
pkg = types.ModuleType("graph_Time_series")
pkg.__path__ = [str(PACKAGE_DIR)]
sys.modules.setdefault("graph_Time_series", pkg)
blocks = types.ModuleType("graph_Time_series.token_blocks")
blocks.__path__ = [str(PACKAGE_DIR / "token_blocks")]
sys.modules.setdefault("graph_Time_series.token_blocks", blocks)

state_mod = load_module("graph_Time_series.state", PACKAGE_DIR / "state.py")
token_mod = load_module("graph_Time_series.token", PACKAGE_DIR / "token.py")
norm_mod = load_module(
    "graph_Time_series.token_blocks.normalization",
    PACKAGE_DIR / "token_blocks" / "normalization.py",
)
rbf_mod = load_module(
    "graph_Time_series.token_blocks.kernel_rbf",
    PACKAGE_DIR / "token_blocks" / "kernel_rbf.py",
)

State = state_mod.State
ZNormalizationToken = norm_mod.ZNormalizationToken
KernelRBFToken = rbf_mod.KernelRBFToken


TOKENS = {
    "ZNormalization": ZNormalizationToken(),
    "kernel_rbf": KernelRBFToken(),
}

# Change this list to test any manual chain.
TOKEN_SEQUENCE = [
    "ZNormalization",
    "kernel_rbf",
]


def make_toy_data(n_samples=8, history_len=24, horizon=6):
    rng = np.random.default_rng(42)
    t_hist = np.linspace(0, 2 * np.pi, history_len)
    t_fut = np.linspace(2 * np.pi, 2.5 * np.pi, horizon)

    history = []
    future = []
    for i in range(n_samples):
        amplitude = 1.0 + 0.2 * i
        offset = 0.5 * i
        noise = 0.05 * rng.standard_normal(history_len)
        history.append(offset + amplitude * np.sin(t_hist) + noise)
        future.append(offset + amplitude * np.sin(t_fut))

    return np.asarray(history, dtype=np.float32), np.asarray(future, dtype=np.float32)


def run_sequence(sequence):
    H, F = make_toy_data()
    state = State(H, F)

    print("Initial:", state)
    for name in sequence:
        token = TOKENS[name]
        print(f"\nToken: {name}")
        print("can_apply:", token.can_apply(state))
        state = token.apply(state)
        print("state:", state)
        print("current_target:", state.current_target.shape)

    forecast = state.get_final_prediction()
    print("\nFinal forecast shape:", forecast.shape)
    print("Final forecast sample:", np.round(forecast[0], 3))
    print("\nLog:")
    state.print_log()
    return state, forecast


if __name__ == "__main__":
    run_sequence(TOKEN_SEQUENCE)
