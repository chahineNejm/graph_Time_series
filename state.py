"""RL State for forecasting pipelines."""

import numpy as np
from copy import deepcopy


class State:
    """
    The RL state for a forecasting pipeline.

    Immutable (set once):
        original_history : (n, d)  raw histories
        original_future  : (n, h)  ground truth

    Mutable (grows as tokens act):
        features           : dict[str, np.ndarray]   named feature arrays
        prediction_stack   : list[np.ndarray]         ordered model outputs
        prediction_names   : list[str]                parallel token names
        current_target     : (n, h)                   residual for next model
        token_sequence     : list[str]                full token chain
        log                : list[dict]               transformation log
        metadata           : dict                     pass-through storage
        terminated         : bool                     True after STOP
        mase               : float | None             set by STOP token
    """

    def __init__(self, original_history: np.ndarray,
                 original_future: np.ndarray):
        self.original_history = original_history
        self.original_future  = original_future

        self.features         = {"raw_history": original_history.copy()}
        self.prediction_stack = []
        self.prediction_names = []
        self.current_target   = original_future.copy()
        self.token_sequence   = []
        self.log              = []
        self.metadata         = {}
        self.terminated       = False
        self.mase             = None

    # ── properties ──────────────────────────────────────────────

    @property
    def n_samples(self) -> int:
        return self.original_history.shape[0]

    @property
    def horizon(self) -> int:
        return self.original_future.shape[1]

    @property
    def last_token(self) -> str:
        return self.token_sequence[-1] if self.token_sequence else "START"

    @property
    def depth(self) -> int:
        return len(self.token_sequence)

    @property
    def n_models_applied(self) -> int:
        return len(self.prediction_stack)

    # ── prediction helpers ──────────────────────────────────────

    def cumulative_prediction(self) -> np.ndarray:
        """Sum of all predictions in the stack."""
        if not self.prediction_stack:
            return np.zeros_like(self.original_future)
        return sum(self.prediction_stack)

    def last_prediction(self) -> np.ndarray:
        """Most recent model output (for decoders / inspectors)."""
        if not self.prediction_stack:
            return np.zeros_like(self.original_future)
        return self.prediction_stack[-1]

    def push_prediction(self, pred: np.ndarray, token_name: str):
        """Add model output to stack and auto-update residual."""
        self.prediction_stack.append(pred)
        self.prediction_names.append(token_name)
        self.current_target = self.original_future - self.cumulative_prediction()

    # ── logging ─────────────────────────────────────────────────

    def log_step(self, token_name: str, reads: dict, writes: dict):
        """Record what a token did. reads/writes: {key: shape_tuple}."""
        self.log.append({
            "step":   len(self.log),
            "token":  token_name,
            "reads":  reads,
            "writes": writes,
        })

    def print_log(self):
        """Pretty-print the transformation log."""
        print(f"{'Step':>4}  {'Token':<20}  {'Reads':<35}  {'Writes':<35}")
        print("-" * 96)
        for e in self.log:
            r = ", ".join(f"{k}{v}" for k, v in e["reads"].items())
            w = ", ".join(f"{k}{v}" for k, v in e["writes"].items())
            print(f"{e['step']:4d}  {e['token']:<20}  {r:<35}  {w:<35}")

    # ── copy ────────────────────────────────────────────────────

    def copy(self) -> "State":
        """Deep copy for MCTS rollouts. Original arrays are shared (immutable)."""
        s = State.__new__(State)
        s.original_history = self.original_history
        s.original_future  = self.original_future
        s.features         = {k: v.copy() for k, v in self.features.items()}
        s.prediction_stack = [p.copy() for p in self.prediction_stack]
        s.prediction_names = list(self.prediction_names)
        s.current_target   = self.current_target.copy()
        s.token_sequence   = list(self.token_sequence)
        s.log              = deepcopy(self.log)
        s.metadata         = deepcopy(self.metadata)
        s.terminated       = self.terminated
        s.mase             = self.mase
        return s

    def __repr__(self):
        return (f"State(tokens={self.token_sequence}, "
                f"feats={list(self.features.keys())}, "
                f"preds={len(self.prediction_stack)}, "
                f"term={self.terminated})")
