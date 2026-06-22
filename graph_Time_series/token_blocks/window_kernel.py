"""Windowed correlation-kernel analog forecaster (best-per-series) -- self-contained.

For each series, slide a length-W window over the panel, keep the SINGLE best
|correlation| window per scanned series, affine-map its continuation onto the
query, and forecast a kernel-weighted blend over those per-series bests.

    forecast = sum_j K_j * (a_j * B_j + b_j) / sum_j K_j,   K_j = exp(gamma*|rho_j|)

where j ranges over the (subset of) series, B_j is the best window's continuation
in series j, and (a_j, b_j) the affine fit of that window to the query.

Knobs:
* ``window``     -- W (default = horizon).
* ``gamma``      -- kernel sharpness (high -> single best, low -> broad average),
                    OR the string ``"auto"`` to solve gamma per query so the
                    blend's effective number of analogs matches ``target_meff``.
                    Adaptive mode is scale-free: it standardizes |rho| by its
                    spread before solving, so the same target means the same
                    thing whether the best correlations sit near 0.99 or 0.3.
* ``target_meff``-- adaptive target: effective # analogs M_eff=(sumK)^2/sumK^2
                    (only used when ``gamma="auto"``).
* ``top_k_per_series`` -- analogs kept per series (default 1).
* ``max_series`` -- CUTOFF: scan only this many series (seeded random subset,
                    always including the query's own series). Bounds the cost.
* ``space``      -- "value" (additive rescale) or "log" (multiplicative / relative).

Self-contained: only numpy + the model base class. Register with
``register_window_kernel(grammar)``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from ..token import ModelToken

if TYPE_CHECKING:
    from ..state import State

_EPS = 1e-9


def _resolve_history(state: "State", source_feature: str | None = None) -> np.ndarray:
    hf = state.historical_features
    names = ([source_feature] if source_feature else []) + ["clean_history", "scaled_history"]
    for name in names:
        if name and name in hf:
            h = np.asarray(hf[name], dtype=np.float64)
            if h.ndim == 2 and h.shape[0] == state.n_samples:
                return h
    if "raw_history" in state.features:
        h = np.asarray(state.features["raw_history"], dtype=np.float64)
        if h.ndim == 2 and h.shape[0] == state.n_samples:
            return h
    return np.asarray(state.original_history, dtype=np.float64)


def _affine_map(cand: np.ndarray, cont: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Map continuation by the affine fit of the window to the query (a*x+b)."""
    cm = cand.mean(); qm = q.mean()
    v = float(((cand - cm) ** 2).sum())
    a = float(((cand - cm) * (q - qm)).sum() / v) if v > 1e-12 else 1.0
    return a * cont + (qm - a * cm)


class WindowKernelToken(ModelToken):
    """Cross-series windowed correlation-kernel analog (best window per series)."""

    learning_scope = "cross_series"
    name = "window_kernel"
    reads = ()
    writes = ("prediction_stack",)
    description = (
        "Cross-series analog kernel: best |corr| window per series, affine-mapped "
        "and blended with exp(gamma*|corr|). gamma fixed or 'auto' (target M_eff). "
        "Cutoff = max_series scanned."
    )

    def __init__(self, window: int | None = None, gamma: float | str = 10.0,
                 target_meff: float = 8.0, top_k_per_series: int = 1,
                 max_series: int = 64, space: str = "value", seed: int = 0,
                 source_feature: str | None = None):
        self.window = window
        # gamma may be a fixed float OR "auto": solve gamma per query so the
        # blend's effective number of analogs M_eff = (sumK)^2/sumK^2 ~= target_meff.
        self._adaptive = isinstance(gamma, str) and gamma.lower() == "auto"
        self.gamma = gamma if self._adaptive else float(gamma)
        self.target_meff = float(target_meff)
        self.top_k_per_series = int(top_k_per_series)
        self.max_series = int(max_series)
        self.space = str(space)
        self.seed = int(seed)
        self.source_feature = source_feature

    def get_model(self) -> Any:
        return {
            "model": "window_kernel", "match": "abs_pearson",
            "window": self.window or "horizon",
            "gamma": "auto" if self._adaptive else self.gamma,
            "target_meff": self.target_meff if self._adaptive else None,
            "top_k_per_series": self.top_k_per_series, "max_series": self.max_series,
            "space": self.space, "scope": "cross_series",
        }

    def _resolve_gamma(self, r: np.ndarray) -> float:
        """Fixed gamma, or (adaptive) the gamma giving M_eff ~= target_meff.

        ``r`` are the candidate |correlations|. We standardize by their spread so
        the bisection range is scale-free, solve for the gamma that hits the
        target on the standardized scale, then convert back to raw |rho| units
        (the caller applies ``exp(gamma * (r - r.max()))``). M_eff is monotone
        decreasing in gamma -- M_eff(0) = len(r), -> 1 as gamma -> inf -- so bisect.
        """
        if not self._adaptive:
            return float(self.gamma)
        n = r.size
        target = min(self.target_meff, float(n))
        if n <= 1 or target >= n:
            return 0.0                                   # flat average
        spread = float(r.std())
        if spread < 1e-9:
            return 0.0                                   # all equal -> flat blend
        rs = (r - r.max()) / spread                      # standardized, <= 0

        def meff(g: float) -> float:
            K = np.exp(g * rs)
            s = K.sum()
            return float((s * s) / max((K * K).sum(), 1e-12))

        lo, hi = 0.0, 200.0
        if meff(hi) > target:                            # |rho| spread too tight
            return hi / spread
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if meff(mid) > target:
                lo = mid
            else:
                hi = mid
        return (0.5 * (lo + hi)) / spread                # back to raw |rho| units

    def check_specific_conditions(self, state: "State") -> bool:
        if not super().check_specific_conditions(state):
            return False
        try:
            hist = _resolve_history(state, self.source_feature)
        except (KeyError, ValueError):
            return False
        W = self.window or int(state.horizon)
        return hist.shape[1] >= W + int(state.horizon) and hist.shape[0] >= 1

    def _normalized_windows(self, y: np.ndarray, W: int, H: int):
        nwin = y.shape[0] - W - H + 1
        WV = sliding_window_view(y, W)[:nwin]          # (nwin, W) windows
        CV = sliding_window_view(y, H)[W:W + nwin]     # (nwin, H) continuations
        Ac = WV - WV.mean(1, keepdims=True)
        An = Ac / np.maximum(np.linalg.norm(Ac, axis=1, keepdims=True), 1e-12)
        return WV, CV, An

    def apply(self, state: "State") -> "State":
        hist = _resolve_history(state, self.source_feature)
        n_s, T = hist.shape
        H = int(state.horizon)
        W = self.window or H
        work = np.log(np.maximum(hist, _EPS)) if self.space == "log" else hist
        nwin = T - W - H + 1

        # Per-series windows don't depend on the query, so cache them -- but only
        # when the whole panel is scanned AND the cache fits a modest budget.
        # Otherwise (large panels with a max_series cutoff) compute on the fly so
        # memory stays bounded to one series at a time.
        est_bytes = n_s * max(nwin, 1) * W * 8 * 3
        use_cache = (n_s <= self.max_series) and (est_bytes < 1.5e9)
        cache: dict[int, tuple] = {}

        def get_windows(j: int):
            hit = cache.get(j)
            if hit is not None:
                return hit
            out = self._normalized_windows(work[j], W, H)
            if use_cache:
                cache[j] = out
            return out

        rng = np.random.default_rng(self.seed)
        pred = np.zeros((n_s, H), dtype=np.float64)

        for i in range(n_s):
            q = work[i, -W:]
            qc = q - q.mean()
            qn = qc / max(np.linalg.norm(qc), 1e-12)
            if n_s <= self.max_series:
                cand = np.arange(n_s)
            else:
                cand = rng.choice(n_s, size=self.max_series, replace=False)
                if i not in cand:
                    cand[0] = i                          # always allow own series
            rhos, conts = [], []
            for j in cand:
                WV, CV, An = get_windows(int(j))
                if An.shape[0] == 0:
                    continue
                rho = An @ qn                            # (nwin,)
                k = min(self.top_k_per_series, rho.shape[0])
                top = np.argpartition(-np.abs(rho), k - 1)[:k]
                for s in top:
                    rhos.append(float(rho[s]))
                    conts.append(_affine_map(WV[s], CV[s], q))
            if not rhos:
                pred[i] = work[i, -1]
                continue
            r = np.abs(np.asarray(rhos)); B = np.stack(conts)
            g = self._resolve_gamma(r)
            K = np.exp(g * (r - r.max()))
            pred[i] = (K[:, None] * B).sum(0) / max(K.sum(), 1e-9)

        if self.space == "log":
            pred = np.exp(pred)
        pred = pred.astype(np.float32)
        state.push_prediction(pred, self.name)
        self._log_execution(
            state,
            reads={"history": hist.shape, "window": W,
                   "series_scanned": int(min(self.max_series, n_s))},
            writes={"prediction_stack[-1]": pred.shape},
        )
        return state


def register_window_kernel(grammar):
    """Register the window-kernel model (follows any scaler/cleaner/feature)."""
    grammar.register(
        WindowKernelToken(),
        follows=["ZNormalization", "MeanAbsScaling", "FourierFeatures",
                 "LinearFill", "ForwardFill"],
        leads_to=["STOP"],
    )
    return grammar


__all__ = ["WindowKernelToken", "register_window_kernel"]
