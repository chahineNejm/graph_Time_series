"""Parrot (analog / nearest-neighbour) forecaster.

The simplest analog model: for each series, find the past window most strongly
correlated (in absolute value) with the most recent window, and copy forward
whatever followed it -- affine-rescaled to the current level.

``ParrotToken`` searches each series' own past (``within_series``);
``ParrotDatasetToken`` searches the whole panel (``cross_series``) with a
budget-first sampler and early stop so the much larger search stays tractable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ..token import ModelToken

if TYPE_CHECKING:
    from ..state import State


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    d = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / d) if d > 1e-12 else 0.0


class ParrotToken(ModelToken):
    """Copy the continuation of the best |correlation| past analog window.

    Per series (``within_series``): the query is the last ``H`` points (``H`` =
    horizon). Slide over the series' own past, score each ``H``-window by
    ``|Pearson|`` against the query, take the single best, and forecast the
    ``H`` points that followed it. Because correlation ignores level/scale, the
    copied continuation is affine-rescaled via the fit ``query ~= a*window + b``
    on the matched window (``a`` may be negative -- an anti-correlated analog is
    flipped). Predicts on the active modeling scale, like every other model.
    """

    learning_scope = "within_series"
    name = "parrot"
    reads = ()
    writes = ("prediction_stack",)
    max_uses = 1
    description = (
        "Analog forecast: copy the continuation of the past window with the "
        "highest |correlation| to the recent window, affine-rescaled."
    )

    def __init__(self, source_feature: str | None = None):
        self.source_feature = source_feature

    def get_model(self) -> Any:
        return {
            "model": "parrot",
            "match": "abs_pearson",
            "window": "horizon",
            "neighbours": 1,
            "source_feature": self.source_feature or "auto",
        }

    def check_specific_conditions(self, state: "State") -> bool:
        if not super().check_specific_conditions(state):
            return False
        try:
            hist, _ = self._resolve_history(state)
        except (KeyError, ValueError):
            return False
        return hist.shape[1] > 0 and np.all(np.isfinite(hist))

    def _resolve_history(self, state: "State") -> tuple[np.ndarray, str]:
        hf = state.historical_features
        candidates: list[tuple[str, np.ndarray]] = []
        if self.source_feature:
            for store_name, store in (
                ("historical_features", state.historical_features),
                ("features", state.features),
            ):
                if self.source_feature in store:
                    candidates.append(
                        (f"{store_name}.{self.source_feature}", store[self.source_feature])
                    )
                    break

        for name in ("scaled_history", "clean_history"):
            if name in hf:
                candidates.append((f"historical_features.{name}", hf[name]))
        if "raw_history" in state.features:
            candidates.append(("features.raw_history", state.features["raw_history"]))

        for source_name, values in candidates:
            hist = np.asarray(values, dtype=np.float64)
            if hist.ndim != 2:
                continue
            if hist.shape[0] != state.n_samples:
                continue
            return hist, source_name
        raise KeyError(
            "parrot requires scaled_history, clean_history, or active raw_history."
        )

    def _forecast_one(self, y: np.ndarray, H: int) -> np.ndarray:
        T = y.shape[0]
        if T < 2 * H or H < 1:
            # too short to hold an analog: persist the most recent values
            if T == 0:
                return np.zeros(H, dtype=np.float64)
            reps = int(np.ceil(H / T))
            return np.tile(y, reps)[-H:]
        query = y[-H:]
        best_abs, best_i = -1.0, 0
        for i in range(0, T - 2 * H + 1):
            c = _pearson(query, y[i:i + H])
            if abs(c) > best_abs:
                best_abs, best_i = abs(c), i
        cand = y[best_i:best_i + H]
        cont = y[best_i + H:best_i + 2 * H]
        var = float(((cand - cand.mean()) ** 2).sum())
        a = (float(((cand - cand.mean()) * (query - query.mean())).sum() / var)
             if var > 1e-12 else 1.0)
        b = float(query.mean() - a * cand.mean())
        return a * cont + b

    def apply(self, state: "State") -> "State":
        hist, input_name = self._resolve_history(state)
        H = int(state.horizon)
        n = hist.shape[0]
        pred = np.zeros((n, H), dtype=np.float32)
        for i in range(n):
            pred[i] = self._forecast_one(hist[i], H).astype(np.float32)

        state.push_prediction(pred, self.name)
        self._log_execution(
            state,
            reads={
                input_name: hist.shape,
                "current_target": state.current_target.shape,
                "n_previous_models": state.n_models_applied - 1,
            },
            writes={
                "prediction_stack[-1]": pred.shape,
                "current_residual": state.current_target.shape,
            },
        )
        return state


class ParrotDatasetToken(ParrotToken):
    """Cross-series parrot: search EVERY series' past for the best analog.

    Same as :class:`ParrotToken` but ``within_series`` -> ``cross_series``: each
    series' query window is matched against the windows of the whole panel, so a
    series with no good self-analog can borrow one from a sibling. The affine
    rescale transplants the match onto the query's level/scale, so a window from
    a different series at a different level is still usable.

    The full cross-series scan is ``O(N^2 * T * H)``, so the search **never**
    materializes the full candidate list by default. It samples up to
    ``max_candidates`` candidate windows per query, optionally limits the number
    of source series via ``max_series_candidates``, and stops early once a match
    clears ``min_corr``. The running best is kept, so even if nothing clears the
    bar the best-seen-within-budget is used. The query's own most-recent window
    is never a candidate (it has no in-history continuation).
    """

    learning_scope = "cross_series"
    name = "parrot_dataset"
    description = (
        "Cross-series analog forecast: search the whole panel for the past window "
        "most |correlation| with the query, stopping early once one is good enough."
    )

    def __init__(self, source_feature: str | None = None,
                 min_corr: float | None = 0.97, max_candidates: int | None = 10000,
                 max_series_candidates: int | None = None, seed: int = 0,
                 show_progress: bool = False, progress_min_samples: int = 16):
        super().__init__(source_feature=source_feature)
        self.min_corr = None if min_corr is None else float(min_corr)
        self.max_candidates = None if max_candidates is None else int(max_candidates)
        self.max_series_candidates = (
            None if max_series_candidates is None else int(max_series_candidates)
        )
        self.seed = int(seed)
        self.show_progress = bool(show_progress)
        self.progress_min_samples = int(progress_min_samples)

    def get_model(self) -> Any:
        return {
            "model": "parrot_dataset",
            "match": "abs_pearson",
            "window": "horizon",
            "neighbours": 1,
            "scope": "cross_series",
            "min_corr": self.min_corr,
            "max_candidates": self.max_candidates,
            "max_series_candidates": self.max_series_candidates,
            "source_feature": self.source_feature or "auto",
        }

    def _forecast_cross(self, query: np.ndarray, hist: np.ndarray, H: int,
                        n_win: int, candidate_ids) -> np.ndarray:
        thr = self.min_corr
        best_abs, bj, bk, scanned = -1.0, 0, 0, 0
        for idx in candidate_ids:
            j = int(idx) // n_win
            k = int(idx) % n_win
            c = _pearson(query, hist[j, k:k + H])
            scanned += 1
            if abs(c) > best_abs:
                best_abs, bj, bk = abs(c), j, k
            if thr is not None and best_abs >= thr:
                break
        cand = hist[bj, bk:bk + H]
        cont = hist[bj, bk + H:bk + 2 * H]
        var = float(((cand - cand.mean()) ** 2).sum())
        a = (float(((cand - cand.mean()) * (query - query.mean())).sum() / var)
             if var > 1e-12 else 1.0)
        b = float(query.mean() - a * cand.mean())
        return a * cont + b

    def _source_series_ids(self, n: int, target_i: int,
                           rng: np.random.Generator) -> np.ndarray:
        cap = self.max_series_candidates
        if cap is None or cap >= n:
            return np.arange(n, dtype=np.int64)

        cap = max(1, int(cap))
        if cap == 1:
            return np.asarray([target_i], dtype=np.int64)

        others = np.concatenate(
            (np.arange(0, target_i, dtype=np.int64),
             np.arange(target_i + 1, n, dtype=np.int64))
        )
        take = min(cap - 1, others.size)
        sampled = (
            rng.choice(others, size=take, replace=False)
            if take
            else np.empty(0, dtype=np.int64)
        )
        return np.concatenate((np.asarray([target_i], dtype=np.int64), sampled))

    def _sample_local_ids(self, total: int, budget: int,
                          rng: np.random.Generator) -> np.ndarray:
        """Sample unique local candidate ids without building ``permutation(total)``."""
        budget = min(max(int(budget), 1), int(total))
        if budget >= total:
            return np.arange(total, dtype=np.int64)

        seen: set[int] = set()
        out: list[int] = []
        while len(out) < budget:
            need = budget - len(out)
            batch_size = min(max(need * 2, 64), 100_000)
            for value in rng.integers(0, total, size=batch_size, dtype=np.int64):
                idx = int(value)
                if idx in seen:
                    continue
                seen.add(idx)
                out.append(idx)
                if len(out) >= budget:
                    break
        return np.asarray(out, dtype=np.int64)

    def _candidate_ids(self, n: int, n_win: int, target_i: int,
                       rng: np.random.Generator):
        series_ids = self._source_series_ids(n, target_i, rng)
        total = int(series_ids.size * n_win)
        budget = self.max_candidates
        if budget is None:
            local_ids = range(total)
        elif budget >= total:
            local_ids = rng.permutation(total)
        else:
            local_ids = self._sample_local_ids(total, int(budget), rng)

        for local in local_ids:
            local = int(local)
            yield int(series_ids[local // n_win]) * n_win + (local % n_win)

    def apply(self, state: "State") -> "State":
        hist, input_name = self._resolve_history(state)
        H = int(state.horizon)
        n, T = hist.shape
        n_win = T - 2 * H + 1
        pred = np.zeros((n, H), dtype=np.float32)

        if n_win <= 0 or H < 1:
            # too short to hold an analog anywhere: per-series persistence fallback
            for i in range(n):
                pred[i] = self._forecast_one(hist[i], H).astype(np.float32)
        else:
            rng = np.random.default_rng(self.seed)
            iterator = range(n)
            if self.show_progress and n >= self.progress_min_samples:
                iterator = _maybe_tqdm(iterator, "parrot_dataset")
            for i in iterator:
                pred[i] = self._forecast_cross(
                    hist[i, -H:],
                    hist,
                    H,
                    n_win,
                    self._candidate_ids(n, n_win, i, rng),
                ).astype(np.float32)

        state.push_prediction(pred, self.name)
        self._log_execution(
            state,
            reads={
                input_name: hist.shape,
                "candidate_windows": int(max(n * n_win, 0)),
                "max_candidates": self.max_candidates,
                "max_series_candidates": self.max_series_candidates,
                "current_target": state.current_target.shape,
            },
            writes={"prediction_stack[-1]": pred.shape},
        )
        return state


def _maybe_tqdm(iterable, desc: str):
    try:
        from tqdm.auto import tqdm

        return tqdm(iterable, desc=desc)
    except Exception:
        return iterable
