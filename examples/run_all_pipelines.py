"""Run EVERY valid pipeline on EVERY dataset, scoring + saving per dataset.

Pipeline space is defined by two knobs:
  * MAX_LENGTH  -- longest token sequence to explore.
  * HELD_OUT    -- token names to exclude from the search entirely.

Datasets: each ``source_subset`` inside each ``GiftEvalPretrain__*`` shard run
(under PRETRAIN_ROOT) is its own panel (~50 series). For every dataset we run the
full sweep and write its own results files:

    <RESULTS_DIR>/<dataset>__live.csv     (appended + flushed per pipeline)
    <RESULTS_DIR>/<dataset>__ranked.csv   (best-MASE-first, rewritten as it goes)

and one combined long table  <RESULTS_DIR>/all_datasets_ranked.csv  with a
``dataset`` column. Progressive + Ctrl-C safe: stopping keeps everything scored.

One tqdm bar per dataset (which dataset + how many pipelines done + best MASE).
Nested per-token progress bars are silenced. A live "champion" line prints
whenever a new global-best pipeline is found.

Run from the repo root:   python examples/run_all_pipelines.py
"""

import sys
import csv
import time
import pickle
from pathlib import Path
from collections import Counter

import numpy as np

try:
    from tqdm.auto import tqdm
except Exception:                       # graceful fallback if tqdm is missing
    class tqdm:                         # type: ignore
        def __init__(self, *a, **k): self.n = 0; self.desc = k.get("desc", "")
        def update(self, n=1): self.n += n
        def set_postfix_str(self, s): pass
        def close(self): pass
        @staticmethod
        def write(s): print(s)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from graph_Time_series.state import State
from pipeline_inspect import build_grammar, make_token_factory, score_prediction

# =========================== CONFIG ===========================
MAX_LENGTH = 5
MAX_MODELS = 1                       # max number of model tokens per pipeline (residual stacking)
HELD_OUT = {"versatile_gb", "lightgbm_tabular", "parrot_dataset", "FlairSamplePaths", 'window_kernel','gb_level_forecast'}
MASE_M = 1                           # naive baseline period for MASE (1 = random walk; safe across datasets)

RESULTS_DIR = Path("pipeline_results")
TIME_BUDGET_PER_DATASET_S = 60*20
MAX_SERIES_PER_DATASET = None        # subsample a dataset's series; None = all (~50)
COLLAPSE_DUPLICATES = True            # drop pipelines whose final prediction is identical
                                     # to one already seen (an upstream token was ignored)

# ---- where the datasets live ----
DATA_SOURCE = "pretrain"             # "pretrain" | "synthetic"
PRETRAIN_ROOT = (Path(__file__).resolve().parents[2] / "nested loops"
                 / "pretrain_extracted_shards")
RUN_GLOB = "GiftEvalPretrain__*"     # e.g. Energy-Transport__..., Nature-Sales__...
# ==============================================================

FIELDS = ["length", "mase", "rmse", "mae", "seconds", "status", "sequence"]


def quiet_factory():
    """Token factory with all nested per-token progress bars switched off."""
    factory = make_token_factory()
    from graph_Time_series.token_blocks import (
        LevelShapeRidgeToken, FlairRidgeLevelToken, FlairSamplePathsToken,
        ParrotDatasetToken,
    )
    silent = {
        "level_shape_ridge": lambda: LevelShapeRidgeToken(show_progress=False),
        "FlairRidgeLevel": lambda: FlairRidgeLevelToken(show_progress=False),
        "FlairSamplePaths": lambda: FlairSamplePathsToken(show_progress=False),
        "parrot_dataset": lambda: ParrotDatasetToken(show_progress=False),
    }
    for name, ctor in silent.items():
        if name in factory:
            factory[name] = ctor
    return factory


def clean_1d(v):
    v = np.asarray(v, np.float32).reshape(-1).copy()
    if v.size == 0:
        return v
    fin = np.isfinite(v)
    if fin.all():
        return v
    if not fin.any():
        return np.zeros_like(v)
    idx = np.arange(v.size)
    v[~fin] = np.interp(idx[~fin], idx[fin], v[fin]).astype(np.float32)
    return v


def build_panel(recs, max_series=None):
    """Stack a dataset's records into (H, F) using their dominant length."""
    lens = Counter((np.asarray(r.get("history", [])).size,
                    np.asarray(r.get("future", [])).size) for r in recs)
    (hl, fl), _ = lens.most_common(1)[0]
    if hl < 2 or fl < 1:
        return None
    H, F = [], []
    for r in recs:
        h = clean_1d(r.get("history", [])); f = clean_1d(r.get("future", []))
        if h.size == hl and f.size == fl:
            H.append(h); F.append(f)
    if len(H) < 2:
        return None
    H, F = np.stack(H), np.stack(F)
    if max_series and H.shape[0] > max_series:
        idx = np.random.default_rng(0).choice(H.shape[0], max_series, replace=False)
        H, F = H[idx], F[idx]
    return H, F


def iter_datasets():
    """Yield (dataset_name, (H, F)) for every source_subset in every shard run."""
    if DATA_SOURCE == "synthetic":
        for d, per in (("sine_a", 24), ("sine_b", 12)):
            rng = np.random.default_rng(hash(d) % 2**32)
            t = np.arange(96); tf = np.arange(96, 96 + 24)
            H = np.stack([8 + 0.3 * k + 2 * np.sin(2 * np.pi * t / per) + rng.normal(0, .3, 96) for k in range(8)]).astype(np.float32)
            F = np.stack([8 + 0.3 * k + 2 * np.sin(2 * np.pi * tf / per) for k in range(8)]).astype(np.float32)
            yield d, (H, F)
        return
    for run in sorted(PRETRAIN_ROOT.glob(RUN_GLOB)):
        ec = run / "extracted_chunks"
        if not ec.is_dir():
            continue
        domain = run.name.split("__")[1] if "__" in run.name else run.name
        recs = []
        for fp in sorted(ec.iterdir()):
            try:
                obj = pickle.load(open(fp, "rb"))
            except Exception:
                continue
            recs += obj if isinstance(obj, list) else [obj]
        groups = {}
        for r in recs:
            groups.setdefault(str(r.get("source_subset", "unknown")), []).append(r)
        for subset, rs in groups.items():
            panel = build_panel(rs, MAX_SERIES_PER_DATASET)
            if panel is not None:
                yield f"{domain}__{subset}", panel


def write_sorted(rows, path):
    ranked = sorted(rows, key=lambda r: (r["mase"] is None,
                    r["mase"] if r["mase"] is not None else float("inf")))
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS); w.writeheader(); w.writerows(ranked)


def sweep_dataset(name, idx, H, F, grammar, factory, combined_writer, combined_fh, champion):
    """Run the full pipeline sweep on one dataset, saving progressively."""
    live_path = RESULTS_DIR / f"{name}__live.csv"
    ranked_path = RESULTS_DIR / f"{name}__ranked.csv"
    live = open(live_path, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(live, fieldnames=FIELDS); writer.writeheader(); live.flush()
    rows, seen = [], set()
    seen_sig = {}                # prediction signature -> first sequence that produced it
    collapsed = [0]              # count of identical-output pipelines dropped
    start = time.time()
    best = [float("inf")]
    pbar = tqdm(desc=f"[{idx}] {name} (n={H.shape[0]}, H={F.shape[1]})",
                unit="pipe", leave=True)

    def record(row):
        rows.append(row)
        writer.writerow(row); live.flush()
        combined_writer.writerow({"dataset": name, **row}); combined_fh.flush()
        mase = row["mase"]
        if mase is not None and mase < best[0]:
            best[0] = mase
        if mase is not None and mase < champion["mase"]:           # new global champion
            champion.update(mase=mase, sequence=row["sequence"], dataset=name)
            tqdm.write(f"  ★ champion  MASE {mase:.4f}  [{name}]  {row['sequence']}")
        pbar.set_postfix_str(f"best={best[0]:.3f} champ={champion['mase']:.3f}"
                             if champion["mase"] < float("inf") else "")
        pbar.update(1)
        if len(rows) % 10 == 0:
            write_sorted(rows, ranked_path)

    def dfs(state, seq, n_models):
        if time.time() - start > TIME_BUDGET_PER_DATASET_S or len(seq) >= MAX_LENGTH:
            return
        try:
            actions = grammar.valid_actions(state)
        except Exception:
            return
        for tname in actions:
            if tname in ("START", "STOP") or tname in HELD_OUT or tname not in factory:
                continue
            tok = factory[tname]()
            is_model = getattr(tok, "token_class", "") == "model"
            if is_model and n_models >= MAX_MODELS:           # cap residual model chains
                continue
            try:
                if not tok.can_apply(state):
                    continue
                child = tok.apply(state.copy())
            except Exception:
                continue
            new_seq = seq + [tname]
            if is_model and tuple(new_seq) not in seen:
                seen.add(tuple(new_seq))
                t0 = time.time()
                try:
                    sig = None
                    if COLLAPSE_DUPLICATES:                       # identity = the prediction, not the sequence
                        pred = np.ascontiguousarray(child.get_final_prediction(), dtype=np.float64)
                        sig = hash(np.round(pred, 6).tobytes())
                    if sig is not None and sig in seen_sig:       # an upstream token was ignored -> duplicate
                        collapsed[0] += 1
                    else:
                        if sig is not None:
                            seen_sig[sig] = " -> ".join(new_seq)
                        m = score_prediction(child, seasonal_period=MASE_M)
                        record({"length": len(new_seq), "mase": m["mase"], "rmse": m["rmse"],
                                "mae": m["mae"], "seconds": round(time.time() - t0, 3),
                                "status": "ok", "sequence": " -> ".join(new_seq)})
                except Exception as exc:
                    record({"length": len(new_seq), "mase": None, "rmse": None, "mae": None,
                            "seconds": None, "status": type(exc).__name__,
                            "sequence": " -> ".join(new_seq)})
            dfs(child, new_seq, n_models + (1 if is_model else 0))

    try:
        dfs(State(H, F), [], 0)
    finally:
        live.close()
        write_sorted(rows, ranked_path)
        live_path.unlink(missing_ok=True)        # keep only the ranked file
        pbar.close()
    best_str = "n/a" if best[0] == float("inf") else f"{best[0]:.3f}"
    tqdm.write(f"  [{idx}] {name}: {len(rows)} pipelines "
               f"({collapsed[0]} dup-output collapsed), best MASE={best_str}  -> {ranked_path}")
    return rows


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    grammar = build_grammar()
    factory = quiet_factory()
    champion = {"mase": float("inf"), "sequence": None, "dataset": None}

    combined_path = RESULTS_DIR / "all_datasets_ranked.csv"
    combined_fh = open(combined_path, "w", newline="", encoding="utf-8")
    combined_writer = csv.DictWriter(combined_fh, fieldnames=["dataset"] + FIELDS)
    combined_writer.writeheader(); combined_fh.flush()

    n_datasets = 0
    src = PRETRAIN_ROOT if DATA_SOURCE == "pretrain" else "synthetic"
    tqdm.write(f"Sweeping datasets from {src}  (MAX_LENGTH={MAX_LENGTH}, held out={sorted(HELD_OUT)})")
    try:
        for name, (H, F) in iter_datasets():
            n_datasets += 1
            sweep_dataset(name, n_datasets, H, F, grammar, factory,
                          combined_writer, combined_fh, champion)
    except KeyboardInterrupt:
        tqdm.write("\n[interrupted] partial results kept.")
    finally:
        combined_fh.close()
    champ = ("none" if champion["sequence"] is None else
             f"MASE {champion['mase']:.4f}  [{champion['dataset']}]  {champion['sequence']}")
    tqdm.write(f"\nDone: {n_datasets} datasets. Champion: {champ}")
    tqdm.write(f"Per-dataset files + combined in '{RESULTS_DIR}/'.")


if __name__ == "__main__":
    main()
