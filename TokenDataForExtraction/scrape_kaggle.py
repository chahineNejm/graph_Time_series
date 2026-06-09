#!/usr/bin/env python3
"""
scrape_ts_notebooks.py
======================

Pull Kaggle notebooks (.ipynb JSON) related to time-series forecasting via the
official Kaggle API. This stage ONLY downloads notebooks. Token extraction is a
separate later step.

What it does
------------
1. Searches Kaggle kernels for a set of forecasting-related queries.
2. Optionally restricts to kernels attached to specific competitions.
3. Deduplicates by kernel ref ("owner/slug").
4. Pulls each kernel's notebook JSON + metadata into its own folder.
5. Writes an index.json summarizing everything that was fetched.

Prerequisites
-------------
    pip install kaggle
    # Put your kaggle.json at ~/.kaggle/kaggle.json  (chmod 600)
    # (Account -> Settings -> API -> Create New API Token)

Usage
-----
    python scrape_ts_notebooks.py --max-per-query 20 --out ./kaggle_ts_notebooks
    python scrape_ts_notebooks.py --competitions store-sales-time-series-forecasting
    python scrape_ts_notebooks.py --queries "fourier forecasting" "wavelet time series"

Notes
-----
- Only public, downloadable kernels are retrievable. Some kernels disable code
  download; those are skipped and recorded in index.json under "failures".
- The Kaggle kernel search is keyword-based and not perfectly precise, so a
  lightweight relevance filter is applied on the returned metadata.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable


# ---------------------------------------------------------------------------
# Default search configuration
# ---------------------------------------------------------------------------

# Queries aimed at the *techniques* you care about, not just generic "time
# series". Edit freely; each runs as a separate kernel search.
DEFAULT_QUERIES = [
    "time series forecasting",
    "fourier transform time series",
    "fft forecasting",
    "wavelet time series",
    "seasonal decomposition forecasting",
    "spectral analysis time series",
    "kalman filter forecasting",
    "state space time series",
    "arima sarima forecasting",
    "prophet forecasting",
    "lstm time series forecasting",
    "gradient boosting time series",
    "feature engineering time series forecasting",
]

# Competitions whose notebooks are a goldmine of forecasting ideas.
# Used only when --competitions is passed (or --use-default-competitions).
DEFAULT_COMPETITIONS = [
    "store-sales-time-series-forecasting",
    "m5-forecasting-accuracy",
    "web-traffic-time-series-forecasting",
    "demand-forecasting-kernels-only",
    "rohlik-orders-forecasting-challenge",
]

# Words that, if present in title/ref, boost confidence the kernel is on-topic.
RELEVANCE_HINTS = [
    "forecast", "time series", "timeseries", "ts ", "arima", "sarima",
    "prophet", "lstm", "gru", "fourier", "fft", "wavelet", "spectral",
    "seasonal", "kalman", "demand", "sales", "traffic", "horizon", "lag",
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class KernelRecord:
    ref: str                       # "owner/slug"
    title: str = ""
    author: str = ""
    last_run_time: str = ""
    total_votes: int = 0
    language: str = ""
    kernel_type: str = ""
    matched_queries: list = field(default_factory=list)
    source_competition: str = ""
    relevance_score: int = 0
    notebook_path: str = ""        # where the .ipynb was written
    metadata_path: str = ""


# ---------------------------------------------------------------------------
# Kaggle API wrapper
# ---------------------------------------------------------------------------

def get_api():
    """Import and authenticate the Kaggle API. Fails loudly with guidance."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        sys.exit(
            "The 'kaggle' package is not installed.\n"
            "  pip install kaggle\n"
            "Then create an API token at kaggle.com (Settings -> API -> "
            "Create New API Token) and place kaggle.json at ~/.kaggle/."
        )
    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as exc:  # noqa: BLE001
        sys.exit(
            f"Kaggle authentication failed: {exc}\n"
            "Ensure ~/.kaggle/kaggle.json exists (chmod 600) or that "
            "KAGGLE_USERNAME / KAGGLE_KEY env vars are set."
        )
    return api


def _attr(obj, *names, default=""):
    """Kaggle SDK objects use camelCase attrs that occasionally shift; try a few."""
    for n in names:
        if hasattr(obj, n) and getattr(obj, n) is not None:
            return getattr(obj, n)
    return default


def score_relevance(title: str, ref: str) -> int:
    blob = f"{title} {ref}".lower()
    return sum(1 for h in RELEVANCE_HINTS if h in blob)


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def search_kernels_by_query(
    api,
    query: str,
    max_results: int,
    sort_by: str = "voteCount",
) -> Iterable:
    """Page through kernel search results for one query."""
    collected = []
    page = 1
    page_size = 100  # Kaggle caps at 100
    while len(collected) < max_results:
        try:
            batch = api.kernels_list(
                search=query,
                sort_by=sort_by,
                page=page,
                page_size=min(page_size, max_results - len(collected)),
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  ! search failed for '{query}' (page {page}): {exc}")
            break
        if not batch:
            break
        collected.extend(batch)
        if len(batch) < page_size:
            break
        page += 1
        time.sleep(0.5)  # be polite
    return collected[:max_results]


def search_kernels_for_competition(api, competition: str, max_results: int) -> Iterable:
    try:
        return api.kernels_list(
            competition=competition,
            sort_by="voteCount",
            page_size=min(100, max_results),
        )[:max_results]
    except Exception as exc:  # noqa: BLE001
        print(f"  ! competition search failed for '{competition}': {exc}")
        return []


# ---------------------------------------------------------------------------
# Pull
# ---------------------------------------------------------------------------

def slugify(ref: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", ref)


def pull_kernel(api, rec: KernelRecord, out_dir: Path) -> bool:
    """Download one kernel's notebook + metadata into its own folder."""
    dest = out_dir / slugify(rec.ref)
    dest.mkdir(parents=True, exist_ok=True)
    try:
        # kernels_pull writes <slug>.ipynb (or .py/.r) plus kernel-metadata.json
        api.kernels_pull(rec.ref, path=str(dest), metadata=True)
    except Exception as exc:  # noqa: BLE001
        print(f"  ! pull failed for {rec.ref}: {exc}")
        return False

    # Locate the notebook file that was written.
    nb = next(iter(dest.glob("*.ipynb")), None)
    if nb is None:
        # Kernel may be a script (.py/.r/.rmd), still useful but not .ipynb.
        script = next(
            (p for p in dest.iterdir() if p.suffix.lower() in {".py", ".r", ".rmd"}),
            None,
        )
        if script is None:
            print(f"  ! no notebook/script file found for {rec.ref}")
            return False
        rec.notebook_path = str(script)
    else:
        rec.notebook_path = str(nb)

    meta = next(iter(dest.glob("*metadata*.json")), None)
    if meta:
        rec.metadata_path = str(meta)
    return True


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def build_records(api, args) -> dict:
    records: dict[str, KernelRecord] = {}

    def add(item, query="", competition=""):
        ref = _attr(item, "ref")
        if not ref:
            return
        if ref in records:
            rec = records[ref]
        else:
            title = _attr(item, "title")
            rec = KernelRecord(
                ref=ref,
                title=title,
                author=_attr(item, "author"),
                last_run_time=str(_attr(item, "lastRunTime", "last_run_time")),
                total_votes=int(_attr(item, "totalVotes", "total_votes", default=0) or 0),
                language=_attr(item, "language"),
                kernel_type=_attr(item, "kernelType", "kernel_type"),
                relevance_score=score_relevance(title, ref),
            )
            records[ref] = rec
        if query and query not in rec.matched_queries:
            rec.matched_queries.append(query)
        if competition:
            rec.source_competition = competition

    # 1. Keyword queries
    if not args.competitions_only:
        for q in args.queries:
            print(f"Searching kernels: '{q}'")
            for item in search_kernels_by_query(api, q, args.max_per_query, args.sort_by):
                add(item, query=q)

    # 2. Competition-attached kernels
    comps = list(args.competitions)
    if args.use_default_competitions:
        comps += [c for c in DEFAULT_COMPETITIONS if c not in comps]
    for comp in comps:
        print(f"Searching competition kernels: '{comp}'")
        for item in search_kernels_for_competition(api, comp, args.max_per_query):
            add(item, competition=comp)

    return records


def main():
    p = argparse.ArgumentParser(
        description="Scrape Kaggle time-series forecasting notebooks (.ipynb JSON)."
    )
    p.add_argument("--out", default="./kaggle_ts_notebooks",
                   help="Output directory (default: ./kaggle_ts_notebooks)")
    p.add_argument("--keyword", "-k", default=None,
                   help="Single search keyword/phrase. Overrides --queries and "
                        "the default query list (e.g. -k \"fourier forecasting\").")
    p.add_argument("--queries", nargs="*", default=DEFAULT_QUERIES,
                   help="Search queries (defaults to a curated forecasting set)")
    p.add_argument("--competitions", nargs="*", default=[],
                   help="Specific competition slugs to pull notebooks from")
    p.add_argument("--use-default-competitions", action="store_true",
                   help="Also include the built-in list of TS competitions")
    p.add_argument("--competitions-only", action="store_true",
                   help="Skip keyword search; only pull competition notebooks")
    p.add_argument("--max-per-query", type=int, default=20,
                   help="Max kernels to fetch per query/competition (default: 20)")
    p.add_argument("--min-relevance", type=int, default=0,
                   help="Drop kernels whose relevance score is below this (default: 0)")
    p.add_argument("--min-votes", type=int, default=0,
                   help="Drop kernels with fewer votes than this (default: 0)")
    p.add_argument("--sort-by", default="voteCount",
                   help="Kaggle sort key: voteCount, hotness, dateRun, etc.")
    p.add_argument("--limit-total", type=int, default=0,
                   help="Hard cap on total kernels actually downloaded (0 = no cap)")
    p.add_argument("--dry-run", action="store_true",
                   help="List what would be pulled without downloading")
    args = p.parse_args()

    # A single --keyword overrides the whole query list.
    if args.keyword:
        args.queries = [args.keyword]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    notebooks_dir = out_dir / "notebooks"
    notebooks_dir.mkdir(exist_ok=True)

    api = get_api()
    records = build_records(api, args)

    # Filter + rank
    selected = [
        r for r in records.values()
        if r.relevance_score >= args.min_relevance and r.total_votes >= args.min_votes
    ]
    selected.sort(key=lambda r: (r.relevance_score, r.total_votes), reverse=True)
    if args.limit_total > 0:
        selected = selected[: args.limit_total]

    print(f"\nFound {len(records)} unique kernels; "
          f"{len(selected)} pass filters and will be {'listed' if args.dry_run else 'pulled'}.\n")

    if args.dry_run:
        for r in selected:
            print(f"  [{r.relevance_score:>2}] {r.total_votes:>4}v  {r.ref}  — {r.title}")
        index = {"dry_run": True, "candidates": [asdict(r) for r in selected]}
        (out_dir / "index.json").write_text(json.dumps(index, indent=2))
        print(f"\nWrote candidate list to {out_dir / 'index.json'}")
        return

    pulled, failures = [], []
    for i, rec in enumerate(selected, 1):
        print(f"[{i}/{len(selected)}] pulling {rec.ref}")
        ok = pull_kernel(api, rec, notebooks_dir)
        (pulled if ok else failures).append(rec)
        time.sleep(0.6)  # rate-limit courtesy

    index = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": {
            "queries": args.queries if not args.competitions_only else [],
            "competitions": list(args.competitions)
            + (DEFAULT_COMPETITIONS if args.use_default_competitions else []),
            "max_per_query": args.max_per_query,
            "min_relevance": args.min_relevance,
            "min_votes": args.min_votes,
            "sort_by": args.sort_by,
        },
        "counts": {
            "unique_found": len(records),
            "selected": len(selected),
            "pulled": len(pulled),
            "failed": len(failures),
        },
        "notebooks": [asdict(r) for r in pulled],
        "failures": [asdict(r) for r in failures],
    }
    (out_dir / "index.json").write_text(json.dumps(index, indent=2))

    print(f"\nDone. Pulled {len(pulled)} notebooks, {len(failures)} failed.")
    print(f"Notebooks in: {notebooks_dir}")
    print(f"Index:        {out_dir / 'index.json'}")


if __name__ == "__main__":
    main()
