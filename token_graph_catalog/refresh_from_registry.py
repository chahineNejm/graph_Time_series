"""Refresh token_catalog.json from the live package registry."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = Path(__file__).resolve().parent / "token_catalog.json"


def main() -> None:
    sys.path.insert(0, str(ROOT))

    from graph_Time_series.grammar import Grammar
    from graph_Time_series.token_blocks import (
        register_default_tokens,
        register_flair_gb_swap,
        register_flair_tokens,
        register_seasonal_tokens,
        register_versatile_tokens,
    )

    grammar = Grammar()
    for register in (
        register_default_tokens,
        register_flair_tokens,
        register_versatile_tokens,
        register_flair_gb_swap,
        register_seasonal_tokens,
    ):
        grammar = register(grammar)

    tokens = [_catalog_node(grammar, node, attrs)
              for node, attrs in grammar.graph.nodes(data=True)]
    tokens.sort(key=_sort_key)

    catalog = {
        "schema_version": 1,
        "project": "graph_Time_series",
        "notes": [
            "Generated from the live package registry in graph_Time_series.token_blocks.",
            repr(grammar),
            "Regenerate after token registrations or valid transitions change.",
        ],
        "tokens": tokens,
    }
    CATALOG_PATH.write_text(
        json.dumps(catalog, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"{repr(grammar)}")
    print(f"Wrote {len(tokens)} nodes to {CATALOG_PATH}")


def _catalog_node(grammar, node: str, attrs: dict) -> dict:
    if node == "START":
        token_class = "control"
        status = "synthetic"
        description = "Synthetic start node for graph visualization."
    elif node == "STOP":
        token_class = "terminal"
        status = "synthetic"
        description = "Terminal node allowed after a model has produced a prediction."
    else:
        token_class = attrs.get("token_class", "")
        status = "package"
        description = attrs.get("description", "")

    return {
        "id": node,
        "status": status,
        "class": token_class,
        "core": True,
        "description": description,
        "reads": list(attrs.get("reads", []) or []),
        "writes": list(attrs.get("writes", []) or []),
        "parents": sorted(grammar.graph.predecessors(node)),
        "next": sorted(grammar.graph.successors(node)),
    }


def _sort_key(token: dict) -> tuple[int, str]:
    class_order = {
        "control": 0,
        "cleaning": 1,
        "transform": 2,
        "feature": 3,
        "model": 4,
        "terminal": 5,
    }
    return class_order.get(token["class"], 99), token["id"].lower()


if __name__ == "__main__":
    main()
