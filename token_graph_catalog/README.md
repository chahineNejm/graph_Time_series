# Token Graph Catalog

This mini-project keeps visual maps of token parents and possible next tokens
separate from the main README.

The current runtime architecture is binder-free by default. The graph should
therefore show direct transitions from transforms/features to models, with
models consuming `state.feature_bundle()` when no legacy `model_input` exists.

## Files

```text
token_graph_catalog/
|-- README.md
|-- token_catalog.json
|-- refresh_from_registry.py
|-- render_token_graph.py
`-- outputs/
    |-- token_graph.html
    |-- token_graph.md
    `-- token_matrix.csv
```

## Current Source Of Truth

The most reliable source of token transitions is the live package registry in
`graph_Time_series/token_blocks/__init__.py`.

Registry helpers:

- `register_default_tokens`
- `register_flair_tokens`
- `register_versatile_tokens`
- `register_flair_gb_swap`
- `register_seasonal_tokens`

With all five enabled, the current graph is:

```text
Grammar(24 tokens, 124 edges)
```

`token_catalog.json` and `outputs/` have been refreshed from that live grammar.

Notebook-only unfinished tokens are intentionally not part of this graph yet:

```text
window_kernel
affine_fold
affine_forecast
```

## Catalog Note

`token_catalog.json` is now refreshed from the live registry and then rendered
by `render_token_graph.py`. If a future token has notebook-only or planned
status, add that entry deliberately after regenerating the package graph.

## Update Workflow

When a token is added or its valid parents/next tokens change:

1. Inspect the live registry in `token_blocks/__init__.py`.
2. Refresh `token_catalog.json` from the live registry:

   ```bash
   python refresh_from_registry.py
   ```

3. Render the readable outputs:

   ```bash
   python render_token_graph.py
   ```

4. Check `outputs/token_graph.md`, `outputs/token_graph.html`, and
   `outputs/token_matrix.csv`.

## Core vs Full Graph

The catalog has a `core` flag per token:

- `core: true` means the token appears in the compact graph.
- `core: false` means the token appears only in the full graph.

Use `core: false` for noisy experimental adapters, aliases, deprecated entries,
or future proposals. Do not mark removed binder/glue tokens as core.

## Visual Classes

Current classes:

- `control`
- `cleaning`
- `transform`
- `feature`
- `model`
- `planned`

The renderer still knows about a historical `binding` class, but active package
docs should treat binder tokens as removed unless a future selector-token design
is explicitly approved.
