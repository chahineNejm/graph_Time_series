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
Grammar(20 tokens, 57 edges)
```

`outputs/token_graph.md` has been refreshed to match that live grammar.

## Catalog Note

`token_catalog.json` is a manual catalog used by `render_token_graph.py`. It may
lag behind the live grammar after architecture changes. Before regenerating the
HTML/CSV outputs, update `token_catalog.json` so it matches the current
binder-free registry.

## Update Workflow

When a token is added or its valid parents/next tokens change:

1. Inspect the live registry in `token_blocks/__init__.py`.
2. Update `token_catalog.json`.
3. Run:

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
