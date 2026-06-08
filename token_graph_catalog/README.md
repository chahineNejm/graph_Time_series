# Token Graph Catalog

This mini-project keeps a visual map of token dependencies separate from the
main code README.

The goal is simple:

- edit one token catalog when tokens change;
- generate readable graph views;
- keep binding/adapter tokens visible in the full graph but hide them from the
  simpler core graph;
- export a CSV matrix for quick inspection.

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

## Update Workflow

When a token is added or its valid parents/next tokens change:

1. Edit `token_catalog.json`.
2. Run:

   ```bash
   python render_token_graph.py
   ```

3. Open `outputs/token_graph.html` or preview `outputs/token_graph.md`.

## Core vs Full Graph

The catalog has a `core` flag per token.

- `core: true` means the token appears in the compact graph.
- `core: false` means the token appears only in the full graph.

Use `core: false` for generic factories, broad binders, aliases, or planned
tokens that would make the day-to-day graph noisy.

## Catalog Fields

Each token entry uses this shape:

```json
{
  "id": "PeriodSelection",
  "status": "notebook",
  "class": "feature",
  "core": true,
  "description": "Select a candidate period and store it in metadata.",
  "reads": ["features.raw_history"],
  "writes": ["metadata.period", "metadata.period_scores"],
  "parents": ["START", "ContextWindow"],
  "next": ["PeriodPhaseOneHot", "PeriodFold"]
}
```

The renderer accepts edges from both `parents` and `next`, merges duplicates,
and reports unknown token references.

## Visual Classes

Current classes:

- `control`
- `cleaning`
- `transform`
- `feature`
- `binding`
- `model`
- `planned`

Binding tokens are intentionally separate from feature tokens because they are
mostly adapters: they choose what becomes `model_input`.
