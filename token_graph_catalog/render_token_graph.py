"""Render token graph views from token_catalog.json.

Outputs:
    outputs/token_graph.md
    outputs/token_graph.html
    outputs/token_matrix.csv
"""

from __future__ import annotations

import csv
import html
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CATALOG_PATH = ROOT / "token_catalog.json"
OUTPUT_DIR = ROOT / "outputs"

CLASS_ORDER = {
    "control": 0,
    "cleaning": 1,
    "transform": 2,
    "feature": 3,
    "binding": 4,
    "model": 5,
    "planned": 6,
}

CLASS_COLORS = {
    "control": {"fill": "#ffe2e2", "stroke": "#b91c1c"},
    "cleaning": {"fill": "#dff0ff", "stroke": "#2563eb"},
    "transform": {"fill": "#fff1c2", "stroke": "#b45309"},
    "feature": {"fill": "#d8f7ef", "stroke": "#047857"},
    "binding": {"fill": "#efe3ff", "stroke": "#7c3aed"},
    "model": {"fill": "#ffe6cf", "stroke": "#ea580c"},
    "planned": {"fill": "#f4f4f5", "stroke": "#71717a"},
}


def main() -> None:
    catalog = load_catalog()
    tokens = catalog["tokens"]
    token_map = {token["id"]: token for token in tokens}
    edges = collect_edges(tokens)
    validate_references(token_map, edges)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(tokens, edges)
    write_markdown(catalog, token_map, edges)
    write_html(catalog, token_map, edges)
    print(f"Wrote outputs to {OUTPUT_DIR}")


def load_catalog() -> dict:
    return json.loads(CATALOG_PATH.read_text(encoding="utf-8"))


def collect_edges(tokens: list[dict]) -> list[tuple[str, str]]:
    edges: set[tuple[str, str]] = set()
    for token in tokens:
        token_id = token["id"]
        for parent in token.get("parents", []):
            edges.add((parent, token_id))
        for child in token.get("next", []):
            edges.add((token_id, child))
    return sorted(edges)


def validate_references(token_map: dict[str, dict], edges: list[tuple[str, str]]) -> None:
    unknown = sorted(
        {node for edge in edges for node in edge if node not in token_map}
    )
    if unknown:
        raise ValueError(f"Unknown token references: {', '.join(unknown)}")


def view_tokens(token_map: dict[str, dict], *, core_only: bool) -> dict[str, dict]:
    if not core_only:
        return dict(token_map)
    return {
        token_id: token
        for token_id, token in token_map.items()
        if bool(token.get("core", True))
    }


def view_edges(
    token_map: dict[str, dict],
    edges: list[tuple[str, str]],
    *,
    core_only: bool,
) -> list[tuple[str, str]]:
    visible = view_tokens(token_map, core_only=core_only)
    return [
        (src, dst)
        for src, dst in edges
        if src in visible and dst in visible
    ]


def write_csv(tokens: list[dict], edges: list[tuple[str, str]]) -> None:
    parents = defaultdict(list)
    children = defaultdict(list)
    for src, dst in edges:
        parents[dst].append(src)
        children[src].append(dst)

    with (OUTPUT_DIR / "token_matrix.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "status",
                "class",
                "core",
                "parents",
                "next",
                "reads",
                "writes",
                "description",
            ],
        )
        writer.writeheader()
        for token in tokens:
            token_id = token["id"]
            writer.writerow(
                {
                    "id": token_id,
                    "status": token.get("status", ""),
                    "class": token.get("class", ""),
                    "core": token.get("core", True),
                    "parents": "; ".join(sorted(set(parents[token_id]))),
                    "next": "; ".join(sorted(set(children[token_id]))),
                    "reads": "; ".join(token.get("reads", [])),
                    "writes": "; ".join(token.get("writes", [])),
                    "description": token.get("description", ""),
                }
            )


def write_markdown(
    catalog: dict,
    token_map: dict[str, dict],
    edges: list[tuple[str, str]],
) -> None:
    core_mermaid = mermaid_graph(token_map, edges, core_only=True)
    full_mermaid = mermaid_graph(token_map, edges, core_only=False)
    rows = token_table_rows(token_map, edges)

    text = [
        "# Token Graph",
        "",
        f"Project: `{catalog.get('project', '')}`",
        "",
        "Generated from `token_catalog.json`.",
        "",
        "## Core Graph",
        "",
        "Core view hides generic binders, aliases, and broad experimental adapters.",
        "",
        "```mermaid",
        core_mermaid,
        "```",
        "",
        "## Full Graph",
        "",
        "Full view includes adapter/factory tokens and aliases.",
        "",
        "```mermaid",
        full_mermaid,
        "```",
        "",
        "## Token Matrix",
        "",
        "| Token | Status | Class | Core | Parents | Next |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    text.extend(rows)
    text.append("")
    (OUTPUT_DIR / "token_graph.md").write_text("\n".join(text), encoding="utf-8")


def mermaid_graph(
    token_map: dict[str, dict],
    edges: list[tuple[str, str]],
    *,
    core_only: bool,
) -> str:
    visible = view_tokens(token_map, core_only=core_only)
    visible_edges = view_edges(token_map, edges, core_only=core_only)

    lines = ["flowchart LR"]
    for token_id in sorted(visible, key=lambda item: sort_key(visible[item])):
        token = visible[token_id]
        label = f"{token_id}<br/>{token.get('status', '')} / {token.get('class', '')}"
        lines.append(f"  {node_id(token_id)}[\"{label}\"]:::{class_name(token)}")
    for src, dst in visible_edges:
        lines.append(f"  {node_id(src)} --> {node_id(dst)}")
    for cls, colors in CLASS_COLORS.items():
        lines.append(
            "  classDef "
            f"{cls} fill:{colors['fill']},stroke:{colors['stroke']},stroke-width:1px,color:#111827"
        )
    return "\n".join(lines)


def write_html(
    catalog: dict,
    token_map: dict[str, dict],
    edges: list[tuple[str, str]],
) -> None:
    core_svg = svg_graph(token_map, edges, core_only=True, title="Core Token Graph")
    full_svg = svg_graph(token_map, edges, core_only=False, title="Full Token Graph")
    table = html_table(token_map, edges)
    styles = html_styles()

    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Token Graph Catalog</title>
  <style>{styles}</style>
</head>
<body>
  <main>
    <header>
      <h1>Token Graph Catalog</h1>
      <p>Project: <code>{html.escape(catalog.get("project", ""))}</code></p>
      <p>Edit <code>token_catalog.json</code>, then rerun <code>python render_token_graph.py</code>.</p>
    </header>
    <section>
      <h2>Core Graph</h2>
      <p>Compact view. Generic binders, broad adapters, and aliases are hidden.</p>
      {core_svg}
    </section>
    <section>
      <h2>Full Graph</h2>
      <p>Complete view with all cataloged tokens.</p>
      {full_svg}
    </section>
    <section>
      <h2>Token Matrix</h2>
      {table}
    </section>
  </main>
</body>
</html>
"""
    (OUTPUT_DIR / "token_graph.html").write_text(page, encoding="utf-8")


def svg_graph(
    token_map: dict[str, dict],
    edges: list[tuple[str, str]],
    *,
    core_only: bool,
    title: str,
) -> str:
    visible = view_tokens(token_map, core_only=core_only)
    visible_edges = view_edges(token_map, edges, core_only=core_only)
    positions = layout_positions(visible)
    width = max((x for x, _ in positions.values()), default=0) + 260
    height = max((y for _, y in positions.values()), default=0) + 120

    parts = [
        f'<svg class="graph" viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">',
        "<defs>",
        '<marker id="arrow" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth">',
        '<path d="M0,0 L0,6 L9,3 z" fill="#64748b" />',
        "</marker>",
        "</defs>",
    ]
    for src, dst in visible_edges:
        if src not in positions or dst not in positions:
            continue
        x1, y1 = positions[src]
        x2, y2 = positions[dst]
        start_x = x1 + 190
        start_y = y1 + 30
        end_x = x2
        end_y = y2 + 30
        c1 = start_x + 45
        c2 = end_x - 45
        parts.append(
            f'<path class="edge" d="M {start_x} {start_y} C {c1} {start_y}, {c2} {end_y}, {end_x} {end_y}" />'
        )
    for token_id, token in sorted(visible.items(), key=lambda item: sort_key(item[1])):
        x, y = positions[token_id]
        colors = CLASS_COLORS.get(token.get("class"), CLASS_COLORS["planned"])
        dash = ' stroke-dasharray="6 4"' if token.get("status") == "planned" else ""
        parts.append(
            f'<g class="node" transform="translate({x},{y})">'
            f'<title>{html.escape(token.get("description", ""))}</title>'
            f'<rect width="190" height="60" rx="8" fill="{colors["fill"]}" stroke="{colors["stroke"]}"{dash}/>'
            f'<text x="12" y="24" class="node-title">{html.escape(token_id)}</text>'
            f'<text x="12" y="45" class="node-meta">{html.escape(token.get("status", ""))} / {html.escape(token.get("class", ""))}</text>'
            "</g>"
        )
    parts.append("</svg>")
    return "\n".join(parts)


def layout_positions(token_map: dict[str, dict]) -> dict[str, tuple[int, int]]:
    columns: dict[int, list[str]] = defaultdict(list)
    for token_id, token in token_map.items():
        columns[CLASS_ORDER.get(token.get("class", "planned"), 6)].append(token_id)

    positions: dict[str, tuple[int, int]] = {}
    x_gap = 250
    y_gap = 90
    for col in sorted(columns):
        ids = sorted(columns[col], key=lambda token_id: token_map[token_id]["id"].lower())
        for row, token_id in enumerate(ids):
            positions[token_id] = (30 + col * x_gap, 35 + row * y_gap)
    return positions


def html_table(token_map: dict[str, dict], edges: list[tuple[str, str]]) -> str:
    parents = defaultdict(list)
    children = defaultdict(list)
    for src, dst in edges:
        parents[dst].append(src)
        children[src].append(dst)

    rows = []
    for token_id, token in sorted(token_map.items(), key=lambda item: sort_key(item[1])):
        rows.append(
            "<tr>"
            f"<td><code>{html.escape(token_id)}</code></td>"
            f"<td>{html.escape(token.get('status', ''))}</td>"
            f"<td>{html.escape(token.get('class', ''))}</td>"
            f"<td>{'yes' if token.get('core', True) else 'no'}</td>"
            f"<td>{html.escape(', '.join(sorted(set(parents[token_id]))))}</td>"
            f"<td>{html.escape(', '.join(sorted(set(children[token_id]))))}</td>"
            f"<td>{html.escape(token.get('description', ''))}</td>"
            "</tr>"
        )
    return (
        "<table>"
        "<thead><tr><th>Token</th><th>Status</th><th>Class</th><th>Core</th>"
        "<th>Parents</th><th>Next</th><th>Description</th></tr></thead>"
        "<tbody>"
        + "\n".join(rows)
        + "</tbody></table>"
    )


def token_table_rows(token_map: dict[str, dict], edges: list[tuple[str, str]]) -> list[str]:
    parents = defaultdict(list)
    children = defaultdict(list)
    for src, dst in edges:
        parents[dst].append(src)
        children[src].append(dst)

    rows = []
    for token_id, token in sorted(token_map.items(), key=lambda item: sort_key(item[1])):
        rows.append(
            "| "
            + " | ".join(
                [
                    f"`{token_id}`",
                    token.get("status", ""),
                    token.get("class", ""),
                    "yes" if token.get("core", True) else "no",
                    ", ".join(f"`{p}`" for p in sorted(set(parents[token_id]))),
                    ", ".join(f"`{c}`" for c in sorted(set(children[token_id]))),
                ]
            )
            + " |"
        )
    return rows


def html_styles() -> str:
    return """
body {
  margin: 0;
  color: #111827;
  background: #f8fafc;
  font-family: Arial, sans-serif;
}
main {
  max-width: 1280px;
  margin: 0 auto;
  padding: 28px;
}
section, header {
  margin-bottom: 28px;
}
h1, h2 {
  margin: 0 0 10px;
}
p {
  margin: 6px 0;
  color: #475569;
}
code {
  background: #e5e7eb;
  border-radius: 4px;
  padding: 1px 4px;
}
.graph {
  width: 100%;
  min-height: 360px;
  border: 1px solid #cbd5e1;
  border-radius: 8px;
  background: white;
  overflow: visible;
}
.edge {
  fill: none;
  stroke: #64748b;
  stroke-width: 1.5;
  marker-end: url(#arrow);
  opacity: 0.8;
}
.node-title {
  font-size: 13px;
  font-weight: 700;
}
.node-meta {
  font-size: 11px;
  fill: #475569;
}
table {
  width: 100%;
  border-collapse: collapse;
  background: white;
  border: 1px solid #cbd5e1;
}
th, td {
  border-bottom: 1px solid #e2e8f0;
  padding: 8px;
  text-align: left;
  vertical-align: top;
  font-size: 13px;
}
th {
  background: #f1f5f9;
}
"""


def sort_key(token: dict) -> tuple[int, str]:
    return (
        CLASS_ORDER.get(token.get("class", "planned"), 6),
        token.get("id", "").lower(),
    )


def class_name(token: dict) -> str:
    cls = token.get("class", "planned")
    return cls if cls in CLASS_COLORS else "planned"


def node_id(token_id: str) -> str:
    safe = []
    for char in token_id:
        safe.append(char if char.isalnum() else "_")
    return "n_" + "".join(safe)


if __name__ == "__main__":
    main()
