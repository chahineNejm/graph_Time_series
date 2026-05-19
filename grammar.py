"""Grammar graph — defines valid token transitions."""

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from .state import State
from .token import Token


TOKEN_COLOURS = {
    "cleaning":  "#8ecae6",
    "feature":   "#219ebc",
    "encoder":   "#ffb703",
    "model":     "#fb8500",
    "decoder":   "#023047",
    "control":   "#e63946",
    "START":     "#95d5b2",
}


class Grammar:
    """
    Directed graph defining the token vocabulary and valid transitions.

    Nodes = token names + "START"
    Edges = valid transitions ("token A can be followed by token B")

    Starts with just START. Grows with each register() call.
    """

    def __init__(self):
        self.graph = nx.DiGraph()
        self.graph.add_node("START", token_class="START",
                            color=TOKEN_COLOURS["START"])
        self.tokens: dict[str, Token] = {}

    def register(self, token: Token,
                 follows: list[str],
                 leads_to: list[str] = None):
        """
        Register a token and its grammar edges.

        Args:
            token:    Token instance
            follows:  token names (or "START") that can precede this token
            leads_to: token names (or "STOP") that can follow this token
        """
        self.tokens[token.name] = token
        self.graph.add_node(
            token.name,
            token_class=token.token_class,
            reads=token.reads,
            writes=token.writes,
            description=getattr(token, "description", ""),
            color=TOKEN_COLOURS.get(token.token_class, "#aaa"),
        )
        for prev in follows:
            self.graph.add_edge(prev, token.name)
        if leads_to:
            for nxt in leads_to:
                self.graph.add_edge(token.name, nxt)

    def valid_actions(self, state: State) -> list[str]:
        """
        Valid next tokens given current state.
        Checks both graph edges AND feature availability.
        """
        last = state.last_token
        if last not in self.graph:
            return ["STOP"]

        successors = list(self.graph.successors(last))
        valid = []
        for name in successors:
            if name in self.tokens:
                tok = self.tokens[name]
                if tok.token_class == "control":
                    if state.n_models_applied > 0:
                        valid.append(name)
                elif tok.can_apply(state):
                    valid.append(name)
        return valid if valid else ["STOP"]

    @property
    def token_names(self) -> list[str]:
        return list(self.tokens.keys())

    @property
    def n_tokens(self) -> int:
        return len(self.tokens)

    def __repr__(self):
        return f"Grammar({self.n_tokens} tokens, {self.graph.number_of_edges()} edges)"


def plot_grammar(g: Grammar, figsize=(14, 8), title="Grammar graph"):
    """Draw the grammar graph. Node colour = token class, size ~ degree."""
    G = g.graph
    if G.number_of_nodes() < 2:
        print("Grammar is empty.")
        return

    try:
        from networkx.drawing.nx_agraph import graphviz_layout
        pos = graphviz_layout(G, prog="dot")
    except Exception:
        pos = nx.spring_layout(G, k=2.5, iterations=80, seed=42)

    colors = [G.nodes[n].get("color", "#aaa") for n in G.nodes]
    sizes = [(3 + G.degree(n)) * 200 for n in G.nodes]

    labels = {}
    for n in G.nodes:
        a = G.nodes[n]
        r, w = a.get("reads", []), a.get("writes", [])
        lbl = n
        if r:
            lbl += f"\nR:{r}"
        if w:
            lbl += f"\nW:{w}"
        labels[n] = lbl

    fig, ax = plt.subplots(figsize=figsize)
    nx.draw(G, pos, ax=ax, node_color=colors, node_size=sizes,
            edgecolors="black", linewidths=0.8, arrows=True,
            arrowsize=15, edge_color="#666", width=1.2,
            connectionstyle="arc3,rad=0.1")
    nx.draw_networkx_labels(G, pos, labels, font_size=7, ax=ax)

    handles = [mpatches.Patch(color=c, label=cls)
               for cls, c in TOKEN_COLOURS.items()
               if any(G.nodes[n].get("token_class") == cls for n in G.nodes)]
    ax.legend(handles=handles, loc="upper left", fontsize=9)
    ax.set_title(title)
    plt.tight_layout()
    plt.show()
