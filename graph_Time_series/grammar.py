"""Grammar graph — defines valid token transitions."""

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from .state import State
from .token import Token


# Decoder/Encoder concepts are replaced by 'transform' since inverses are handled implicitly
TOKEN_COLOURS = {
    "cleaning":  "#8ecae6",
    "feature":   "#219ebc",
    "binding":   "#6d597a",
    "transform": "#ffb703",  
    "model":     "#fb8500",
    "control":   "#e63946",
    "START":     "#95d5b2",
    "STOP":      "#e63946",
}


class Grammar:
    """
    Directed graph defining the token vocabulary and valid transitions.

    Nodes = token names + "START" + "STOP"
    Edges = valid transitions ("token A can be followed by token B")

    The graph only models the FORWARD pass of the pipeline. Inverses 
    (like reversing a Z-Score) are handled automatically by the State 
    when the STOP token is reached.
    """

    def __init__(self):
        self.graph = nx.DiGraph()
        self.graph.add_node("START", token_class="START", color=TOKEN_COLOURS["START"])
        self.graph.add_node("STOP", token_class="control", color=TOKEN_COLOURS["STOP"])
        self.tokens: dict[str, Token] = {}

    def register(self, token: Token, follows: list[str], leads_to: list[str] = None):
        """
        Register a token and its grammar edges.

        Args:
            token:    Token instance
            follows:  token names (or "START") that can precede this token
            leads_to: token names (or "STOP") that can follow this token
        
        additional comments:    
            TIP FOR SCALING (The Markov Trap):
        If you have tokens that can be played multiple times in any order 
        (e.g., "Lag", "RollingMean", "DayOfWeek"), do NOT hardcode their 
        names into every single register() call. This causes combinatorial 
        explosion and spaghetti code. 
        
        Instead, use global lists in your script to group them conceptually:
            ALL_FEATURES = ["Lag", "RollingMean", "DayOfWeek"]
            ALL_MODELS = ["XGBoost", "LinearTrend"]
            
            grammar.register(LagToken, 
                             follows=["START"] + ALL_FEATURES, 
                             leads_to=ALL_FEATURES + ALL_MODELS)
        """
        self.tokens[token.name] = token
        
        self.graph.add_node(
            token.name,
            token_class=token.token_class,
            reads=getattr(token, "reads", []),
            writes=getattr(token, "writes", []),   # could be useful afterwards for automated graph checking
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
        Checks both graph edges AND logical feature availability.
        """
        last = state.last_token
        
        # If the last token isn't in the graph (or we somehow reached a dead end)
        if last not in self.graph:
            return ["STOP"]

        successors = list(self.graph.successors(last))
        valid = []
        
        for name in successors:
            if name == "STOP":
                # Only allow STOP if the pipeline has generated at least one prediction
                if state.n_models_applied > 0:
                    valid.append("STOP")
            elif name in self.tokens:
                tok = self.tokens[name]
                # Delegate all other business logic directly to the token
                if tok.can_apply(state):
                    valid.append(name)
                    
        # Always provide an exit hatch if no other actions are valid
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

    # Attempt hierarchical layout, fallback to multi-partite or spring
    try:
        from networkx.drawing.nx_agraph import graphviz_layout
        pos = graphviz_layout(G, prog="dot")
    except ImportError:
        try:
            pos = nx.multipartite_layout(G, subset_key="token_class")
        except Exception:
            pos = nx.spring_layout(G, k=2.5, iterations=80, seed=42)

    colors = [G.nodes[n].get("color", "#aaa") for n in G.nodes]
    sizes = [(3 + G.degree(n)) * 200 for n in G.nodes]

    labels = {n:n for n in G.nodes} # can be changed afterwards for better naming convention

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
