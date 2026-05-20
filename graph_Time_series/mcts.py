"""MCTS with PUCT for pipeline discovery."""

import math
import numpy as np
from tqdm.auto import tqdm

from .state import State
from .grammar import Grammar
from .heuristics import compute_mi_score, compute_action_priors


MAX_CHAIN_LEN = 6


class MCTSNode:
    """A node in the MCTS tree."""

    def __init__(self, name: str, parent=None, prior: float = 1.0):
        self.name = name
        self.parent = parent
        self.children: dict[str, "MCTSNode"] = {}
        self.visits = 0
        self.total_reward = 0.0
        self.prior = prior

    @property
    def avg_reward(self) -> float:
        return self.total_reward / self.visits if self.visits > 0 else 0.0

    def puct(self, c: float = 1.5) -> float:
        """PUCT score (AlphaZero-style)."""
        if self.visits == 0:
            return float("inf")
        return (self.avg_reward +
                c * self.prior * math.sqrt(self.parent.visits) / (1 + self.visits))


def mcts_search(grammar: Grammar,
                initial_state: State,
                n_iterations: int = 40,
                puct_c: float = 1.5,
                verbose: bool = True) -> dict:
    """
    Run MCTS over the grammar starting from initial_state.

    Returns dict with:
        best_chain  : str
        best_mase   : float
        all_chains  : {chain_str: best_mase}
        history     : list of per-iteration dicts
        root        : MCTSNode (full tree)
    """
    root = MCTSNode("root")
    history = []
    best_chains = {}
    best_mase_so_far = float("inf")

    pbar = tqdm(range(n_iterations), desc="MCTS", leave=True)

    for iteration in pbar:
        state = initial_state.copy()
        path_nodes = [root]
        current_node = root

        for step in range(MAX_CHAIN_LEN):
            actions = grammar.valid_actions(state)

            if not actions:
                if "STOP" in grammar.tokens:
                    grammar.tokens["STOP"].apply(state)
                break

            # MI-based priors
            mi = (compute_mi_score(state)
                  if "model_input" in state.features else 0.0)
            priors = compute_action_priors(mi, actions)

            # Expand
            for a in actions:
                if a not in current_node.children:
                    current_node.children[a] = MCTSNode(
                        a, parent=current_node,
                        prior=priors.get(a, 1.0 / len(actions)))

            # Select
            chosen = max([current_node.children[a] for a in actions],
                         key=lambda n: n.puct(puct_c))

            # Apply
            token = grammar.tokens[chosen.name]
            state = token.apply(state)

            current_node = chosen
            path_nodes.append(current_node)

            if state.terminated:
                break

        # Force STOP if chain ended without termination
        if not state.terminated and "STOP" in grammar.tokens:
            grammar.tokens["STOP"].apply(state)

        # Reward
        mase = state.mase if state.mase is not None else 10.0
        reward = 1.0 / (1.0 + mase)

        # Backpropagate
        for node in path_nodes:
            node.visits += 1
            node.total_reward += reward

        chain_str = " -> ".join(state.token_sequence)
        history.append({
            "iteration": iteration,
            "chain":     chain_str,
            "mase":      mase,
            "reward":    reward,
            "depth":     state.depth,
            "log":       state.log,
        })
        best_chains[chain_str] = min(best_chains.get(chain_str, 999), mase)
        best_mase_so_far = min(best_mase_so_far, mase)
        pbar.set_postfix(best_MASE=f"{best_mase_so_far:.4f}")

        if verbose:
            tqdm.write(f"  {iteration:3d}  {chain_str:<60s}  MASE={mase:.4f}")

    best = min(best_chains, key=best_chains.get)
    return {
        "best_chain":  best,
        "best_mase":   best_chains[best],
        "all_chains":  best_chains,
        "history":     history,
        "root":        root,
    }


def print_mcts_tree(node: MCTSNode, indent: int = 0, max_depth: int = 6):
    """Print indented tree with visit counts and estimated MASE."""
    prefix = "  " * indent
    name = node.name if node.name != "root" else "ROOT"
    if node.visits > 0:
        mase_est = ((1.0 / node.avg_reward - 1.0)
                    if node.avg_reward > 0 else float("inf"))
        print(f"{prefix}|-- {name}  [v={node.visits}, ~MASE={mase_est:.2f}]")
    if indent < max_depth:
        for ch in sorted(node.children.values(),
                         key=lambda c: c.visits, reverse=True):
            if ch.visits > 0:
                print_mcts_tree(ch, indent + 1, max_depth)
