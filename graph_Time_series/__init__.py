"""
CLP - Forecasting Pipeline Discovery via MCTS.

Grammar-guided search over cleaning -> feature -> model token chains.
"""

from .state import State
from .grammar import Grammar, plot_grammar
from .pipeline_ast import (
    PipelineAST,
    PipelineSyntaxError,
    PipelineValidationError,
    TokenCall,
    apply_pipeline,
    parse_pipeline,
    pipeline_to_sequence,
    validate_pipeline,
)

try:
    from .mcts import mcts_search, print_mcts_tree
except ModuleNotFoundError as exc:
    if exc.name != f"{__name__}.mcts":
        raise

    def mcts_search(*_args, **_kwargs):
        raise ImportError("mcts.py is not present in this checkout.")

    def print_mcts_tree(*_args, **_kwargs):
        raise ImportError("mcts.py is not present in this checkout.")

__all__ = [
    "State",
    "Grammar",
    "plot_grammar",
    "mcts_search",
    "print_mcts_tree",
    "PipelineAST",
    "PipelineSyntaxError",
    "PipelineValidationError",
    "TokenCall",
    "apply_pipeline",
    "parse_pipeline",
    "pipeline_to_sequence",
    "validate_pipeline",
]
