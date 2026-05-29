"""Optional Lark AST layer for token pipelines.

This module keeps the execution model deliberately small: parse a pipeline
expression into an inspectable AST, validate it against the existing Grammar,
then apply the same token instances that the graph grammar already owns.
"""

from __future__ import annotations

import json
from ast import literal_eval
from dataclasses import dataclass
from typing import Any, Iterable

from .grammar import Grammar
from .state import State


PIPELINE_GRAMMAR = r"""
    ?start: pipeline
    pipeline: stop
            | token_call (ARROW token_call)* (ARROW stop)?

    token_call: NAME call_args?
    call_args: "(" [kwarg ("," kwarg)* [","]] ")"
    kwarg: NAME "=" value

    ?value: SIGNED_NUMBER      -> number
          | ESCAPED_STRING     -> string
          | "true"             -> true
          | "false"            -> false
          | "None"             -> none
          | "null"             -> none

    stop: "STOP"

    ARROW: "->" | "=>"
    NAME: /[A-Za-z_][A-Za-z0-9_]*/

    %import common.ESCAPED_STRING
    %import common.SIGNED_NUMBER
    %import common.WS
    %ignore WS
"""


@dataclass(frozen=True)
class TokenCall:
    """One token invocation in a parsed pipeline."""

    name: str
    kwargs: dict[str, Any]

    def to_source(self) -> str:
        if not self.kwargs:
            return self.name
        args = ", ".join(
            f"{key}={_format_value(value)}" for key, value in self.kwargs.items()
        )
        return f"{self.name}({args})"


@dataclass(frozen=True)
class PipelineAST:
    """A linear token pipeline parsed from DSL text."""

    steps: tuple[TokenCall, ...]
    stop: bool = False

    @property
    def names(self) -> tuple[str, ...]:
        tail = ("STOP",) if self.stop else ()
        return tuple(step.name for step in self.steps) + tail

    def to_source(self) -> str:
        parts = [step.to_source() for step in self.steps]
        if self.stop:
            parts.append("STOP")
        return " -> ".join(parts)


class PipelineSyntaxError(ValueError):
    """Raised when pipeline DSL text cannot be parsed."""


class PipelineValidationError(ValueError):
    """Raised when parsed pipeline names or transitions are invalid."""


def parse_pipeline(source: str) -> PipelineAST:
    """Parse pipeline DSL text into a PipelineAST.

    Lark is imported lazily so the rest of the package can still be used when
    the optional parser dependency is not installed.
    """

    try:
        from lark import Lark, Transformer, v_args
        from lark.exceptions import LarkError
    except ImportError as exc:
        raise ImportError(
            "The pipeline AST parser requires lark. Install it with "
            "`pip install lark` to use parse_pipeline()."
        ) from exc

    @v_args(inline=True)
    class _PipelineTransformer(Transformer):
        def pipeline(self, *items):
            steps: list[TokenCall] = []
            stop = False
            for item in items:
                if item == "STOP":
                    stop = True
                elif isinstance(item, TokenCall):
                    steps.append(item)
            return PipelineAST(tuple(steps), stop=stop)

        def token_call(self, name, kwargs=None):
            return TokenCall(str(name), kwargs or {})

        def call_args(self, *items):
            args = {}
            for key, value in items:
                args[key] = value
            return args

        def kwarg(self, name, value):
            return str(name), value

        def number(self, value):
            text = str(value)
            return float(text) if any(ch in text for ch in ".eE") else int(text)

        def string(self, value):
            return literal_eval(str(value))

        def true(self):
            return True

        def false(self):
            return False

        def none(self):
            return None

        def stop(self):
            return "STOP"

        def ARROW(self, _token):
            return None

    try:
        parser = Lark(
            PIPELINE_GRAMMAR,
            parser="lalr",
            start="start",
            transformer=_PipelineTransformer(),
        )
        return parser.parse(source)
    except LarkError as exc:
        raise PipelineSyntaxError(f"Invalid pipeline syntax: {source!r}") from exc


def validate_pipeline(
    pipeline: str | PipelineAST,
    grammar: Grammar,
    state: State | None = None,
) -> PipelineAST:
    """Validate a pipeline against the graph grammar and optional state.

    When state is supplied, tokens are replayed on a copy so later dependency
    checks see the features produced by earlier tokens. Omit state for cheap
    static validation of names and graph edges only.
    """

    ast = parse_pipeline(pipeline) if isinstance(pipeline, str) else pipeline
    _validate_token_names(ast, grammar)
    _validate_edges(ast.names, grammar)

    if state is not None:
        trial = state.copy()
        _walk_state(ast, grammar, trial, apply_tokens=False)

    return ast


def apply_pipeline(
    pipeline: str | PipelineAST,
    grammar: Grammar,
    state: State,
    *,
    copy_state: bool = False,
) -> State:
    """Apply a parsed pipeline by delegating to the existing token blocks."""

    ast = validate_pipeline(pipeline, grammar)
    target = state.copy() if copy_state else state
    return _walk_state(ast, grammar, target, apply_tokens=True)


def pipeline_to_sequence(pipeline: str | PipelineAST) -> tuple[str, ...]:
    """Return token names from text or an existing AST."""

    ast = parse_pipeline(pipeline) if isinstance(pipeline, str) else pipeline
    return ast.names


def _validate_token_names(ast: PipelineAST, grammar: Grammar) -> None:
    missing = [name for name in ast.names if name != "STOP" and name not in grammar.tokens]
    if missing:
        known = ", ".join(sorted(grammar.tokens))
        raise PipelineValidationError(
            f"Unknown token(s): {', '.join(missing)}. Registered tokens: {known}"
        )


def _validate_edges(names: Iterable[str], grammar: Grammar) -> None:
    last = "START"
    for name in names:
        if not grammar.graph.has_edge(last, name):
            raise PipelineValidationError(
                f"Invalid transition {last!r} -> {name!r} for this grammar."
            )
        last = name


def _walk_state(
    ast: PipelineAST,
    grammar: Grammar,
    state: State,
    *,
    apply_tokens: bool,
) -> State:
    for step in ast.steps:
        token = grammar.tokens[step.name]
        if step.kwargs:
            raise PipelineValidationError(
                f"Token arguments are parsed but not applied yet: {step.to_source()}. "
                "Register a configured token instance in Grammar instead."
            )
        if not token.can_apply(state):
            raise PipelineValidationError(
                f"Token {step.name!r} cannot apply after {state.last_token!r}."
            )
        if apply_tokens:
            state = token.apply(state)
        else:
            # Validation needs state mutations to make later dependency checks
            # meaningful, but callers asked only for validation. Mutate the
            # copied trial state, never the original state.
            state = token.apply(state)

    if ast.stop:
        if state.n_models_applied <= 0:
            raise PipelineValidationError("STOP requires at least one model token.")
        if apply_tokens:
            state.terminated = True
            state.features["final_forecast"] = state.get_final_prediction()
            state.record_token(
                "STOP",
                "control",
                {"prediction_stack": len(state.prediction_stack)},
                {"final_forecast": state.features["final_forecast"].shape},
            )

    return state


def _format_value(value: Any) -> str:
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "None"
    return repr(value)
