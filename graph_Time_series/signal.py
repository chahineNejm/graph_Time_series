"""Signal layer: the normalized unit of information that flows between tokens.

This module is the heart of the combinatorial architecture. Tokens stop
coupling on *names* (the old ``scaled_history`` / ``model_input`` magic strings)
and start coupling on *meaning*: a ``Signal`` carries

    (value, sem, axes, alignment, space)

and tokens declare ``Port`` requirements/provisions over those fields. A small
controlled ``sem`` vocabulary means many different tokens speak about the same
meanings (every scaler emits ``series``; every feature token emits ``features``),
and an ``adapter`` registry lets a model accept a broad panel of inputs by
auto-coercing whatever is on the board into the shape it wants.

The layer is intentionally dependency-free (numpy only) and additive: the
existing ``State`` keeps all of its old stores and methods, and simply *also*
maintains a board of ``Signal`` objects derived from those stores.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable

import numpy as np

Array = np.ndarray


# ---------------------------------------------------------------------------
# Controlled vocabularies
# ---------------------------------------------------------------------------

# WHAT a value means. Closed on purpose: adding a sem is a deliberate act, like
# adding a type. Tokens reuse these instead of inventing names.
SEMANTICS: dict[str, str] = {
    "series": "target history in some scale, axes (sample, time[, ...])",
    "features": "model-ready covariates, axes (sample, feature) or (sample, T/H, feature)",
    "level": "per-period aggregate, axes (sample, period_idx)",
    "shape": "within-period profile, axes (sample, phase)",
    "forecast": "point prediction, axes (sample, horizon)",
    "samples": "stochastic paths, axes (sample, path, horizon)",
    "mask": "validity / cleaning mask aligned to a series",
    "param": "scalar or structured parameter (use 'param:<name>')",
}

# WHAT the dimensions are. Used for shape validation and adapter matching.
AXES = ("sample", "time", "horizon", "feature", "phase", "period_idx", "path")

ALIGNMENTS = ("history", "future", "static")


def sem_root(sem: str) -> str:
    """Return the family of a sem, e.g. 'param:period' -> 'param'."""
    return sem.split(":", 1)[0]


def is_known_sem(sem: str) -> bool:
    return sem_root(sem) in SEMANTICS


# ---------------------------------------------------------------------------
# Space: scale lineage (replaces ad-hoc is_*_scaled flags)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Space:
    """The scale a value lives in, as an ordered lineage of transform names.

    ``Space(())`` is the raw data scale. ``Space(("clean", "znorm"))`` is the
    scale after cleaning then z-normalization. Two signals are scale-compatible
    iff their lineages are equal.
    """

    lineage: tuple[str, ...] = ()

    @property
    def id(self) -> str:
        return "/".join(self.lineage) if self.lineage else "raw"

    def then(self, name: str) -> "Space":
        return Space(self.lineage + (name,))

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"Space({self.id})"


RAW_SPACE = Space(())


@dataclass
class Transform:
    """A first-class, invertible scale change.

    ``forward`` may be ``None`` for legacy transforms that only registered an
    inverse; space-lifting is only attempted when ``forward`` is available.
    """

    name: str
    inverse: Callable[[Array], Array] | None = None
    forward: Callable[[Array], Array] | None = None
    params: dict[str, Any] = field(default_factory=dict)
    affects: str = "target"


# ---------------------------------------------------------------------------
# Signal: the unit of information
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Signal:
    """One named-by-meaning value on the board.

    Equality/identity for the pipeline is the tuple ``(sem, axes, alignment,
    space)``. ``name`` is provenance only and is never matched on.
    """

    value: Any
    sem: str
    axes: tuple[str, ...] = ()
    alignment: str = "history"
    space: Space = RAW_SPACE
    name: str = ""
    source: str = ""
    tags: frozenset[str] = frozenset()
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not is_known_sem(self.sem):
            raise ValueError(
                f"Unknown sem {self.sem!r}. Known families: {sorted(SEMANTICS)}."
            )
        if self.alignment not in ALIGNMENTS:
            raise ValueError(
                f"alignment must be one of {ALIGNMENTS}, got {self.alignment!r}."
            )
        for ax in self.axes:
            if ax not in AXES:
                raise ValueError(f"Unknown axis {ax!r}. Known axes: {AXES}.")
        # Auto-label for readable logs/inspection when none was supplied.
        if not self.name:
            object.__setattr__(
                self,
                "name",
                f"{self.sem}/{self.alignment}@{self.space.id}"
                + (f"#{self.source}" if self.source else ""),
            )

    @property
    def signature(self) -> tuple[str, tuple[str, ...], str, str]:
        return (self.sem, self.axes, self.alignment, self.space.id)

    @property
    def is_array(self) -> bool:
        return isinstance(self.value, np.ndarray)

    def with_value(self, value: Any, **changes: Any) -> "Signal":
        return replace(self, value=value, **changes)


# ---------------------------------------------------------------------------
# Port: what a token requires / provides
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Port:
    """A typed requirement or provision over signals.

    ``axes=None`` matches any axes. ``space`` is "current" (the running scale),
    "raw", "any", or a concrete :class:`Space`. ``multiple`` collects every
    match (used by models that fuse all available features). ``optional`` ports
    never block ``can_apply``.
    """

    sem: str
    axes: tuple[str, ...] | None = None
    alignment: str = "history"
    space: str | Space = "current"
    tags: frozenset[str] = frozenset()
    multiple: bool = False
    optional: bool = False
    coerce: bool = False  # allow adapters to satisfy this port

    def resolve_space(self, current: Space) -> Space | None:
        if self.space == "any":
            return None
        if self.space == "current":
            return current
        if self.space == "raw":
            return RAW_SPACE
        if isinstance(self.space, Space):
            return self.space
        raise ValueError(f"Bad port space {self.space!r}.")

    def matches(self, sig: Signal, current: Space) -> bool:
        if sem_root(sig.sem) != sem_root(self.sem):
            return False
        if sig.sem != self.sem and ":" in self.sem:
            return False  # param:period must match param:period exactly
        if self.alignment != "any" and sig.alignment != self.alignment:
            return False
        if self.axes is not None and sig.axes != self.axes:
            return False
        want_space = self.resolve_space(current)
        if want_space is not None and sig.space.id != want_space.id:
            return False
        if self.tags and not self.tags.issubset(sig.tags):
            return False
        return True


# ---------------------------------------------------------------------------
# Adapter registry: coercions that make models versatile
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Adapter:
    name: str
    src_sem: str
    dst_sem: str
    dst_axes: tuple[str, ...]
    fn: Callable[[Signal, int], Array]  # (signal, n_samples) -> coerced array
    src_axes: tuple[str, ...] | None = None

    def can_apply(self, sig: Signal) -> bool:
        if sem_root(sig.sem) != sem_root(self.src_sem):
            return False
        if self.src_axes is not None and sig.axes != self.src_axes:
            return False
        return True


_ADAPTERS: list[Adapter] = []


def register_adapter(adapter: Adapter) -> Adapter:
    _ADAPTERS.append(adapter)
    return adapter


def _flatten_samples(sig: "Signal", n_samples: int) -> Array:
    raw = sig.value if isinstance(sig, Signal) else sig
    arr = np.asarray(raw, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.shape[0] != n_samples:
        raise ValueError(
            f"adapter expected first dim {n_samples}, got {arr.shape[0]}."
        )
    return arr.reshape(n_samples, -1)


# Any sample-leading array can become a flat (sample, feature) matrix. This one
# rule is what lets a tabular model sit after series, period matrices, levels,
# shapes, or pre-built feature blocks without bespoke code.
for _src in ("series", "features", "level", "shape"):
    register_adapter(
        Adapter(
            name=f"flatten_{_src}",
            src_sem=_src,
            dst_sem="features",
            dst_axes=("sample", "feature"),
            fn=_flatten_samples,
        )
    )


def coerce_to(
    sig: Signal, dst_sem: str, dst_axes: tuple[str, ...], n_samples: int
) -> tuple[Array, str] | None:
    """Return (array, adapter_name) coercing ``sig`` to the target, or None.

    Identity match (already the right sem+axes) returns the value untouched.
    """
    if sem_root(sig.sem) == sem_root(dst_sem) and sig.axes == dst_axes:
        return np.asarray(sig.value, dtype=np.float32), "identity"
    for ad in _ADAPTERS:
        if ad.dst_sem == dst_sem and ad.dst_axes == dst_axes and ad.can_apply(sig):
            try:
                return ad.fn(sig, n_samples), ad.name
            except Exception:
                continue
    return None


# ---------------------------------------------------------------------------
# Auto-bundle: fuse every coercible feature signal into one design matrix
# ---------------------------------------------------------------------------


@dataclass
class Bundle:
    """A fused (n_samples, n_features) matrix plus per-block provenance."""

    matrix: Array
    blocks: list[dict[str, Any]]
    space: Space

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.matrix.shape)


def build_feature_bundle(
    signals: list[Signal],
    n_samples: int,
    *,
    current_space: Space,
    require_space: bool = True,
    select_tags: frozenset[str] = frozenset(),
    select_names: tuple[str, ...] = (),
) -> Bundle:
    """Fuse history-aligned signals coercible to features(sample, feature).

    Selection rules (in priority order):
      * if ``select_names`` is given, keep only those signal names;
      * elif ``select_tags`` is given, keep only signals carrying all tags;
      * else keep everything coercible.
    Predictions/forecasts and future-aligned signals are excluded so a model
    never trains on its own output or on horizon-side data.
    """
    dst_axes = ("sample", "feature")
    blocks: list[dict[str, Any]] = []
    arrays: list[Array] = []
    used_names: set[str] = set()

    for sig in signals:
        if sig.alignment != "history":
            continue
        if sem_root(sig.sem) in {"forecast", "samples", "param", "mask"}:
            continue
        if not sig.is_array:
            continue
        if require_space and sig.space.id != current_space.id:
            # Only fuse signals in the active modeling scale for coherence.
            if sem_root(sig.sem) == "series":
                continue
        if select_names:
            if sig.name not in select_names:
                continue
        elif select_tags:
            if not select_tags.issubset(sig.tags):
                continue
        if sig.name in used_names:
            continue
        coerced = coerce_to(sig, "features", dst_axes, n_samples)
        if coerced is None:
            continue
        block, adapter_name = coerced
        if block.shape[1] == 0:
            continue
        arrays.append(block)
        blocks.append(
            {
                "name": sig.name,
                "sem": sig.sem,
                "source": sig.source,
                "via": adapter_name,
                "cols": int(block.shape[1]),
                "space": sig.space.id,
            }
        )
        used_names.add(sig.name)

    if not arrays:
        raise ValueError("No history-aligned features could be bundled.")
    matrix = np.concatenate(arrays, axis=1).astype(np.float32)
    return Bundle(matrix=matrix, blocks=blocks, space=current_space)
