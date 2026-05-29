"""Lightweight metadata for state artifacts and model input bundles."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ArtifactSpec:
    """Description of one named value available in State."""

    name: str
    store: str
    kind: str = "array"
    shape: tuple[int, ...] | None = None
    role: str = "feature"
    target_space: str = "active_target"
    source_token: str | None = None
    tags: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "store": self.store,
            "kind": self.kind,
            "shape": self.shape,
            "role": self.role,
            "target_space": self.target_space,
            "source_token": self.source_token,
            "tags": self.tags,
        }


@dataclass(frozen=True)
class InputBundle:
    """Approved model input assembled from one or more artifacts."""

    name: str
    kind: str
    artifact_names: tuple[str, ...]
    shape: tuple[int, ...] | None = None
    target_space: str = "active_target"
    source_token: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "artifact_names": self.artifact_names,
            "shape": self.shape,
            "target_space": self.target_space,
            "source_token": self.source_token,
            "metadata": dict(self.metadata),
        }
