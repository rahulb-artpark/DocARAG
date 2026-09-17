"""Resumable refinement state for the graph stage.

Entity resolution issues one or two model calls per entity and is the longest
phase of indexing a large document, but the graph it mutates was only persisted
once the whole stage finished. Any crash discarded every hour of it. The
snapshot here is written at each batch boundary and holds the graph, the entity
alias map and the refined-node set in a single file, so one atomic rename keeps
the three mutually consistent.

The entity vector store is not snapshotted: it is a derived index of the
graph's entities and is rebuilt on resume.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path

from retrieval_engine.indexes.graph import Graph, SetEncoder

log = logging.getLogger(__name__)

STATE_FILENAME = "kg_refinement_state.json"
SCHEMA_VERSION = "docarag.kg-refinement-state.v1"


@dataclass(frozen=True)
class RefinementState:
    graph: Graph
    alias_map: dict[str, str]
    refined_node_ids: set[int]


def state_path(save_path: str | os.PathLike[str]) -> Path:
    return Path(save_path) / STATE_FILENAME


def save_state(
    save_path: str | os.PathLike[str],
    *,
    graph: Graph,
    alias_map: dict[str, str],
    refined_node_ids: set[int],
    variant: str | None = None,
) -> None:
    """Atomically replace the snapshot for this run."""

    target = state_path(save_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "variant": variant,
        "refined_node_ids": sorted(refined_node_ids),
        "alias_map": dict(alias_map),
        "graph_payload": graph.to_payload(),
    }

    temporary = target.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, cls=SetEncoder, ensure_ascii=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)


def load_state(
    save_path: str | os.PathLike[str], *, variant: str | None = None
) -> RefinementState | None:
    """Return a resumable snapshot, or None when there is nothing usable."""

    target = state_path(save_path)
    if not target.is_file():
        return None
    try:
        with target.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError):
        log.warning("Unreadable refinement snapshot at %s; starting over", target)
        return None

    if not isinstance(payload, dict):
        log.warning("Malformed refinement snapshot at %s; starting over", target)
        return None
    if payload.get("schema_version") != SCHEMA_VERSION:
        log.warning(
            "Refinement snapshot at %s has schema %r, not %r; starting over",
            target,
            payload.get("schema_version"),
            SCHEMA_VERSION,
        )
        return None
    if payload.get("variant") != variant:
        # A different embedder/reranker pairing produced these entities, so its
        # vectors and merge decisions do not carry over.
        log.warning(
            "Refinement snapshot at %s belongs to variant %r, not %r; starting over",
            target,
            payload.get("variant"),
            variant,
        )
        return None

    try:
        graph = Graph.from_payload(
            payload["graph_payload"], save_path=str(save_path), variant=variant
        )
        refined = {int(value) for value in payload["refined_node_ids"]}
        alias_map = {
            str(key): str(value) for key, value in payload["alias_map"].items()
        }
    except (KeyError, TypeError, ValueError):
        log.warning("Corrupt refinement snapshot at %s; starting over", target)
        return None

    log.info(
        "Resuming refinement from %s: %d nodes already refined, %d graph nodes",
        target,
        len(refined),
        len(graph.kg.nodes),
    )
    return RefinementState(
        graph=graph, alias_map=alias_map, refined_node_ids=refined
    )


def clear_state(save_path: str | os.PathLike[str]) -> None:
    """Drop the snapshot so a later run does not resume a finished stage."""

    state_path(save_path).unlink(missing_ok=True)
    state_path(save_path).with_suffix(".json.tmp").unlink(missing_ok=True)


__all__ = [
    "RefinementState",
    "SCHEMA_VERSION",
    "STATE_FILENAME",
    "clear_state",
    "load_state",
    "save_state",
    "state_path",
]
