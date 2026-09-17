"""Verify the tree a run will actually be answered from.

Two things carry a tree out of the tree stage, and only one of them is read
afterwards.  The run directory keeps ``tree.pkl`` and ``tree.json`` as build
artifacts, while the graph and vector stages open the store
(:class:`document_services.persistence.vector_tree_source.StoreTreeSource`) and the query path opens
:class:`docarag.rag.stored_index.StoredIndex`.  This module checks both, but
they answer different questions:

* ``artifacts`` -- the tree stage wrote a structurally sound tree and bound it
  to the source it claims.  Cheap, and it runs before anything downstream has
  consumed the tree.
* ``store`` -- *the rows the next stage will read* are sound.  This is the
  check that matters, because a legacy artifact is allowed to disagree with
  the store and did: every chart and diagram in a document reached the graph
  stage holding ``cas://sha256/<digest>`` where a filename belonged, lost all
  of its entities, and nothing failed.  ``tree.pkl`` held a clean local path
  the whole time, so comparing the two artifacts to each other said the tree
  was fine.

An artifact check can only ever prove the artifacts agree with each other.
The store check proves the tree is usable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import sys
from typing import Any

# Every import the store check needs is deferred into the functions that need
# it: the artifact check runs under the controller's dependency-light
# interpreter, which has no BookRAG on its path at all.
#
# The kinds MinerU always crops an image for.  A chart the store holds no
# openable image for is the failure this module exists to catch, so their
# absence is an error rather than a note; a table or an equation legitimately
# carries only text.
VISUAL_NODE_TYPES = frozenset({"image", "chart", "diagram"})


class TreeIntegrityError(ValueError):
    """Raised when the two tree representations cannot be trusted together."""


def _json_canonical(value: Any) -> Any:
    """Return ``value`` reduced to the JSON types ``tree.json`` is stored as.

    ``tree.json`` is written with ``json.dump`` and reloaded with ``json.load``,
    so tuples (e.g. MinerU ``bbox`` coordinates carried in node metadata) become
    lists and any non-string mapping keys become strings.  The pickle preserves
    the original Python objects, so the two must be compared through the same
    JSON normalization; otherwise a benign representation difference is misread
    as tree corruption.  Any real structural or content divergence still fails.
    """

    return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True))


def _metadata(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        payload = model_dump()
    elif isinstance(value, dict):
        payload = value
    else:
        payload = vars(value)
    if not isinstance(payload, dict):
        raise TreeIntegrityError("tree metadata is not an object")
    return payload


def _summary_from_tree(tree: Any) -> dict[str, Any]:
    serializer = getattr(tree, "to_json_summary", None)
    if callable(serializer):
        payload = serializer()
        if not isinstance(payload, dict):
            raise TreeIntegrityError("tree pickle serializer did not return an object")
        return payload

    # A duck-typed fallback keeps the integrity checker independently testable
    # without weakening the checks used for the real DocumentTree.
    nodes = getattr(tree, "nodes", None)
    if not isinstance(nodes, list):
        raise TreeIntegrityError("tree pickle has no node list")
    summaries: list[dict[str, Any]] = []
    for node in nodes:
        parent = getattr(node, "parent", None)
        node_type = getattr(node, "type", None)
        summaries.append(
            {
                "index_id": getattr(node, "index_id", None),
                "parent_id": getattr(parent, "index_id", None) if parent else None,
                "type": str(node_type) if node_type else None,
                "meta_info": _metadata(getattr(node, "meta_info", None)),
                "summary": getattr(node, "summary", ""),
            }
        )
    return {"nodes": summaries, "meta_info": _metadata(getattr(tree, "meta_info", None))}


def _validate_topology(nodes: object) -> None:
    if not isinstance(nodes, list) or not nodes:
        raise TreeIntegrityError("tree does not contain a non-empty nodes list")
    ids: list[int] = []
    parents: dict[int, int | None] = {}
    for item in nodes:
        if not isinstance(item, dict):
            raise TreeIntegrityError("tree contains a non-object node")
        node_id = item.get("index_id")
        parent_id = item.get("parent_id")
        if not isinstance(node_id, int) or isinstance(node_id, bool):
            raise TreeIntegrityError("tree node index_id is not an integer")
        if parent_id is not None and (
            not isinstance(parent_id, int) or isinstance(parent_id, bool)
        ):
            raise TreeIntegrityError(f"tree node {node_id} has an invalid parent_id")
        ids.append(node_id)
        parents[node_id] = parent_id
    if len(ids) != len(set(ids)):
        raise TreeIntegrityError("tree contains duplicate node IDs")
    if set(ids) != set(range(len(ids))):
        raise TreeIntegrityError("tree node IDs are not contiguous from zero")
    if parents.get(0) is not None:
        raise TreeIntegrityError("tree root node 0 has a parent")
    known = set(ids)
    for node_id, parent_id in parents.items():
        if node_id != 0 and parent_id not in known:
            raise TreeIntegrityError(f"tree node {node_id} references a missing parent")
        visited: set[int] = set()
        current: int | None = node_id
        while current is not None:
            if current in visited:
                raise TreeIntegrityError(f"tree contains a parent cycle at node {node_id}")
            visited.add(current)
            current = parents.get(current)


def verify_tree_pair(
    run_dir: Path,
    *,
    expected_source_sha256: str,
    expected_source_uri: str,
    expected_snapshot_path: Path,
    expected_source_name: str,
) -> None:
    """Verify topology, exact pickle/JSON agreement, and source identity."""

    run = run_dir.expanduser().resolve()
    pickle_path = run / "tree.pkl"
    json_path = run / "tree.json"
    for path in (pickle_path, json_path):
        if path.is_symlink() or not path.is_file() or path.stat().st_size == 0:
            raise TreeIntegrityError(f"tree artifact is missing, empty, or a symlink: {path}")

    try:
        with pickle_path.open("rb") as source:
            tree = pickle.load(source)
    except Exception as exc:  # Pickle/library failures are integrity failures.
        raise TreeIntegrityError(f"tree.pkl cannot be loaded: {exc}") from exc
    try:
        json_summary = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TreeIntegrityError(f"tree.json cannot be loaded: {exc}") from exc

    pickle_summary = _summary_from_tree(tree)
    _validate_topology(pickle_summary.get("nodes"))
    _validate_topology(json_summary.get("nodes") if isinstance(json_summary, dict) else None)
    if _json_canonical(pickle_summary) != _json_canonical(json_summary):
        raise TreeIntegrityError("tree.pkl and tree.json describe different trees")

    expected_snapshot = expected_snapshot_path.expanduser().resolve(strict=True)
    expected = {
        "file_name": expected_source_name,
        "source_uri": expected_source_uri,
        "source_sha256": expected_source_sha256,
        "snapshot_path": str(expected_snapshot),
    }
    for representation, metadata in (
        ("tree.pkl", _metadata(getattr(tree, "meta_info", None))),
        ("tree.json", json_summary.get("meta_info", {})),
    ):
        if not isinstance(metadata, dict):
            raise TreeIntegrityError(f"{representation} source metadata is not an object")
        actual = {key: metadata.get(key) for key in expected}
        if actual != expected:
            raise TreeIntegrityError(
                f"{representation} source identity differs: expected {expected!r}, got {actual!r}"
            )


def _stored_summaries(views: Any) -> list[dict[str, Any]]:
    """The topology fields, taken from the rows rather than from a file."""

    return [
        {
            "index_id": view.index_id,
            "parent_id": view.parent_id,
            "type": str(view.type) if view.type else None,
        }
        for view in views
    ]


def _resolve_image(path: str, run: Path) -> Path | None:
    """Return the image ``path`` names, whether it is absolute or run-relative."""

    candidate = Path(path).expanduser()
    if candidate.is_file():
        return candidate
    if not candidate.is_absolute():
        relative = (run / candidate).resolve()
        if relative.is_file():
            return relative
    return None


def _image_faults(views: Any, run: Path) -> tuple[list[str], list[str]]:
    """Separate broken image references from figures the parser never cropped.

    These are different failures and deserve different answers.  A reference
    that exists but cannot be opened is corruption: something wrote an image
    into the store that nothing can read, which is exactly what cost a document
    its chart and diagram entities, and it is never correct.  A visual node with
    no reference at all is a parser outcome -- a real 500-page book has one
    captioned figure MinerU produced no crop for -- and failing the whole index
    over it would trade a silent loss for a blocked run, which is the worse of
    the two.  So the first is an error and the second is reported.
    """

    from document_services.persistence.vector_tree_source import asset_cas_root, openable_image

    # Resolved exactly as the stages resolve it, so the gate passes a tree if
    # and only if the next stage can open its figures.
    cas_root = asset_cas_root(run)
    broken: list[str] = []
    uncropped: list[str] = []
    for view in views:
        kind = (str(view.type) if view.type else "").lower()
        if kind not in VISUAL_NODE_TYPES:
            continue
        evidence = dict(getattr(view, "evidence", None) or {})
        candidates = (evidence.get("img_path"), view.meta_info.img_path)
        path = openable_image(*candidates, cas_root=cas_root)
        if not path:
            named = [item for item in candidates if isinstance(item, str) and item]
            if named:
                broken.append(
                    f"node {view.index_id} ({kind}) carries an image reference "
                    f"nothing can open: {named[0]!r}"
                )
            else:
                uncropped.append(f"node {view.index_id} ({kind})")
            continue
        if _resolve_image(path, run) is None:
            broken.append(f"node {view.index_id} ({kind}) names a missing image: {path}")
    return broken, uncropped


def verify_stored_tree(run_dir: Path) -> None:
    """Verify the tree as the store holds it, which is the tree that is read."""

    from document_services.persistence.document_store import DocumentStore

    run = run_dir.expanduser().resolve()
    try:
        store = DocumentStore.open(str(run))
    except Exception as exc:
        # A refused connection and an unbound revision both mean the same thing
        # to this gate: the tree the next stage will read cannot be seen, so
        # nothing about it has been verified. Reported, never passed over.
        raise TreeIntegrityError(f"the stored tree cannot be opened: {exc}") from exc
    try:
        views = store.tree.nodes()
    except Exception as exc:  # A store that cannot be read is an integrity failure.
        raise TreeIntegrityError(f"the stored tree cannot be read: {exc}") from exc
    finally:
        store.close()

    if not views:
        raise TreeIntegrityError(
            "the store holds no tree for this revision; the tree stage's output "
            "was never persisted"
        )
    _validate_topology(_stored_summaries(views))
    broken, uncropped = _image_faults(views, run)
    if uncropped:
        # Stated, so the loss is known rather than discovered later in an
        # answer that quietly had nothing to show.
        print(
            f"tree integrity note: {len(uncropped)} visual node(s) have no image "
            "and will contribute only their caption: " + ", ".join(uncropped),
            file=sys.stderr,
        )
    if broken:
        raise TreeIntegrityError(
            "the stored tree carries images no stage can open:\n  "
            + "\n  ".join(broken)
        )


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify BookRAG tree artifacts")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument(
        "--check",
        choices=("artifacts", "store", "all"),
        default="artifacts",
        help=(
            "artifacts: the tree stage's own output is sound. "
            "store: the rows the next stage reads are sound. "
            "all: both."
        ),
    )
    parser.add_argument("--source-sha256")
    parser.add_argument("--source-uri")
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--source-name")
    args = parser.parse_args(argv)
    try:
        if args.check in ("store", "all"):
            verify_stored_tree(args.run)
        if args.check == "store":
            return 0
        if not all(
            (args.source_sha256, args.source_uri, args.snapshot, args.source_name)
        ):
            manifest = json.loads(
                (args.run / "run-manifest.json").read_text(encoding="utf-8")
            )
            source = manifest.get("source") if isinstance(manifest, dict) else None
            if not isinstance(source, dict):
                raise TreeIntegrityError("run manifest has no source identity")
            args.source_sha256 = source.get("sha256")
            args.source_uri = source.get("uri") or source.get("path")
            snapshot = source.get("snapshot_path")
            args.snapshot = Path(snapshot) if isinstance(snapshot, str) else None
            source_path = source.get("path")
            args.source_name = (
                Path(source_path).name if isinstance(source_path, str) else None
            )
        if not (
            isinstance(args.source_sha256, str)
            and isinstance(args.source_uri, str)
            and isinstance(args.snapshot, Path)
            and isinstance(args.source_name, str)
        ):
            raise TreeIntegrityError("expected source identity is incomplete")
        verify_tree_pair(
            args.run,
            expected_source_sha256=args.source_sha256,
            expected_source_uri=args.source_uri,
            expected_snapshot_path=args.snapshot,
            expected_source_name=args.source_name,
        )
    except (OSError, TreeIntegrityError) as exc:
        print(f"tree integrity error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
