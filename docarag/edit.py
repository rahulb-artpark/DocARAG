"""Edit stored content directly, and re-derive exactly what the edit reached.

An edit is a store write, a reseal, and a reconcile. The write and the reseal
are the easy part; the reconcile is what keeps the document answerable, because
a resealed revision whose vectors still describe the old text fails closed at
query time with "TreeVDB attestation does not match the sealed store revision".

Only the points whose text actually changed are re-embedded. Which points those
are is not worked out here: a tree point's ID is content-addressed over every
vector's source, so re-projecting the tree and diffing IDs against Qdrant finds
them, including the ones an edit reached indirectly -- a section heading's
summary is quoted into the ``context`` vector of every figure and table beneath
it, and those points re-ID themselves.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from docarag.document_store import scope_for_run
from docarag.persistence.postgres import PostgresLedger
from docarag.settings import load_settings


def _reconcile_run_vectors(run: Path) -> str:
    """Re-derive the run's tree vectors in place and republish its manifest."""

    # Imported here: this module is the thin edit surface, and the vector stack
    # pulls in model providers that a caller reading a summary should not pay
    # for.
    import yaml

    from Core.configs.system_config import SystemConfig
    from Core.pipelines.vdb_index import (
        expected_tree_vector_index_attestation,
        reconcile_tree_vector_index,
    )
    from Core.utils.vector_index_manifest import (
        publish_vector_index_attestation,
        read_vector_index_attestation,
        vector_index_lock,
    )
    from docarag.pipeline import republish_run_artifacts
    from docarag.vdb_tree_source import StoreTreeSource

    from Core.utils.artifact_paths import resolve_run_artifact_path

    config = SystemConfig(**yaml.safe_load((run / "run_config.yaml").read_text()))
    vdb_cfg = config.vdb
    # The config names the attestation directory relative to the run, and the
    # stage resolves it before use. Reading it raw finds nothing and reports a
    # missing index rather than the wrong path.
    directory = resolve_run_artifact_path(
        config.save_path, vdb_cfg.manifest_dir_name, label="vdb.manifest_dir_name"
    )
    vdb_cfg.manifest_dir_name = str(directory)
    with vector_index_lock(directory, exclusive=True):
        previous = read_vector_index_attestation(directory)
        if not isinstance(previous, dict) or not previous.get("lexical_average_length"):
            raise SystemExit(
                "error: no attested vector index to reconcile against; "
                "run `docarag stage vdb --run <run>` first"
            )
        frozen = float(previous["lexical_average_length"])
        tree = StoreTreeSource.open(str(run))
        outcome = reconcile_tree_vector_index(
            tree, vdb_cfg, frozen_average_length=frozen
        )
        expected = expected_tree_vector_index_attestation(
            tree, vdb_cfg, frozen_average_length=frozen
        )
        publish_vector_index_attestation(directory, expected)
    published = republish_run_artifacts(run)
    if published != 0:
        raise SystemExit(f"error: artifact republication failed ({published})")
    return outcome.describe() + (
        f"; BM25 length normalizer held at {outcome.average_length:.3f} "
        f"(corpus {outcome.corpus_average_length:.3f}, drift {outcome.drift:.1%})"
    )


def edit_node_summary(run: Path, node_no: int, text: str) -> None:
    scope, revision_id = scope_for_run(run)
    ledger = PostgresLedger(load_settings().resolve_database_url())
    revision_key = ledger.revision_key(scope, revision_id)
    with ledger._connection() as connection:
        blob_key = ledger._intern_text(connection, text)
        updated = connection.execute(
            "UPDATE docarag.tree_node SET summary_blob_key = %s "
            "WHERE revision_key = %s AND node_no = %s",
            (blob_key, revision_key, node_no),
        ).rowcount
        if updated != 1:
            raise SystemExit(f"error: revision has no tree node {node_no}")
        ledger._seal_revision(connection, revision_key)
        # Stale until the reconcile below says otherwise. If this process dies
        # in between, the store says the vectors describe the old text, which
        # is exactly what happened.
        connection.execute(
            "INSERT INTO docarag.stage_state (revision_key, stage, status) "
            "VALUES (%s, 'vdb', 'stale') "
            "ON CONFLICT (revision_key, stage) DO UPDATE SET status = 'stale'",
            (revision_key,),
        )
        seal = str(
            connection.execute(
                "SELECT tree_sha256 FROM docarag.revision_seal "
                "WHERE revision_key = %s",
                (revision_key,),
            ).fetchone()["tree_sha256"]
        )
    print(f"node {node_no} summary updated; revision resealed {seal[:16]}…")

    report = _reconcile_run_vectors(run)
    with ledger._connection() as connection:
        connection.execute(
            "INSERT INTO docarag.stage_state (revision_key, stage, status) "
            "VALUES (%s, 'vdb', 'succeeded') "
            "ON CONFLICT (revision_key, stage) DO UPDATE SET status = 'succeeded'",
            (revision_key,),
        )
    print(report)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Edit stored content in place")
    sub = parser.add_subparsers(dest="what", required=True)
    node = sub.add_parser("summary", help="Replace one tree node's summary")
    node.add_argument("--run", type=Path, required=True)
    node.add_argument("--node", type=int, required=True)
    node.add_argument("--text", required=True)
    args = parser.parse_args(argv)
    edit_node_summary(args.run, args.node, args.text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
