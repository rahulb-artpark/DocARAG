"""Edit stored content directly, with downstream consequences recorded.

An edit is a store write followed by a reseal: the revision's seal changes,
the vector attestation stops matching, and stage state records what must be
re-derived.  Nothing is re-run here -- the store says what is stale and
`docarag stage` re-runs exactly that.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from docarag.document_store import scope_for_run
from docarag.persistence.postgres import PostgresLedger
from docarag.settings import load_settings


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
    print("vdb is stale: run `docarag stage vdb --run <run>` to re-derive vectors")


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
