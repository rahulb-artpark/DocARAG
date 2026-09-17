"""Open one run's store: resolve its identity, then serve bounded reads.

A run directory names the revision it published.  This turns that name into the
surrogate the evidence tables are keyed by, opens one connection for the life of
a query, and hands back a :class:`TreeStore` over the slice.

The connection is owned here, not by the store, so several reads belonging to
one question share a transaction and see a consistent slice.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from document_services.persistence.contracts import DocumentScope
from document_services.persistence.postgres import PostgresLedger, resolve_database_url
from document_services.persistence.tree_store import TreeStore

RUN_MANIFEST = "run-manifest.json"


class DocumentStoreError(RuntimeError):
    """A run could not be bound to stored evidence."""


def scope_for_run(run_dir: str | Path) -> tuple[DocumentScope, str]:
    """Return the ledger scope and revision a run directory published."""

    path = Path(run_dir).expanduser().resolve() / RUN_MANIFEST
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise DocumentStoreError(f"{RUN_MANIFEST} is missing or unreadable") from exc
    if not isinstance(manifest, dict):
        raise DocumentStoreError(f"{RUN_MANIFEST} must be an object")

    missing = [
        name
        for name in (
            "tenant_id",
            "corpus_id",
            "document_id",
            "document_version_id",
            "revision_id",
        )
        if not isinstance(manifest.get(name), str) or not manifest[name].strip()
    ]
    if missing:
        raise DocumentStoreError(
            f"{RUN_MANIFEST} does not identify this run: missing "
            + ", ".join(missing)
        )
    scope = DocumentScope(
        tenant_id=manifest["tenant_id"],
        corpus_id=manifest["corpus_id"],
        document_id=manifest["document_id"],
        document_version_id=manifest["document_version_id"],
    )
    return scope, manifest["revision_id"]


class DocumentStore:
    """A store bound to one run, holding the connection its reads share."""

    __slots__ = ("_connection", "_closed", "revision_key", "revision_id", "run_dir", "scope", "tree")

    def __init__(
        self,
        scope: DocumentScope,
        revision_key: int,
        connection: Any,
        revision_id: str = "",
        run_dir: Path | None = None,
    ) -> None:
        self.scope = scope
        self.revision_key = revision_key
        self.revision_id = revision_id
        self.run_dir = run_dir
        self._connection = connection
        self._closed = False
        self.tree = TreeStore(self._borrow, revision_key, scope.document_version_id)

    @classmethod
    def open(
        cls,
        run_dir: str | Path,
        *,
        database_url: str | None = None,
        ledger: PostgresLedger | None = None,
    ) -> "DocumentStore":
        """Bind a run directory to the evidence its revision wrote."""

        scope, revision_id = scope_for_run(run_dir)
        url = resolve_database_url(database_url)
        revision_key = (ledger or PostgresLedger(url)).revision_key(scope, revision_id)
        return cls(
            scope,
            revision_key,
            _open_connection(url),
            revision_id=revision_id,
            run_dir=Path(run_dir).expanduser().resolve(),
        )

    def evidence_catalog(self):
        """Build the canonical evidence catalog from the store's own rows."""

        from pathlib import PurePosixPath

        from document_services.evidence.hydration import (
            CanonicalEvidenceCatalog,
            EvidenceHydrationError,
        )
        from document_services.domain.evidence import CanonicalBlock, DocumentRef

        connection = self._borrow()
        header = connection.execute(
            "SELECT canonical_document FROM docarag.revision_document "
            "WHERE revision_key = %s",
            (self.revision_key,),
        ).fetchone()
        document_payload = (header or {}).get("canonical_document")
        if not isinstance(document_payload, dict):
            raise EvidenceHydrationError(
                "this revision has no canonical document header; re-run "
                "evidence persistence"
            )
        rows = connection.execute(
            "SELECT blob.body FROM docarag.block blk "
            "JOIN docarag.text_blob blob ON blob.blob_key = blk.canonical_blob_key "
            "WHERE blk.revision_key = %s ORDER BY blk.block_no",
            (self.revision_key,),
        ).fetchall()
        blocks = tuple(
            CanonicalBlock.from_dict(json.loads(row["body"])) for row in rows
        )
        if not blocks:
            raise EvidenceHydrationError(
                "this revision has no canonical blocks in the store; re-run "
                "evidence persistence"
            )
        return CanonicalEvidenceCatalog(
            artifact_path=PurePosixPath(f"postgres://revision/{self.revision_id}"),
            document=DocumentRef.from_dict(document_payload),
            blocks=blocks,
        )

    def _borrow(self) -> Any:
        if self._closed:
            raise DocumentStoreError("document store is closed")
        return self._connection

    def connect(self) -> Any:
        """The store's own connection, for a corpus-scoped reader sharing it."""

        return self._borrow()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        close = getattr(self._connection, "close", None)
        if callable(close):
            close()

    def __enter__(self) -> "DocumentStore":
        return self

    def __exit__(self, *_exc: object) -> bool:
        self.close()
        return False


def _open_connection(database_url: str) -> Any:
    """Open one read connection with mapping rows, which the store expects."""

    try:
        import psycopg
        from psycopg.rows import dict_row
    except ImportError as exc:  # pragma: no cover - depends on installation
        raise DocumentStoreError(
            "reading stored evidence requires psycopg 3"
        ) from exc
    # Reads are single-statement slices over sealed evidence.  Autocommit
    # keeps the connection out of "idle in transaction" between them, which
    # would hold locks for the life of the viewer and block any DDL.
    return psycopg.connect(
        database_url,
        application_name="docarag-read",
        row_factory=dict_row,
        autocommit=True,
    )


__all__ = ["DocumentStore", "DocumentStoreError", "scope_for_run"]
