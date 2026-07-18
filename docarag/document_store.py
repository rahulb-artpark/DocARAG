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

from docarag.persistence.contracts import DocumentScope
from docarag.persistence.postgres import PostgresLedger, resolve_database_url
from docarag.tree_store import TreeStore

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

    __slots__ = ("_connection", "_closed", "revision_key", "scope", "tree")

    def __init__(
        self,
        scope: DocumentScope,
        revision_key: int,
        connection: Any,
    ) -> None:
        self.scope = scope
        self.revision_key = revision_key
        self._connection = connection
        self._closed = False
        self.tree = TreeStore(self._borrow, revision_key)

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
        return cls(scope, revision_key, _open_connection(url))

    def _borrow(self) -> Any:
        if self._closed:
            raise DocumentStoreError("document store is closed")
        return self._connection

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
    return psycopg.connect(
        database_url,
        application_name="docarag-read",
        row_factory=dict_row,
    )


__all__ = ["DocumentStore", "DocumentStoreError", "scope_for_run"]
