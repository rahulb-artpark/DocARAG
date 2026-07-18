from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from docarag.persistence.contracts import DocumentScope
from docarag.document_store import (
    DocumentStore,
    DocumentStoreError,
    scope_for_run,
)


IDENTITY = {
    "tenant_id": "default",
    "corpus_id": "default",
    "document_id": "doc_a",
    "document_version_id": "docv_a",
    "revision_id": "rev_a",
}


class _Connection:
    def __init__(self):
        self.closed = 0

    def execute(self, _query, _params=None):  # pragma: no cover - unused here
        raise AssertionError("no read was expected")

    def close(self):
        self.closed += 1


def run_dir(directory: Path, manifest: dict | None = None) -> Path:
    run = directory / "run"
    run.mkdir()
    (run / "run-manifest.json").write_text(
        json.dumps(IDENTITY if manifest is None else manifest), encoding="utf-8"
    )
    return run


class ScopeForRunTests(unittest.TestCase):
    def test_a_run_names_the_revision_it_published(self):
        with tempfile.TemporaryDirectory() as directory:
            scope, revision_id = scope_for_run(run_dir(Path(directory)))

        self.assertEqual("doc_a", scope.document_id)
        self.assertEqual("docv_a", scope.document_version_id)
        self.assertEqual("rev_a", revision_id)

    def test_an_unreadable_manifest_is_reported_not_guessed(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(DocumentStoreError, "missing or unreadable"):
                scope_for_run(Path(directory))

    def test_an_incomplete_manifest_names_what_it_is_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            partial = {k: v for k, v in IDENTITY.items() if k != "revision_id"}
            run = run_dir(Path(directory), partial)

            with self.assertRaises(DocumentStoreError) as caught:
                scope_for_run(run)

        self.assertIn("revision_id", str(caught.exception))


class DocumentStoreLifetimeTests(unittest.TestCase):
    def store(self):
        connection = _Connection()
        scope = DocumentScope("default", "default", "doc_a", "docv_a")
        return DocumentStore(scope, 7, connection), connection

    def test_the_tree_is_bound_to_the_resolved_revision(self):
        store, _connection = self.store()

        self.assertEqual(7, store.revision_key)
        self.assertEqual(7, store.tree.revision_key)

    def test_closing_releases_the_connection_once(self):
        store, connection = self.store()

        store.close()
        store.close()

        self.assertEqual(1, connection.closed)

    def test_a_closed_store_refuses_further_reads(self):
        # Several reads of one question share a connection, so using the store
        # after it is released has to fail loudly rather than reconnect and
        # answer from a different snapshot.
        store, _connection = self.store()
        store.close()

        with self.assertRaisesRegex(DocumentStoreError, "closed"):
            store.tree.root()

    def test_the_store_closes_when_its_block_ends(self):
        store, connection = self.store()

        with store:
            self.assertEqual(0, connection.closed)

        self.assertEqual(1, connection.closed)


if __name__ == "__main__":
    unittest.main()
