"""A revision's entities are searched by what that revision said about them.

``docarag.entity`` is corpus-scoped: one row per (tenant, corpus, canonical,
kind), so the same entity read by two documents keeps one description. Whichever
revision reached the row first owns it -- the writer's ``ON CONFLICT`` updates
``attrs`` and leaves ``description_blob_key`` alone -- and every other document
inherits that wording.

Measured on the live store, 605 of the Celestron EdgeHD paper's 862 entities
carried a description written while indexing something else. The entity index
embeds the description, so serving the corpus row would answer a question about
one document using another document's account of the same entity. After
recording descriptions per revision, all 862 match that paper's own extraction.

The corpus row still exists -- it is what makes an entity one entity across
documents, and it remains the fallback for an entity a revision never described.
"""

from __future__ import annotations

import unittest
from contextlib import contextmanager
from types import SimpleNamespace

from docarag.persistence.postgres import PostgresLedger
from docarag.tree_store import _GRAPH_ENTITIES_SQL


class _Cursor:
    def __init__(self, rows):
        self._rows = rows
        self.itersize = None

    def execute(self, *_args):
        return self

    def __iter__(self):
        return iter(self._rows)


class _Connection:
    def __init__(self, rows):
        self._rows = rows
        self.sealed_with: str | None = None
        self.statements: list[str] = []

    def execute(self, sql, params=None):
        self.statements.append(sql)
        if "INSERT INTO docarag.revision_seal" in sql:
            self.sealed_with = params[1]
        return self

    @contextmanager
    def cursor(self, name=None):
        yield _Cursor(self._rows)


def _entity(canonical, kind="TECHNOLOGY", description_sha="d" * 64):
    """One graph-seal row as the query returns it."""

    return {
        "canonical": canonical,
        "entity_kind": kind,
        "description_sha256": description_sha,
    }


def _seal(rows):
    ledger = PostgresLedger.__new__(PostgresLedger)
    connection = _Connection(rows)
    PostgresLedger._seal_graph(ledger, connection, 1)
    return connection.sealed_with


class TheGraphSealFollowsItsEntities(unittest.TestCase):
    def test_an_edited_description_changes_the_seal(self):
        before = _seal([_entity("edgehd", description_sha="a" * 64)])
        after = _seal([_entity("edgehd", description_sha="b" * 64)])

        self.assertNotEqual(before, after)

    def test_a_new_entity_changes_the_seal(self):
        self.assertNotEqual(
            _seal([_entity("edgehd")]),
            _seal([_entity("edgehd"), _entity("corrector")]),
        )

    def test_the_entity_kind_is_part_of_it(self):
        self.assertNotEqual(
            _seal([_entity("edgehd", kind="TECHNOLOGY")]),
            _seal([_entity("edgehd", kind="SECTION_TITLE")]),
        )

    def test_the_same_entities_seal_the_same_way(self):
        self.assertEqual(_seal([_entity("edgehd")]), _seal([_entity("edgehd")]))

    def test_a_described_entity_and_an_undescribed_one_differ(self):
        self.assertNotEqual(
            _seal([_entity("edgehd", description_sha=None)]),
            _seal([_entity("edgehd", description_sha="a" * 64)]),
        )

    def test_it_is_written_to_the_graph_column(self):
        connection = _Connection([_entity("edgehd")])
        PostgresLedger._seal_graph(PostgresLedger.__new__(PostgresLedger), connection, 1)
        insert = next(s for s in connection.statements if "revision_seal" in s)

        self.assertIn("graph_sha256", insert)
        self.assertNotIn("tree_sha256 = EXCLUDED", insert)

    def test_a_mapping_row_is_digested_by_value(self):
        # The tree seal digested its own column names for exactly this reason.
        rows = [_entity("edgehd")]

        self.assertEqual(_seal(rows), _seal([tuple(row.values()) for row in rows]))


class EveryEntityReachesTheIndex(unittest.TestCase):
    """An index built from a display cap cannot answer about what it dropped."""

    def _store(self, count):
        from docarag.tree_store import TreeStore

        rows = [
            {
                "canonical": f"entity-{index:05d}",
                "entity_kind": "TECHNOLOGY",
                "description": f"description {index}",
            }
            for index in range(count)
        ]

        class _Paged:
            def execute(self, _sql, params):
                after, kind, limit = params[3], params[4], params[5]
                page = [
                    row
                    for row in rows
                    if (row["canonical"], row["entity_kind"]) > (after, kind)
                ][:limit]
                return SimpleNamespace(fetchall=lambda: page)

        # Through the real connection factory: TreeStore uses __slots__, and
        # the paging under test is in iter_graph_entities, not in the session.
        return TreeStore(lambda: _Paged(), 1), rows

    def test_more_entities_than_one_page_are_all_returned(self):
        store, rows = self._store(4500)

        streamed = list(store.iter_graph_entities(page_size=1000))

        self.assertEqual(len(rows), len(streamed))

    def test_no_entity_is_yielded_twice(self):
        store, _rows = self._store(4500)

        streamed = list(store.iter_graph_entities(page_size=1000))
        keys = [(row["canonical"], row["entity_kind"]) for row in streamed]

        self.assertEqual(len(keys), len(set(keys)))

    def test_they_arrive_in_identity_order(self):
        store, _rows = self._store(2500)

        streamed = list(store.iter_graph_entities(page_size=500))
        keys = [(row["canonical"], row["entity_kind"]) for row in streamed]

        self.assertEqual(sorted(keys), keys)

    def test_an_exact_multiple_of_the_page_size_terminates(self):
        store, _rows = self._store(2000)

        self.assertEqual(2000, len(list(store.iter_graph_entities(page_size=1000))))

    def test_an_empty_revision_yields_nothing(self):
        store, _rows = self._store(0)

        self.assertEqual([], list(store.iter_graph_entities(page_size=1000)))

    def test_a_page_size_must_be_positive(self):
        store, _rows = self._store(10)

        with self.assertRaisesRegex(ValueError, "at least 1"):
            list(store.iter_graph_entities(page_size=0))


class TheStoreServesThisRevisionsWording(unittest.TestCase):
    def test_the_revisions_own_description_is_preferred(self):
        # COALESCE order is the whole behaviour: own first, corpus as fallback.
        self.assertIn("COALESCE(own_blob.body, blob.body)", _GRAPH_ENTITIES_SQL)

    def test_the_lookup_is_scoped_to_the_revision(self):
        self.assertIn("own.revision_key = %s", _GRAPH_ENTITIES_SQL)

    def test_the_corpus_row_remains_the_fallback(self):
        # An entity this revision never described still gets a description
        # rather than an empty vector.
        self.assertIn("LEFT JOIN docarag.entity_description own", _GRAPH_ENTITIES_SQL)
        self.assertIn("blob.blob_key = ent.description_blob_key", _GRAPH_ENTITIES_SQL)


if __name__ == "__main__":
    unittest.main()
