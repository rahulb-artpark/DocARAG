"""The revision seal must be a digest of the rows, not of their column names.

``_seal_revision`` streams the tree through a server-side cursor and hashes
each row. The ledger's row factory is ``dict_row``, and iterating a mapping
yields its **keys** -- so every row contributed the same eleven column names,
and the seal was a pure function of how many nodes a revision had.

Measured on the live store before the fix: 52 sealed revisions held 34 distinct
seals, and one value was shared by ``10P_CH11.pdf``, ``312_LG_E_L3.pdf`` and
``physics_N.pdf`` -- three unrelated documents that happen to have 44 nodes
each. 30 revisions collided into 12 seals; recomputed by value, the same 30
hold 30.

This is not cosmetic. ``store_seal_sha256`` in the vector attestation binds the
vectors to the sealed rows -- "the binding truth: the vectors attest to the
sealed store rows" -- and query readiness refuses when the live seal differs.
A seal that cannot change means an edited tree still matches its old vectors,
so the index quietly answers from text the document no longer contains.
"""

from __future__ import annotations

import unittest
from contextlib import contextmanager

from document_services.persistence.postgres import PostgresLedger


class _Cursor:
    def __init__(self, rows):
        self._rows = rows
        self.itersize = None

    def execute(self, *_args):
        return self

    def __iter__(self):
        return iter(self._rows)


class _Connection:
    """Enough of a connection to seal a revision, returning mapping rows."""

    def __init__(self, rows):
        self._rows = rows
        self.sealed_with: str | None = None

    def execute(self, sql, params=None):
        if "INSERT INTO docarag.revision_seal" in sql:
            self.sealed_with = params[1]
        return self

    @contextmanager
    def cursor(self, name=None):
        yield _Cursor(self._rows)


def _row(node_no, summary_sha, retrieval_md5="feed", attrs_md5="attrs"):
    """One seal row in the shape the query returns it, keys and all."""

    return {
        "node_no": node_no,
        "parent_no": None,
        "node_kind": "text",
        "depth": 1,
        "sibling_no": node_no,
        "summary_sha256": summary_sha,
        "attrs_md5": attrs_md5,
        "retrieval_md5": retrieval_md5,
        "block_uid": f"block_{node_no}",
        "content_sha256": f"content_{node_no}",
    }


def _seal(rows):
    ledger = PostgresLedger.__new__(PostgresLedger)
    connection = _Connection(rows)
    PostgresLedger._seal_revision(ledger, connection, 1)
    return connection.sealed_with


class TheSealFollowsTheContent(unittest.TestCase):
    def test_an_edited_summary_changes_the_seal(self):
        before = _seal([_row(1, "aaa"), _row(2, "bbb")])
        after = _seal([_row(1, "aaa"), _row(2, "ccc")])

        self.assertNotEqual(before, after)

    def test_two_documents_of_equal_size_do_not_share_a_seal(self):
        # The live failure exactly: same node count, unrelated content.
        first = _seal([_row(1, "aaa"), _row(2, "bbb")])
        second = _seal([_row(1, "xxx"), _row(2, "yyy")])

        self.assertNotEqual(first, second)

    def test_changed_retrieval_text_changes_the_seal(self):
        before = _seal([_row(1, "aaa", retrieval_md5="one")])
        after = _seal([_row(1, "aaa", retrieval_md5="two")])

        self.assertNotEqual(before, after)

    def test_changed_node_attributes_change_the_seal(self):
        before = _seal([_row(1, "aaa", attrs_md5="one")])
        after = _seal([_row(1, "aaa", attrs_md5="two")])

        self.assertNotEqual(before, after)

    def test_the_same_rows_seal_the_same_way(self):
        self.assertEqual(
            _seal([_row(1, "aaa"), _row(2, "bbb")]),
            _seal([_row(1, "aaa"), _row(2, "bbb")]),
        )

    def test_reordering_the_tree_changes_the_seal(self):
        self.assertNotEqual(
            _seal([_row(1, "aaa"), _row(2, "bbb")]),
            _seal([_row(2, "bbb"), _row(1, "aaa")]),
        )

    def test_a_tuple_row_seals_identically_to_a_mapping_row(self):
        # The fix reads values from a mapping; a driver returning plain tuples
        # must not produce a different seal for the same rows.
        rows = [_row(1, "aaa"), _row(2, "bbb")]

        self.assertEqual(
            _seal(rows), _seal([tuple(row.values()) for row in rows])
        )

    def test_the_column_names_alone_do_not_decide_it(self):
        # The defect stated as a property: rows that differ only in their
        # values must not seal alike, whatever their keys are called.
        renamed = [
            {f"column_{index}": value for index, value in enumerate(_row(1, "zzz").values())}
        ]

        self.assertEqual(_seal([tuple(_row(1, "zzz").values())]), _seal(renamed))


if __name__ == "__main__":
    unittest.main()
