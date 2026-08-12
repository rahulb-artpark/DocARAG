"""Rewriting an entity's description touches that entity, in that document.

Two things make this narrower than it looks.

The entity collection is shared: every document's entities live in one Qdrant
collection, separated by payload scope. So a rebuild's ``reset()`` would take
this document's whole scope to change one entity, and a careless delete could
take another document's. ``reindex_entities`` neither resets nor deletes -- an
entity's point ID is derived from its name and metadata, not its description,
so a rewritten description keeps the ID and an upsert replaces the point.

And ``docarag.entity`` is corpus-scoped, one row per (tenant, corpus, canonical,
kind). Writing the description there would change every document mentioning the
entity: 730 of one paper's 862 entities are mentioned by another revision, and
'edgehd design' by five. The edit goes to the per-revision row instead, which is
also the row the entity index and the graph seal read.

Measured live: editing 'edgehd design' on one revision re-embedded 1 point in 3
seconds, left the other revision's wording and the corpus row untouched, and
left exactly one point in that revision's scope carrying the new text.
"""

from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from Core.Index.GBCIndex import GBC


def _records():
    return [
        (
            f"Name: entity-{index}\nType: TECHNOLOGY",
            {
                "entity_name": f"entity-{index}",
                "entity_type": "TECHNOLOGY",
                "description": f"description {index}",
                "node_id": f"Name: entity-{index}\nType: TECHNOLOGY",
            },
        )
        for index in range(4)
    ]


def _index(point_count=4):
    index = object.__new__(GBC)
    index.save_dir = "/tmp/run"
    index.entity_attestation_path = "/tmp/run/entity-attestation"
    index.entity_vdb = Mock()
    index.entity_vdb.count.return_value = point_count
    return index


class OnlyTheNamedEntityIsReEmbedded(unittest.TestCase):
    def _run(self, names, point_count=4):
        index = _index(point_count)
        with (
            patch.object(
                GBC, "_expected_entity_attestation",
                return_value={"point_count": 4},
            ),
            patch.object(GBC, "_entity_records", return_value=iter(_records())),
            patch("Core.Index.GBCIndex.vector_index_lock"),
            patch("Core.Index.GBCIndex.invalidate_vector_index_attestation"),
            patch("Core.Index.GBCIndex.publish_vector_index_attestation") as publish,
        ):
            written = index.reindex_entities(names)
        return index, written, publish

    def test_one_name_writes_one_point(self):
        index, written, _publish = self._run(["Name: entity-2\nType: TECHNOLOGY"])

        self.assertEqual(1, written)
        records = index.entity_vdb.add_entity_points.call_args.args[0]
        self.assertEqual(1, len(records))
        self.assertEqual("description 2", records[0]["metadata"]["description"])

    def test_the_shared_scope_is_never_reset(self):
        # reset() deletes this document's whole scope; the collection holds
        # every other document's entities beside it.
        index, _written, _publish = self._run(["Name: entity-0\nType: TECHNOLOGY"])

        index.entity_vdb.reset.assert_not_called()

    def test_nothing_is_deleted(self):
        index, _written, _publish = self._run(["Name: entity-0\nType: TECHNOLOGY"])

        index.entity_vdb.delete_points.assert_not_called()

    def test_it_refuses_before_writing_if_the_collection_cannot_hold_it(self):
        index = _index()
        index.entity_vdb.require_writable.side_effect = RuntimeError("unnamed vector")
        with (
            patch.object(
                GBC, "_expected_entity_attestation", return_value={"point_count": 4}
            ),
            patch("Core.Index.GBCIndex.invalidate_vector_index_attestation") as clear,
            self.assertRaisesRegex(RuntimeError, "unnamed vector"),
        ):
            index.reindex_entities(["Name: entity-0\nType: TECHNOLOGY"])

        clear.assert_not_called()
        index.entity_vdb.add_entity_points.assert_not_called()

    def test_an_entity_this_revision_does_not_mention_is_refused(self):
        with self.assertRaisesRegex(ValueError, "does not mention"):
            self._run(["Name: elsewhere\nType: TECHNOLOGY"])

    def test_no_names_is_refused_rather_than_silently_doing_nothing(self):
        index = _index()

        with self.assertRaisesRegex(ValueError, "at least one entity"):
            index.reindex_entities([])

    def test_a_scope_that_lost_points_is_caught(self):
        # The attestation's point count is the guard: a scope that no longer
        # holds every entity must not be attested as though it did.
        with self.assertRaisesRegex(ValueError, "expected 4"):
            self._run(["Name: entity-0\nType: TECHNOLOGY"], point_count=3)

    def test_the_attestation_is_republished(self):
        _index_, _written, publish = self._run(["Name: entity-1\nType: TECHNOLOGY"])

        publish.assert_called_once()


class TheEditIsScopedToOneRevision(unittest.TestCase):
    """Where the description is written, and where it deliberately is not."""

    @staticmethod
    def _source():
        from pathlib import Path

        from docarag import edit

        return Path(edit.__file__).read_text(encoding="utf-8")

    def test_the_write_targets_the_per_revision_row(self):
        self.assertIn("INSERT INTO docarag.entity_description", self._source())

    def test_the_corpus_row_is_not_written(self):
        # Writing docarag.entity would change every document sharing the entity.
        source = self._source()

        self.assertNotIn("UPDATE docarag.entity ", source)
        self.assertNotIn("INSERT INTO docarag.entity (", source)

    def test_the_graph_is_resealed(self):
        self.assertIn("_seal_graph", self._source())


if __name__ == "__main__":
    unittest.main()
