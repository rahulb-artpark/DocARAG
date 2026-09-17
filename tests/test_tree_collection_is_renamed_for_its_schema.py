"""The tree collection's name carries its vector schema.

Qdrant cannot convert an unnamed vector into named ones, and the tree
collection is shared across documents by payload scope, so `reset()` deletes
one scope rather than the collection -- dropping it would take every other
document's vectors with it. The entity store hit exactly this and solved it by
putting the schema in the name; the same reasoning applies here.

Two things this pins. The tree configs point at the new name, so a rebuild
writes a new collection and the old one survives as a rollback. And the
collections that keep the single unnamed vector are *not* renamed, because a
rename there would silently start a second empty index.
"""

from __future__ import annotations

import pathlib
import unittest

import yaml

CONFIG = pathlib.Path(__file__).resolve().parent.parent / "config"
#: Collections that are not the tree index and keep one unnamed vector.
UNNAMED_COLLECTIONS = {"PDFVanillaVDB", "VanillaVDB", "RaptorVDB", "BM25VDB"}
#: index_type values routed to build_other_vdb_index, which writes chunks with
#: one unnamed vector and never touches the tree projection.
CHUNK_INDEX_TYPES = {"vanilla", "bm25", "raptor", "pdf_vanilla"}


def _tree_collection_names():
    """Every collection name a *tree* index is written to or read from.

    A config whose index_type routes to the chunk writer is not one of these:
    its vdb.collection_name names a chunk collection that keeps the single
    unnamed vector.
    """

    for path in sorted(CONFIG.glob("*.yaml")):
        document = yaml.safe_load(path.read_text()) or {}
        if document.get("index_type") in CHUNK_INDEX_TYPES:
            continue
        vdb = document.get("vdb")
        if isinstance(vdb, dict) and vdb.get("collection_name"):
            yield path.name, "vdb", vdb["collection_name"]
        rag = document.get("rag")
        if not isinstance(rag, dict):
            continue
        nested = rag.get("vdb_config")
        if isinstance(nested, dict) and nested.get("collection_name"):
            yield path.name, "rag.vdb_config", nested["collection_name"]


class TheSchemaIsInTheName(unittest.TestCase):
    def test_every_tree_collection_declares_the_hybrid_schema(self):
        for name, section, collection in _tree_collection_names():
            with self.subTest(config=name, section=section):
                self.assertTrue(
                    collection.endswith("-hybrid-v1"),
                    f"{name}:{section} still points at {collection!r}; a rebuild "
                    "would write named vectors into an unnamed collection",
                )

    def test_the_active_config_names_the_embedder_and_the_schema(self):
        document = yaml.safe_load((CONFIG / "ollama-gbc.yaml").read_text())

        self.assertEqual(
            "TreeVDB-qwen3vl-2048-hybrid-v1", document["vdb"]["collection_name"]
        )

    def test_no_tree_config_still_uses_a_pre_migration_name(self):
        stale = {"TreeVDB", "TreeVDB-qwen3vl-2048"}
        found = {c for _, _, c in _tree_collection_names()} & stale

        self.assertEqual(set(), found)


class TheOtherCollectionsAreLeftAlone(unittest.TestCase):
    def test_the_unnamed_collections_keep_their_names(self):
        # Renaming one of these would start a second, empty index rather than
        # migrating anything: nothing about their shape has changed.
        seen = set()
        for path in sorted(CONFIG.glob("*.yaml")):
            document = yaml.safe_load(path.read_text()) or {}
            for section in (document.get("vdb"), *(document.get("rag") or {}).values()):
                if isinstance(section, dict):
                    name = section.get("collection_name")
                    if name in UNNAMED_COLLECTIONS:
                        seen.add(name)

        self.assertEqual(UNNAMED_COLLECTIONS, seen)

    def test_the_entity_collection_is_untouched(self):
        # It made this same migration already, under its own name, and the
        # graph stage that writes it is out of scope here.
        from retrieval_engine.configuration.knowledge_graph import GraphConfig

        self.assertEqual(
            "bookrag-entities-hybrid-v1", GraphConfig().entity_collection_name
        )


if __name__ == "__main__":
    unittest.main()
