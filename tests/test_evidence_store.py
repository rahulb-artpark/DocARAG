from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from docarag.evidence_projection import build_document_evidence
from docarag.persistence.contracts import (
    AssetRecord,
    BlockAssetLink,
    BlockRecord,
    DocumentEvidence,
    DocumentScope,
    EntityRecord,
    GTLinkRecord,
    RelationRecord,
    RepresentationRecord,
    TreeNodeRecord,
)
from docarag.persistence.postgres import PostgresLedger


class _FakeCursor:
    def __init__(self, rows):
        self._rows = rows

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def fetchall(self):
        return tuple(self._rows)


class _FakeConnection:
    def __init__(self, handler=None):
        self.executions = []
        self._handler = handler or (lambda q, p: ())

    def execute(self, query, params=None):
        self.executions.append((query, params))
        return _FakeCursor(list(self._handler(query, params)))


class EvidenceRepositoryTests(unittest.TestCase):
    def _evidence(self) -> DocumentEvidence:
        return DocumentEvidence(
            index_revision_id="rev1",
            assets=(AssetRecord("c" * 64, "image/jpeg", 10, "cas://sha256/" + "c" * 64),),
            blocks=(
                BlockRecord(
                    block_id="b0",
                    revision_id="r0",
                    block_type="table",
                    page_index=0,
                    representations=(RepresentationRecord(kind="table_html", content="<t/>"),),
                    asset_links=(BlockAssetLink(asset_id="c" * 64, role="table_crop"),),
                ),
            ),
            tree_nodes=(TreeNodeRecord(node_id=0, node_path="0", node_type="root", level=0),),
            entities=(EntityRecord("e1", "ai", "TECH"),),
            relations=(RelationRecord("rl1", "e1", "e1", "self", directed=True),),
            gt_links=(GTLinkRecord("e1", 0, "b0"),),
        )

    def test_persist_replaces_then_inserts_every_table(self):
        connection = _FakeConnection()
        ledger = PostgresLedger(connection_factory=lambda: connection)
        scope = DocumentScope("t", "c", "d", "v")

        ledger.persist_document_evidence(scope, self._evidence())

        queries = [q for q, _ in connection.executions]
        for table in (
            "canonical_blocks",
            "block_representations",
            "block_assets",
            "tree_nodes",
            "kg_entities",
            "kg_relations",
            "gt_links",
        ):
            self.assertTrue(any(f"DELETE FROM docarag.{table}" in q for q in queries))
            self.assertTrue(any(f"INSERT INTO docarag.{table}" in q for q in queries))
        self.assertTrue(any("INSERT INTO docarag.assets" in q for q in queries))
        self.assertTrue(any("::ltree" in q for q in queries))

    def test_subtree_query_uses_ltree_containment(self):
        connection = _FakeConnection()
        ledger = PostgresLedger(connection_factory=lambda: connection)
        scope = DocumentScope("t", "c", "d", "v")

        ledger.node_subtree(scope, "rev1", "0.1")

        query = connection.executions[-1][0]
        self.assertIn("node_path <@ %s::ltree", query)


class EvidenceProjectionTests(unittest.TestCase):
    def _write_run(self, run: Path) -> None:
        (run / "vlm" / "images").mkdir(parents=True)
        (run / "vlm" / "images" / "fig.jpg").write_bytes(b"\xff\xd8jpegbytes\xff\xd9")
        (run / "canonical_blocks.json").write_text(
            json.dumps(
                {
                    "canonical_blocks": [
                        {
                            "block_id": "blk0",
                            "block_type": "title",
                            "page_index": 0,
                            "reading_order": 0,
                            "revision_id": "r0",
                            "representations": [{"kind": "text", "content": "Title"}],
                            "assets": [],
                        },
                        {
                            "block_id": "blk1",
                            "block_type": "table",
                            "page_index": 0,
                            "reading_order": 1,
                            "revision_id": "r0",
                            "representations": [{"kind": "html", "content": "<table/>"}],
                            "assets": [
                                {
                                    "kind": "region_crop",
                                    "media_type": "image/jpeg",
                                    "sha256": None,
                                    "uri": str(run / "vlm" / "images" / "fig.jpg"),
                                }
                            ],
                        },
                    ]
                }
            ),
            encoding="utf-8",
        )
        (run / "tree.json").write_text(
            json.dumps(
                {
                    "nodes": [
                        {"index_id": 0, "parent_id": None, "type": "root", "meta_info": {}, "summary": ""},
                        {"index_id": 1, "parent_id": 0, "type": "title", "meta_info": {}, "summary": "s1"},
                        {"index_id": 2, "parent_id": 1, "type": "text", "meta_info": {}, "summary": "s2"},
                    ]
                }
            ),
            encoding="utf-8",
        )
        a = "Name: a\nType: X"
        b = "Name: b\nType: Y"
        (run / "graph_data.json").write_text(
            json.dumps(
                {
                    "graph": {
                        "nodes": [
                            {"id": a, "entity_name": "a", "entity_type": "X", "description": "da"},
                            {"id": b, "entity_name": "b", "entity_type": "Y", "description": "db"},
                        ],
                        "links": [
                            {"source": a, "target": b, "relation_name": "rel1", "weight": "2.0"},
                            {"source": a, "target": b, "relation_name": "rel2", "weight": "1.0"},
                        ],
                    },
                    "tree2kg": {"1": [a], "2": [a, b]},
                }
            ),
            encoding="utf-8",
        )

    def test_projection_maps_blocks_tree_graph_and_publishes_assets(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "run"
            run.mkdir()
            self._write_run(run)
            cas = Path(directory) / "cas"

            evidence = build_document_evidence(run, "rev1", asset_cas_root=cas)

            self.assertEqual(2, len(evidence.blocks))
            self.assertEqual({"title", "table"}, {b.block_type for b in evidence.blocks})
            table = next(b for b in evidence.blocks if b.block_type == "table")
            self.assertEqual("html", table.representations[0].kind)
            paths = {node.node_id: node.node_path for node in evidence.tree_nodes}
            self.assertEqual({0: "0", 1: "0.1", 2: "0.1.2"}, paths)
            self.assertEqual(2, len(evidence.entities))
            # both parallel relations are preserved as distinct directed edges
            self.assertEqual(2, len(evidence.relations))
            self.assertTrue(all(r.directed for r in evidence.relations))
            self.assertEqual(2, len({r.relation_id for r in evidence.relations}))
            # gt-links: node1->{a}, node2->{a,b} == 3
            self.assertEqual(3, len(evidence.gt_links))
            # the table crop is content-addressed, linked to its block, and in CAS
            self.assertEqual(1, len(evidence.assets))
            digest = evidence.assets[0].asset_id
            self.assertEqual(64, len(digest))
            self.assertTrue((cas / digest[:2] / digest).is_file())
            self.assertEqual(digest, table.asset_links[0].asset_id)
            self.assertEqual("region_crop", table.asset_links[0].role)

    def test_an_already_published_crop_is_registered_not_only_linked(self):
        # Crops reach the canonical layer already published to a
        # content-addressed store, so they arrive carrying a digest. Linking one
        # without registering it violates the store's foreign key and fails the
        # whole document's persistence.
        digest = "d" * 64
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "run"
            run.mkdir()
            self._write_run(run)
            cas = Path(directory) / "cas"
            (cas / digest[:2]).mkdir(parents=True)
            (cas / digest[:2] / digest).write_bytes(b"published crop bytes")

            payload = json.loads((run / "canonical_blocks.json").read_text())
            payload["canonical_blocks"][1]["assets"] = [
                {
                    "kind": "region_crop",
                    "media_type": "image/jpeg",
                    "sha256": digest,
                    "uri": f"cas://sha256/{digest}",
                }
            ]
            (run / "canonical_blocks.json").write_text(json.dumps(payload))

            evidence = build_document_evidence(run, "rev1", asset_cas_root=cas)

            registered = {asset.asset_id for asset in evidence.assets}
            linked = {
                link.asset_id
                for block in evidence.blocks
                for link in block.asset_links
            }
            self.assertTrue(
                linked <= registered,
                f"linked assets missing from the assets table: {linked - registered}",
            )
            record = next(a for a in evidence.assets if a.asset_id == digest)
            self.assertEqual(f"cas://sha256/{digest}", record.cas_uri)
            self.assertEqual("image/jpeg", record.mime_type)
            self.assertEqual(len(b"published crop bytes"), record.byte_size)

    def test_projection_is_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "run"
            run.mkdir()
            self._write_run(run)
            first = build_document_evidence(run, "rev1")
            second = build_document_evidence(run, "rev1")
            self.assertEqual(
                [r.relation_id for r in first.relations],
                [r.relation_id for r in second.relations],
            )


if __name__ == "__main__":
    unittest.main()
