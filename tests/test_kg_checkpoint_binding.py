from __future__ import annotations

import json
import os
from pathlib import Path
import stat
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from retrieval_engine.indexes.graph import Entity
from retrieval_engine.indexes.tree import NodeType, TreeNode
from retrieval_engine.indexing.graph.extractor import (
    KG_CHECKPOINT_SCHEMA_VERSION,
    KGExtractor,
)


class _Model:
    def __init__(self, model_name: str, api_key: str):
        self.config = SimpleNamespace(
            backend="test",
            model_name=model_name,
            api_base="http://model.test/v1",
            api_key=api_key,
            temperature=0.1,
            max_tokens=512,
        )


class _CountingExtractor:
    def __init__(self, llm: _Model, vlm: _Model):
        self.llm = llm
        self.vlm = vlm
        self.calls = 0
        self.title_calls = 0

    @staticmethod
    def _result(node: TreeNode, description: str):
        return {
            "entities": [
                Entity(
                    entity_name="Checkpoint Entity",
                    entity_type="TEST",
                    description=description,
                    source_ids={node.index_id},
                )
            ],
            "relations": [],
            "node_idx": node.index_id,
        }

    def extract(self, node: TreeNode):
        self.calls += 1
        return self._result(node, f"node extraction {self.calls}")

    def extract_title(self, node, title_path, sibling_nodes):
        self.title_calls += 1
        return self._result(node, f"title extraction {self.title_calls}")


def _node(
    index: int = 7,
    node_type: NodeType = NodeType.TEXT,
    **overrides,
) -> TreeNode:
    metadata = {
        "source_sha256": "a" * 64,
        "document_version_id": "revision-1",
        "canonical_block_id": "block-1",
        "canonical_revision_id": "block-revision-1",
        "observation_id": "observation-1",
        "source_observation_ids": ["observation-1", "observation-2"],
        "page_idx": 3,
        "page_id": "page-3",
        "page_path": "/immutable/pages/page-3.png",
        "source_geometry": {"bbox": [10, 20, 30, 40], "space": "pdf"},
        "content": "The source evidence",
        "table_body": None,
        "caption": "Evidence caption",
        "footnote": "Evidence footnote",
        "img_path": None,
    }
    metadata.update(overrides)
    node = TreeNode(metadata)
    node.index_id = index
    node.type = node_type
    return node


def _kg_extractor(directory: str, *, force_rebuild: bool = False):
    llm = _Model("llm-v1", "super-secret-llm-key")
    vlm = _Model("vlm-v1", "super-secret-vlm-key")
    worker = _CountingExtractor(llm, vlm)
    extractor = object.__new__(KGExtractor)
    extractor.cfg_graph = SimpleNamespace(
        extractor_type="llm",
        local_model_name="unused",
        local_device="cpu",
        image_description_force=True,
        max_gleaning=1,
        extraction_max_workers=2,
        extraction_max_in_flight=2,
        extraction_batch_size=8,
    )
    extractor.extractor = worker
    extractor.llm = llm
    extractor.vlm = vlm
    extractor.save_path = os.path.join(directory, "kg_extractor_res")
    os.makedirs(extractor.save_path, exist_ok=True)
    extractor.force_rebuild = force_rebuild
    return extractor, worker


class KGCheckpointBindingTests(unittest.TestCase):
    def test_valid_checkpoint_resumes_across_extractor_instances(self):
        with tempfile.TemporaryDirectory() as directory:
            node = _node()
            first, first_worker = _kg_extractor(directory)
            first_result = first.extract_kg(node)
            self.assertEqual(first_worker.calls, 1)

            resumed, resumed_worker = _kg_extractor(directory)
            resumed_result = resumed.extract_kg(node)

            self.assertEqual(resumed_worker.calls, 0)
            self.assertEqual(
                resumed_result["entities"][0].description,
                first_result["entities"][0].description,
            )
            self.assertEqual(resumed_result["node_idx"], node.index_id)

            checkpoint = json.loads(
                Path(resumed._checkpoint_path(node.index_id)).read_text("utf-8")
            )
            self.assertEqual(
                checkpoint["schema_version"], KG_CHECKPOINT_SCHEMA_VERSION
            )
            self.assertRegex(checkpoint["payload_sha256"], r"^[0-9a-f]{64}$")
            serialized = json.dumps(checkpoint)
            self.assertNotIn("super-secret-llm-key", serialized)
            self.assertNotIn("super-secret-vlm-key", serialized)

    def test_changed_node_evidence_with_same_node_id_is_recomputed(self):
        mutations = {
            "node_type": lambda node: setattr(node, "type", NodeType.EQUATION),
            "content": lambda node: setattr(node.meta_info, "content", "changed"),
            "table_body": lambda node: setattr(
                node.meta_info, "table_body", "<table><tr><td>changed</td></tr></table>"
            ),
            "caption": lambda node: setattr(node.meta_info, "caption", "changed"),
            "source_sha256": lambda node: setattr(
                node.meta_info, "source_sha256", "b" * 64
            ),
            "document_version": lambda node: setattr(
                node.meta_info, "document_version_id", "revision-2"
            ),
            "canonical_block": lambda node: setattr(
                node.meta_info, "canonical_block_id", "block-2"
            ),
            "canonical_revision": lambda node: setattr(
                node.meta_info, "canonical_revision_id", "block-revision-2"
            ),
            "observation": lambda node: setattr(
                node.meta_info, "observation_id", "observation-3"
            ),
            "source_observations": lambda node: setattr(
                node.meta_info,
                "source_observation_ids",
                ["observation-1", "observation-3"],
            ),
            "derived_source_blocks": lambda node: setattr(
                node.meta_info,
                "source_canonical_block_ids",
                ["block-1", "block-2"],
            ),
            "page_idx": lambda node: setattr(node.meta_info, "page_idx", 4),
            "page_id": lambda node: setattr(node.meta_info, "page_id", "page-4"),
            "geometry": lambda node: setattr(
                node.meta_info,
                "source_geometry",
                {"bbox": [11, 20, 30, 40], "space": "pdf"},
            ),
        }

        for name, mutate in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                extractor, worker = _kg_extractor(directory)
                node = _node()
                extractor.extract_kg(node)
                mutate(node)
                result = extractor.extract_kg(node)

                self.assertEqual(worker.calls, 2)
                self.assertEqual(result["node_idx"], node.index_id)
                self.assertEqual(
                    result["entities"][0].description, "node extraction 2"
                )

    def test_model_and_prompt_changes_invalidate_but_secrets_do_not(self):
        with tempfile.TemporaryDirectory() as directory:
            extractor, worker = _kg_extractor(directory)
            node = _node()
            extractor.extract_kg(node)

            extractor.llm.config.api_key = "rotated-secret"
            extractor.extract_kg(node)
            self.assertEqual(worker.calls, 1)

            extractor.llm.config.model_name = "llm-v2"
            extractor.extract_kg(node)
            self.assertEqual(worker.calls, 2)

            with patch(
                "retrieval_engine.indexing.graph.extractor.ENTITY_EXTRACTION",
                "changed entity extraction prompt {input_text}",
            ):
                extractor.extract_kg(node)
            self.assertEqual(worker.calls, 3)

            extractor.cfg_graph.max_gleaning = 2
            extractor.extract_kg(node)
            self.assertEqual(worker.calls, 4)

    def test_endpoint_identity_drops_url_credentials_but_keeps_routing(self):
        with tempfile.TemporaryDirectory() as directory:
            extractor, worker = _kg_extractor(directory)
            node = _node()
            extractor.llm.config.api_base = (
                "https://url-user:url-password@model.test:8443/v1/chat"
                "?api_key=query-secret#secret-fragment"
            )
            extractor.extract_kg(node)

            checkpoint_path = Path(extractor._checkpoint_path(node.index_id))
            checkpoint = json.loads(checkpoint_path.read_text("utf-8"))
            endpoint = checkpoint["identity"]["models"]["llm"]["config"][
                "api_base"
            ]
            self.assertEqual(endpoint, "https://model.test:8443/v1/chat")
            serialized = json.dumps(checkpoint)
            for credential in (
                "url-user",
                "url-password",
                "query-secret",
                "secret-fragment",
            ):
                self.assertNotIn(credential, serialized)

            # Credential rotation does not change model behavior or invalidate work.
            extractor.llm.config.api_base = (
                "https://other-user:other-password@model.test:8443/v1/chat"
                "?api_key=other-query-secret"
            )
            extractor.extract_kg(node)
            self.assertEqual(worker.calls, 1)

            # Routing changes remain part of the operational model identity.
            extractor.llm.config.api_base = "https://model.test:8443/v2/chat"
            extractor.extract_kg(node)
            self.assertEqual(worker.calls, 2)

    def test_image_checkpoint_is_bound_to_image_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "figure.png"
            image_path.write_bytes(b"first-image-bytes")
            node = _node(node_type=NodeType.IMAGE, img_path=str(image_path))
            extractor, worker = _kg_extractor(directory)

            extractor.extract_kg(node)
            extractor.extract_kg(node)
            self.assertEqual(worker.calls, 1)

            image_path.write_bytes(b"second-image-bytes")
            result = extractor.extract_kg(node)
            self.assertEqual(worker.calls, 2)
            self.assertEqual(result["entities"][0].description, "node extraction 2")

            extractor.vlm.config.model_name = "vlm-v2"
            extractor.extract_kg(node)
            self.assertEqual(worker.calls, 3)

    def test_title_checkpoint_is_bound_to_path_and_sibling_context(self):
        with tempfile.TemporaryDirectory() as directory:
            extractor, worker = _kg_extractor(directory)
            title = _node(index=7, node_type=NodeType.TITLE, content="Current title")
            title_path = [
                _node(index=1, node_type=NodeType.TITLE, content="Parent title")
            ]
            siblings = [
                _node(index=6, node_type=NodeType.TITLE, content="Previous title"),
                _node(index=8, node_type=NodeType.TITLE, content="Next title"),
            ]

            extractor.extract_title(title, title_path, siblings)
            extractor.extract_title(title, title_path, siblings)
            self.assertEqual(worker.title_calls, 1)

            siblings[0].meta_info.content = "Changed previous title"
            extractor.extract_title(title, title_path, siblings)
            self.assertEqual(worker.title_calls, 2)

            title_path[0].meta_info.content = "Changed parent title"
            result = extractor.extract_title(title, title_path, siblings)
            self.assertEqual(worker.title_calls, 3)
            self.assertEqual(result["entities"][0].description, "title extraction 3")

    def test_corrupt_legacy_and_wrong_node_payloads_are_recomputed(self):
        corruptions = {}

        def truncated(_envelope):
            return "{\"schema_version\":"

        def legacy(envelope):
            return json.dumps(envelope["payload"])

        def wrong_node(envelope):
            envelope["payload"]["node_idx"] = 999
            return json.dumps(envelope)

        def tampered_payload(envelope):
            envelope["payload"]["entities"][0]["description"] = "tampered"
            return json.dumps(envelope)

        def invalid_utf8(_envelope):
            return b"\xff\xfe\x00"

        corruptions["truncated"] = truncated
        corruptions["legacy"] = legacy
        corruptions["wrong-node"] = wrong_node
        corruptions["tampered-payload"] = tampered_payload
        corruptions["invalid-utf8"] = invalid_utf8

        for name, corrupt in corruptions.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                extractor, worker = _kg_extractor(directory)
                node = _node()
                extractor.extract_kg(node)
                path = Path(extractor._checkpoint_path(node.index_id))
                envelope = json.loads(path.read_text("utf-8"))
                corrupted = corrupt(envelope)
                if isinstance(corrupted, bytes):
                    path.write_bytes(corrupted)
                else:
                    path.write_text(corrupted, encoding="utf-8")

                result = extractor.extract_kg(node)

                self.assertEqual(worker.calls, 2)
                self.assertEqual(result["node_idx"], node.index_id)
                repaired = json.loads(path.read_text("utf-8"))
                self.assertEqual(
                    repaired["schema_version"], KG_CHECKPOINT_SCHEMA_VERSION
                )
                self.assertEqual(repaired["payload"]["node_idx"], node.index_id)

    def test_checkpoint_write_is_owner_only_atomic_and_preserves_previous_file(self):
        with tempfile.TemporaryDirectory() as directory:
            extractor, _worker = _kg_extractor(directory)
            node = _node()
            result = extractor.extract_kg(node)
            path = Path(extractor._checkpoint_path(node.index_id))
            original = path.read_bytes()

            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            self.assertEqual(list(path.parent.glob("*.tmp")), [])
            self.assertEqual(list(path.parent.glob(".*.tmp")), [])

            identity = extractor._checkpoint_identity(node, kind="node")
            with patch(
                "retrieval_engine.indexing.graph.extractor.json.dump",
                side_effect=RuntimeError("interrupted serialization"),
            ):
                with self.assertRaisesRegex(RuntimeError, "interrupted"):
                    extractor.save_tmp_res(result, node.index_id, identity)

            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(path.parent.glob("*.tmp")), [])
            self.assertEqual(list(path.parent.glob(".*.tmp")), [])

    def test_force_rebuild_ignores_a_valid_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            node = _node()
            first, first_worker = _kg_extractor(directory)
            first.extract_kg(node)
            self.assertEqual(first_worker.calls, 1)

            forced, forced_worker = _kg_extractor(directory, force_rebuild=True)
            result = forced.extract_kg(node)

            self.assertEqual(forced_worker.calls, 1)
            self.assertEqual(result["entities"][0].description, "node extraction 1")


if __name__ == "__main__":
    unittest.main()
