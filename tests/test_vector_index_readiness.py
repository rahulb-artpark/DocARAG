from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from Core.Index.Tree import DocumentTree, NodeType, TreeNode
from Core.pipelines import vdb_index
from Core.utils.artifact_paths import QdrantCollectionState, QdrantUnavailableError
from Core.utils.vector_index_manifest import (
    VECTOR_INDEX_ATTESTATION_NAME,
    read_vector_index_attestation,
)


class FakeTextEmbedder:
    MM_EMBEDDER = False

    def __init__(self, events: list[str] | None = None):
        self.events = events if events is not None else []

    def embed_texts(self, texts):
        self.events.append("embed")
        return [[float(len(text)), 1.0] for text in texts]

    def close(self):
        self.events.append("embedder-close")


class FakeScopedQdrant:
    def __init__(self, remote: dict[tuple, dict[str, object]], events=None, **kwargs):
        self.remote = remote
        self.events = events if events is not None else []
        self.collection_name = kwargs["collection_name"]
        self.scope = tuple(sorted(kwargs["scope_filter"].items()))
        self.embedding_model = kwargs["embedding_model"]
        self.closed = False
        self.events.append("store-open")

    @property
    def key(self):
        return self.collection_name, self.scope

    def reset(self):
        self.events.append("reset")
        self.remote[self.key] = {"count": 0}

    def add_texts(self, *, texts, **_kwargs):
        self.events.append("write")
        self.embedding_model.embed_texts(texts)
        self.remote.setdefault(self.key, {"count": 0})["count"] += len(texts)

    def add_images(self, *, image_paths, **_kwargs):
        self.remote.setdefault(self.key, {"count": 0})["count"] += len(image_paths)

    def count(self):
        self.events.append("count")
        return int(self.remote.get(self.key, {}).get("count", 0))

    def close(self):
        self.events.append("store-close")
        self.closed = True


def _config(directory: str, *, model_name: str = "embedding-v1", url: str = "http://qdrant:6333"):
    return SimpleNamespace(
        mm_embedding=False,
        tenant_id="tenant-a",
        corpus_id="corpus-a",
        qdrant_url=url,
        api_key_env="QDRANT_API_KEY",
        timeout_seconds=1.0,
        embedding_config=SimpleNamespace(
            model_name=model_name,
            backend="local",
            api_base=None,
            max_length=256,
            device="cpu",
        ),
        manifest_dir_name=str(Path(directory) / "vector-attestation"),
        collection_name="bookrag-tree",
        index_batch_size=2,
    )


def _tree(directory: str, texts: tuple[str, ...] = ("alpha",)) -> DocumentTree:
    tree = DocumentTree(
        meta_dict={
            "file_name": "source.pdf",
            "source_uri": "urn:test:source.pdf",
            "source_sha256": "a" * 64,
            "document_version_id": Path(directory).name,
        },
        cfg=SimpleNamespace(save_path=directory),
    )
    for index, text in enumerate(texts, start=1):
        node = TreeNode(
            {
                "pdf_id": index,
                "page_idx": index - 1,
                "canonical_block_id": f"block-{index}",
                "canonical_revision_id": f"block-{index}:r1",
                "content": text,
            }
        )
        node.type = NodeType.TEXT
        tree.add_node(node)
        tree.root_node.add_child(node)
    return tree


class VectorIndexReadinessTests(unittest.TestCase):
    def setUp(self):
        self.remote: dict[tuple, dict[str, object]] = {}

    def _state(self, config, scope):
        key = (config.collection_name, tuple(sorted(scope.items())))
        if key not in self.remote:
            return QdrantCollectionState(False, None)
        return QdrantCollectionState(True, int(self.remote[key]["count"]))

    def _build(self, tree, config, *, events=None, store_type=FakeScopedQdrant, **kwargs):
        events = events if events is not None else []

        def store_factory(**store_kwargs):
            return store_type(self.remote, events=events, **store_kwargs)

        with (
            patch.object(
                vdb_index,
                "TextEmbeddingProvider",
                side_effect=lambda **_kwargs: FakeTextEmbedder(events),
            ),
            patch.object(vdb_index, "VectorStore", side_effect=store_factory),
            patch.object(
                vdb_index,
                "_qdrant_state",
                side_effect=lambda cfg, scope: self._state(cfg, scope),
            ),
        ):
            return vdb_index.build_vdb_index(tree, config, **kwargs)

    def test_exact_attestation_reuses_without_loading_embedding_model(self):
        with tempfile.TemporaryDirectory() as directory:
            tree = _tree(directory)
            tree.save_to_file()
            config = _config(directory)
            self.assertTrue(self._build(tree, config))
            path = Path(config.manifest_dir_name) / VECTOR_INDEX_ATTESTATION_NAME
            observed = read_vector_index_attestation(config.manifest_dir_name)
            self.assertEqual(
                vdb_index.expected_tree_vector_index_attestation(tree, config),
                observed,
            )
            self.assertEqual(0, os.stat(path).st_mode & 0o077)

            with (
                patch.object(
                    vdb_index,
                    "TextEmbeddingProvider",
                    side_effect=AssertionError("reuse loaded embedding model"),
                ),
                patch.object(
                    vdb_index,
                    "_qdrant_state",
                    side_effect=lambda cfg, scope: self._state(cfg, scope),
                ),
            ):
                self.assertFalse(
                    vdb_index.build_vdb_index(tree, config, reuse_if_ready=True)
                )

    def test_tree_embedding_and_endpoint_changes_invalidate_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            original = _tree(directory, ("alpha", "obsolete"))
            config = _config(directory)
            self._build(original, config)
            first = read_vector_index_attestation(config.manifest_dir_name)
            self.assertEqual(2, first["point_count"])

            current = _tree(directory, ("changed",))
            self.assertTrue(self._build(current, config, reuse_if_ready=True))
            second = read_vector_index_attestation(config.manifest_dir_name)
            self.assertEqual(1, second["point_count"])
            self.assertNotEqual(first["projection_sha256"], second["projection_sha256"])

            changed = _config(directory, model_name="embedding-v2", url="http://other:6333")
            self.assertTrue(self._build(current, changed, reuse_if_ready=True))
            third = read_vector_index_attestation(config.manifest_dir_name)
            self.assertNotEqual(
                second["qdrant_endpoint_sha256"], third["qdrant_endpoint_sha256"]
            )

    def test_write_failure_removes_attestation_and_closes_resources(self):
        class FailingStore(FakeScopedQdrant):
            def add_texts(self, **_kwargs):
                self.events.append("write")
                raise RuntimeError("interrupted write")

        with tempfile.TemporaryDirectory() as directory:
            tree = _tree(directory)
            config = _config(directory)
            events: list[str] = []
            with self.assertRaisesRegex(RuntimeError, "interrupted write"):
                self._build(tree, config, events=events, store_type=FailingStore)
            self.assertIsNone(read_vector_index_attestation(config.manifest_dir_name))
            self.assertEqual(
                ["store-open", "reset", "write", "store-close", "embedder-close"],
                events,
            )

    def test_attestation_is_published_after_store_and_embedder_close(self):
        with tempfile.TemporaryDirectory() as directory:
            tree = _tree(directory)
            config = _config(directory)
            events: list[str] = []
            real_publish = vdb_index.publish_vector_index_attestation

            def publish(path, attestation):
                self.assertEqual(["store-close", "embedder-close"], events[-2:])
                events.append("publish")
                real_publish(path, attestation)

            with patch.object(
                vdb_index, "publish_vector_index_attestation", side_effect=publish
            ):
                self._build(tree, config, events=events)
            self.assertEqual("publish", events[-1])

    def test_live_count_mismatch_is_not_ready(self):
        with tempfile.TemporaryDirectory() as directory:
            tree = _tree(directory)
            config = _config(directory)
            self._build(tree, config)
            key = (config.collection_name, tuple(sorted(vdb_index.tree_vector_scope(tree, config).items())))
            self.remote[key]["count"] = 0
            with patch.object(
                vdb_index,
                "_qdrant_state",
                side_effect=lambda cfg, scope: self._state(cfg, scope),
            ):
                self.assertFalse(vdb_index.tree_vector_index_is_ready(tree, config))

    def test_unreachable_qdrant_is_distinct_and_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            tree = _tree(directory)
            config = _config(directory)
            self._build(tree, config)
            with (
                patch.object(
                    vdb_index,
                    "_qdrant_state",
                    side_effect=QdrantUnavailableError("offline"),
                ),
                self.assertRaisesRegex(QdrantUnavailableError, "offline"),
            ):
                vdb_index.tree_vector_index_is_ready(tree, config)

    def test_attestation_directory_symlink_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            tree = _tree(directory)
            config = _config(directory)
            target = Path(directory) / "outside"
            target.mkdir()
            Path(config.manifest_dir_name).symlink_to(target, target_is_directory=True)
            with self.assertRaisesRegex(
                vdb_index.VectorIndexBindingError, "symbolic link"
            ):
                self._build(tree, config)

    def test_attestation_directory_rejects_legacy_local_vector_files(self):
        with tempfile.TemporaryDirectory() as directory:
            tree = _tree(directory)
            config = _config(directory)
            local = Path(config.manifest_dir_name)
            local.mkdir(parents=True)
            (local / "chroma.sqlite3").write_bytes(b"legacy-local-index")

            with self.assertRaisesRegex(
                vdb_index.VectorIndexBindingError, "unexpected local/legacy"
            ):
                self._build(tree, config)
            self.assertIsNone(read_vector_index_attestation(local))

    def test_extra_file_invalidates_an_existing_attestation(self):
        with tempfile.TemporaryDirectory() as directory:
            tree = _tree(directory)
            config = _config(directory)
            self._build(tree, config)
            local = Path(config.manifest_dir_name)
            (local / "unexpected.bin").write_bytes(b"not-an-attestation")

            with patch.object(
                vdb_index,
                "_qdrant_state",
                side_effect=lambda cfg, scope: self._state(cfg, scope),
            ):
                self.assertFalse(vdb_index.tree_vector_index_is_ready(tree, config))

    def test_tenant_and_corpus_are_part_of_exact_scope(self):
        with tempfile.TemporaryDirectory() as directory:
            tree = _tree(directory)
            config = _config(directory)
            scope = vdb_index.tree_vector_scope(tree, config)
            self.assertEqual("tenant-a", scope["tenant_id"])
            self.assertEqual("corpus-a", scope["corpus_id"])
            config.corpus_id = ""
            with self.assertRaisesRegex(vdb_index.VectorIndexBindingError, "corpus_id"):
                vdb_index.tree_vector_scope(tree, config)


if __name__ == "__main__":
    unittest.main()
