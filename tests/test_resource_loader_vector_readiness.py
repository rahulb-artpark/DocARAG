from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

from Core.Index.Tree import DocumentTree, NodeType, TreeNode
from Core.pipelines import vdb_index
from Core.utils import resource_loader
from Core.utils.vector_index_manifest import vector_index_lock
from Core.utils.artifact_paths import QdrantCollectionState


class FakeGmeEmbedder:
    MM_EMBEDDER = True

    def embed_texts(self, texts):
        return [[float(len(text)), 1.0] for text in texts]

    def close(self):
        pass


def _tree(directory: str) -> DocumentTree:
    tree = DocumentTree(
        meta_dict={
            "file_name": "source.pdf",
            "source_uri": "urn:test:source.pdf",
            "source_sha256": "a" * 64,
            "document_version_id": Path(directory).name,
        },
        cfg=SimpleNamespace(save_path=directory),
    )
    node = TreeNode(
        {
            "pdf_id": 1,
            "canonical_block_id": "block-1",
            "canonical_revision_id": "block-1:r1",
            "content": "source text",
        }
    )
    node.type = NodeType.TEXT
    tree.add_node(node)
    tree.root_node.add_child(node)
    tree.save_to_file()
    return tree


def _config(directory: str, *, model_name: str = "gme-v1"):
    embedding = SimpleNamespace(
        type="gme",
        model_name=model_name,
        device="cpu",
        backend="local",
        api_base=None,
        max_length=256,
    )
    vdb = SimpleNamespace(
        mm_embedding=True,
        tenant_id="tenant-a",
        corpus_id="corpus-a",
        qdrant_url="http://qdrant:6333",
        api_key_env="QDRANT_API_KEY",
        timeout_seconds=1.0,
        manifest_dir_name="vectors",
        collection_name="bookrag-tree",
        embedding_config=embedding,
        index_batch_size=2,
    )
    return SimpleNamespace(
        save_path=directory,
        source=SimpleNamespace(
            sha256="a" * 64,
            original_uri="urn:test:source.pdf",
        ),
        rag=SimpleNamespace(
            strategy_config=SimpleNamespace(
                strategy="mmr",
                vdb_config=vdb,
            )
        ),
    )


class ResourceLoaderVectorReadinessTests(unittest.TestCase):
    def setUp(self):
        self.remote: dict[tuple, int] = {}
        self.state_patch = patch.object(
            vdb_index,
            "_qdrant_state",
            side_effect=self._remote_state,
        )
        self.state_patch.start()
        # The store seal is the binding truth; unit fixtures share one seal so
        # publication and readiness agree without a live PostgreSQL.
        self.seal_patch = patch.object(
            vdb_index, "_store_revision_seal", return_value="a" * 64
        )
        self.seal_patch.start()

    def tearDown(self):
        self.state_patch.stop()
        self.seal_patch.stop()

    def _remote_state(self, config, scope):
        key = (config.collection_name, tuple(sorted(scope.items())))
        if key not in self.remote:
            return QdrantCollectionState(False, None)
        return QdrantCollectionState(True, self.remote[key])

    def _publish_index(self, tree, config):
        runtime_vdb = SimpleNamespace(**vars(config.rag.strategy_config.vdb_config))
        runtime_vdb.manifest_dir_name = str(Path(config.save_path) / "vectors")
        remote = self.remote

        class FakeStore:
            def __init__(self, **kwargs):
                self.key = (
                    kwargs["collection_name"],
                    tuple(sorted(kwargs["scope_filter"].items())),
                )

            def reset(self):
                remote[self.key] = 0

            def add_texts(self, *, texts, **_kwargs):
                remote[self.key] += len(texts)

            def add_images(self, *, image_paths, **_kwargs):
                remote[self.key] += len(image_paths)

            def count(self):
                return remote[self.key]

            def close(self):
                pass

        with (
            patch.object(
                vdb_index,
                "GmeEmbeddingProvider",
                side_effect=lambda **_kwargs: FakeGmeEmbedder(),
            ),
            patch.object(vdb_index, "VectorStore", FakeStore),
        ):
            vdb_index.build_vdb_index(tree, runtime_vdb)

    def _assert_exclusive_lock_available(self, path: str) -> None:
        acquired = threading.Event()

        def acquire():
            with vector_index_lock(path, exclusive=True):
                acquired.set()

        worker = threading.Thread(target=acquire, daemon=True)
        worker.start()
        self.assertTrue(acquired.wait(1), "exclusive vector lock remained leased")
        worker.join(timeout=1)

    def test_mmr_loader_rejects_missing_manifest_before_model_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            _tree(directory)
            config = _config(directory)

            with (
                patch(
                    "Core.provider.embedding.GmeEmbeddingProvider",
                    side_effect=AssertionError("model loaded before readiness"),
                ) as model,
                self.assertRaisesRegex(
                    vdb_index.VectorIndexBindingError,
                    "sealed store revision",
                ),
            ):
                resource_loader.prepare_rag_dependencies(config)

            model.assert_not_called()
            self._assert_exclusive_lock_available(str(Path(directory) / "vectors"))

    def test_mmr_loader_accepts_exact_manifest_and_then_loads_resources(self):
        with tempfile.TemporaryDirectory() as directory:
            tree = _tree(directory)
            config = _config(directory)
            self._publish_index(tree, config)
            model = Mock()

            class LeaseAwareStore:
                def __init__(self):
                    self.lease = None

                def attach_read_lease(self, lease):
                    self.lease = lease

                def close(self):
                    if self.lease is not None:
                        self.lease.close()
                        self.lease = None

            store = LeaseAwareStore()

            with (
                patch(
                    "Core.provider.embedding.GmeEmbeddingProvider",
                    return_value=model,
                ) as model_factory,
                patch("Core.provider.vdb.VectorStore", return_value=store) as store_factory,
                patch.object(
                    vdb_index,
                    "expected_tree_vector_index_attestation",
                    side_effect=AssertionError(
                        "query startup recomputed the full asset projection"
                    ),
                ),
            ):
                dependencies = resource_loader.prepare_rag_dependencies(config)

            self.assertIs(store, dependencies["vector_store"])
            model_factory.assert_called_once_with(model_name="gme-v1", device="cpu")
            store_factory.assert_called_once_with(
                embedding_model=model,
                qdrant_url="http://qdrant:6333",
                collection_name="bookrag-tree",
                api_key_env="QDRANT_API_KEY",
                timeout_seconds=1.0,
                scope_filter={
                    "tenant_id": "tenant-a",
                    "corpus_id": "corpus-a",
                    "document_version_id": Path(directory).name,
                    "index_revision_id": Path(directory).name,
                    "source_sha256": "a" * 64,
                    "index_kind": "tree",
                },
            )
            self.assertIsNotNone(store.lease)

            acquired = threading.Event()

            def acquire_exclusive():
                with vector_index_lock(
                    str(Path(directory) / "vectors"),
                    exclusive=True,
                ):
                    acquired.set()

            writer = threading.Thread(target=acquire_exclusive, daemon=True)
            writer.start()
            self.assertFalse(
                acquired.wait(0.1),
                "rebuild lock acquired while the query store was open",
            )
            store.close()
            self.assertTrue(acquired.wait(1))
            writer.join(timeout=1)

    def test_mmr_loader_rejects_source_or_embedding_mismatch_before_model(self):
        with tempfile.TemporaryDirectory() as directory:
            tree = _tree(directory)
            config = _config(directory)
            self._publish_index(tree, config)

            config.source.sha256 = "b" * 64
            with (
                patch("Core.provider.embedding.GmeEmbeddingProvider") as model,
                self.assertRaisesRegex(
                    vdb_index.VectorIndexBindingError,
                    "sealed store revision",
                ),
            ):
                resource_loader.prepare_rag_dependencies(config)
            model.assert_not_called()
            self._assert_exclusive_lock_available(str(Path(directory) / "vectors"))

            changed_collection = _config(directory)
            changed_collection.rag.strategy_config.vdb_config.collection_name = (
                "another-collection"
            )
            with (
                patch("Core.provider.embedding.GmeEmbeddingProvider") as model,
                self.assertRaisesRegex(
                    vdb_index.VectorIndexBindingError,
                    "sealed store revision",
                ),
            ):
                resource_loader.prepare_rag_dependencies(changed_collection)
            model.assert_not_called()
            self._assert_exclusive_lock_available(str(Path(directory) / "vectors"))

            changed_model = _config(directory, model_name="gme-v2")
            with (
                patch("Core.provider.embedding.GmeEmbeddingProvider") as model,
                self.assertRaisesRegex(
                    vdb_index.VectorIndexBindingError,
                    "sealed store revision",
                ),
            ):
                resource_loader.prepare_rag_dependencies(changed_model)
            model.assert_not_called()
            self._assert_exclusive_lock_available(str(Path(directory) / "vectors"))

    def test_mmr_query_provider_uses_same_effective_mode_as_builder(self):
        with tempfile.TemporaryDirectory() as directory:
            tree = _tree(directory)
            config = _config(directory)
            # ``type`` is a historical redundant field. The tree-vector builder
            # uses mm_embedding=True and therefore writes GME vectors.
            config.rag.strategy_config.vdb_config.embedding_config.type = "text"
            self._publish_index(tree, config)
            model = Mock()

            class LeaseAwareStore:
                def __init__(self):
                    self.lease = None

                def attach_read_lease(self, lease):
                    self.lease = lease

                def close(self):
                    if self.lease is not None:
                        self.lease.close()
                        self.lease = None

            store = LeaseAwareStore()
            with (
                patch(
                    "Core.provider.embedding.GmeEmbeddingProvider",
                    return_value=model,
                ) as gme_factory,
                patch(
                    "Core.provider.embedding.TextEmbeddingProvider",
                    side_effect=AssertionError("query selected incompatible provider"),
                ) as text_factory,
                patch("Core.provider.vdb.VectorStore", return_value=store),
            ):
                dependencies = resource_loader.prepare_rag_dependencies(config)

            self.assertIs(store, dependencies["vector_store"])
            gme_factory.assert_called_once_with(model_name="gme-v1", device="cpu")
            text_factory.assert_not_called()
            store.close()

    def test_model_start_failure_releases_query_read_lease(self):
        with tempfile.TemporaryDirectory() as directory:
            tree = _tree(directory)
            config = _config(directory)
            self._publish_index(tree, config)

            with (
                patch(
                    "Core.provider.embedding.GmeEmbeddingProvider",
                    side_effect=RuntimeError("model unavailable"),
                ),
                self.assertRaisesRegex(RuntimeError, "model unavailable"),
            ):
                resource_loader.prepare_rag_dependencies(config)

            self._assert_exclusive_lock_available(str(Path(directory) / "vectors"))

    def test_changed_store_rows_invalidate_query_attestation(self):
        with tempfile.TemporaryDirectory() as directory:
            tree = _tree(directory)
            config = _config(directory)
            self._publish_index(tree, config)
            # The store's rows changed after the vectors were built: the live
            # seal no longer matches the one the attestation recorded.
            self.seal_patch.stop()
            self.seal_patch = patch.object(
                vdb_index, "_store_revision_seal", return_value="b" * 64
            )
            self.seal_patch.start()

            with (
                patch("Core.provider.embedding.GmeEmbeddingProvider") as model,
                self.assertRaisesRegex(
                    vdb_index.VectorIndexBindingError,
                    "sealed store revision",
                ),
            ):
                resource_loader.prepare_rag_dependencies(config)

            model.assert_not_called()
            self._assert_exclusive_lock_available(str(Path(directory) / "vectors"))

    def test_mmr_store_attach_failure_closes_store_model_and_lease(self):
        with tempfile.TemporaryDirectory() as directory:
            tree = _tree(directory)
            config = _config(directory)
            self._publish_index(tree, config)
            model = Mock()
            store = Mock()
            store.attach_read_lease.side_effect = RuntimeError("attach failed")

            with (
                patch(
                    "Core.provider.embedding.GmeEmbeddingProvider",
                    return_value=model,
                ),
                patch("Core.provider.vdb.VectorStore", return_value=store),
                self.assertRaisesRegex(RuntimeError, "attach failed"),
            ):
                resource_loader.prepare_rag_dependencies(config)

            store.close.assert_called_once_with()
            model.close.assert_called_once_with()
            self._assert_exclusive_lock_available(str(Path(directory) / "vectors"))

    def test_other_vector_dependency_factories_close_partial_resources(self):
        with tempfile.TemporaryDirectory() as directory:
            embedding = SimpleNamespace(
                type="text",
                model_name="text-v1",
                device="cpu",
                backend="local",
                api_base=None,
                max_length=256,
            )

            def vdb(path, collection):
                return SimpleNamespace(
                    mm_embedding=False,
                    tenant_id="tenant-a",
                    corpus_id="corpus-a",
                    qdrant_url="http://qdrant:6333",
                    api_key_env="QDRANT_API_KEY",
                    timeout_seconds=1.0,
                    manifest_dir_name=path,
                    collection_name=collection,
                    embedding_config=embedding,
                )

            model = Mock()
            tree_store = Mock()
            gbc_config = SimpleNamespace(
                save_path=directory,
                source=SimpleNamespace(sha256="a" * 64),
                index_type="gbc",
                rag=SimpleNamespace(
                    strategy_config=SimpleNamespace(
                        strategy="gbcvanilla",
                        tree_vdb_config=vdb("tree-vectors", "tree"),
                        graph_vdb_config=vdb("graph-vectors", "graph"),
                    )
                ),
            )
            with (
                patch.object(
                    resource_loader,
                    "_acquire_ready_scoped_qdrant",
                    side_effect=[Mock(), Mock()],
                ),
                patch(
                    "Core.provider.embedding.TextEmbeddingProvider",
                    return_value=model,
                ),
                patch(
                    "Core.provider.vdb.VectorStore",
                    side_effect=[tree_store, RuntimeError("graph open failed")],
                ),
                self.assertRaisesRegex(RuntimeError, "graph open failed"),
            ):
                resource_loader.prepare_rag_dependencies(gbc_config)
            tree_store.close.assert_called_once_with()
            model.close.assert_called_once_with()

            vanilla_model = Mock()
            vanilla_config = SimpleNamespace(
                save_path=directory,
                source=SimpleNamespace(sha256="a" * 64),
                index_type="vanilla",
                rag=SimpleNamespace(
                    strategy_config=SimpleNamespace(
                        strategy="vanilla",
                        retrieval_method="vanilla",
                        vdb_config=vdb("vanilla-vectors", "vanilla"),
                    )
                ),
            )
            with (
                patch(
                    "Core.provider.embedding.TextEmbeddingProvider",
                    return_value=vanilla_model,
                ),
                patch.object(
                    resource_loader,
                    "_acquire_ready_scoped_qdrant",
                    return_value=Mock(),
                ),
                patch(
                    "Core.provider.vdb.VectorStore",
                    side_effect=RuntimeError("vanilla open failed"),
                ),
                self.assertRaisesRegex(RuntimeError, "vanilla open failed"),
            ):
                resource_loader.prepare_rag_dependencies(vanilla_config)
            vanilla_model.close.assert_called_once_with()

    def test_gbc_loader_opens_attested_tree_vdb_as_active_dependency(self):
        gbc = Mock()
        tree_vdb = Mock()
        cfg = SimpleNamespace(
            save_path="/run/document-version",
            vdb=SimpleNamespace(),
            rag=SimpleNamespace(
                strategy_config=SimpleNamespace(strategy="gbc")
            ),
        )
        with (
            patch(
                "Core.rag.stored_index.StoredIndex.open",
                return_value=gbc,
            ) as load_gbc,
            patch.object(
                resource_loader,
                "_open_ready_bookrag_tree_vdb",
                return_value=tree_vdb,
            ) as open_tree,
        ):
            dependencies = resource_loader.prepare_rag_dependencies(cfg)

        self.assertEqual(
            {"gbc_index": gbc, "tree_vdb": tree_vdb}, dependencies
        )
        load_gbc.assert_called_once_with(cfg.save_path, config=cfg)
        open_tree.assert_called_once_with(cfg, cfg.vdb)

    def test_gbc_loader_closes_graph_resources_when_tree_vdb_open_fails(self):
        gbc = Mock()
        cfg = SimpleNamespace(
            save_path="/run/document-version",
            vdb=SimpleNamespace(),
            rag=SimpleNamespace(
                strategy_config=SimpleNamespace(strategy="gbc")
            ),
        )
        with (
            patch(
                "Core.rag.stored_index.StoredIndex.open",
                return_value=gbc,
            ),
            patch.object(
                resource_loader,
                "_open_ready_bookrag_tree_vdb",
                side_effect=RuntimeError("tree vectors unavailable"),
            ),
            self.assertRaisesRegex(RuntimeError, "tree vectors unavailable"),
        ):
            resource_loader.prepare_rag_dependencies(cfg)

        gbc.close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
