from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import ANY, Mock, patch

from retrieval_engine.providers.language_model import LLM
from retrieval_engine.providers.vision_model import VLM
from retrieval_engine.rag import create_rag_agent
from retrieval_engine.rag.base_strategy import BaseRAG
from retrieval_engine.rag.strategies.gbc_vanilla import GBCVanillaRAG
from retrieval_engine.rag.gbc.agent import GBCRAG
from retrieval_engine.rag.strategies.graph import GraphRAG
from retrieval_engine.rag.strategies.multimodal import MMRAG
from retrieval_engine.rag.strategies.vanilla import VanillaRAG
from retrieval_engine.configuration.strategies.multimodal import MMConfig
from retrieval_engine.configuration.strategies.gbc import GBCRAGConfig
from retrieval_engine.configuration.strategies.vanilla import VanillaConfig


class _MinimalRAG(BaseRAG):
    def _retrieve(self, query: str, **kwargs):
        return []

    def _create_augmented_prompt(self, query: str):
        return query

    def generation(self, query: str, query_output_dir: str):
        return "", []


class ProviderControllerLifecycleTests(unittest.TestCase):
    def test_llm_wrapper_closes_controller_once_and_breaks_reference(self):
        provider = object.__new__(LLM)
        provider._closed = False
        provider.llm = Mock()
        controller = provider.llm

        provider.close()
        provider.close()

        controller.close.assert_called_once_with()
        self.assertIsNone(provider.llm)

    def test_vlm_wrapper_closes_controller_once_and_breaks_reference(self):
        provider = object.__new__(VLM)
        provider._closed = False
        provider.vlm = Mock()
        controller = provider.vlm

        provider.close()
        provider.close()

        controller.close.assert_called_once_with()
        self.assertIsNone(provider.vlm)

    def test_common_close_continues_after_one_controller_fails(self):
        llm = Mock()
        vlm = Mock()
        vlm.close.side_effect = RuntimeError("VLM close failed")
        rag = _MinimalRAG(llm)
        rag.vlm = vlm

        rag.close()
        rag.close()

        vlm.close.assert_called_once_with()
        llm.close.assert_called_once_with()
        self.assertIsNone(rag.vlm)
        self.assertIsNone(rag.llm)

    def test_factory_closes_partial_clients_when_vlm_construction_fails(self):
        llm = Mock()
        embedder = Mock()
        store = Mock()
        store.embedding_model = embedder
        with (
            patch("retrieval_engine.rag.factory.LLM", return_value=llm),
            patch("retrieval_engine.rag.factory.VLM", side_effect=RuntimeError("VLM startup failed")),
            self.assertRaisesRegex(RuntimeError, "VLM startup failed"),
        ):
            create_rag_agent(
                strategy_config=MMConfig(),
                llm_config=object(),
                vlm_config=object(),
                vector_store=store,
            )

        llm.close.assert_called_once_with()
        embedder.close.assert_called_once_with()
        store.close.assert_called_once_with()

    def test_factory_closes_both_clients_when_dispatch_fails(self):
        llm = Mock()
        vlm = Mock()
        embedder = Mock()
        store = Mock()
        store.embedding_model = embedder
        with (
            patch("retrieval_engine.rag.factory.LLM", return_value=llm),
            patch("retrieval_engine.rag.factory.VLM", return_value=vlm),
            patch(
                "retrieval_engine.rag.factory._create_rag_agent_from_clients",
                side_effect=RuntimeError("strategy startup failed"),
            ),
            self.assertRaisesRegex(RuntimeError, "strategy startup failed"),
        ):
            create_rag_agent(
                strategy_config=MMConfig(),
                llm_config=object(),
                vlm_config=object(),
                vector_store=store,
            )

        vlm.close.assert_called_once_with()
        llm.close.assert_called_once_with()
        embedder.close.assert_called_once_with()
        store.close.assert_called_once_with()

    def test_unsupported_strategy_fails_before_allocating_models(self):
        with (
            patch("retrieval_engine.rag.factory.LLM") as llm_factory,
            patch("retrieval_engine.rag.factory.VLM") as vlm_factory,
            self.assertRaisesRegex(NotImplementedError, "unsupported"),
        ):
            create_rag_agent(
                strategy_config=SimpleNamespace(strategy="unsupported"),
                llm_config=object(),
                vlm_config=object(),
            )

        llm_factory.assert_not_called()
        vlm_factory.assert_not_called()

    def test_text_only_vanilla_does_not_allocate_a_vlm(self):
        llm = Mock()
        with (
            patch("retrieval_engine.rag.factory.LLM", return_value=llm),
            patch("retrieval_engine.rag.factory.VLM") as vlm_factory,
            patch(
                "retrieval_engine.rag.factory._create_rag_agent_from_clients",
                return_value="agent",
            ) as dispatch,
        ):
            result = create_rag_agent(
                strategy_config=VanillaConfig(),
                llm_config=object(),
                vlm_config=object(),
                vector_store=object(),
            )

        self.assertEqual("agent", result)
        vlm_factory.assert_not_called()
        self.assertIsNone(dispatch.call_args.args[2])

    def test_missing_dependency_fails_before_allocating_models(self):
        with (
            patch("retrieval_engine.rag.factory.LLM") as llm_factory,
            patch("retrieval_engine.rag.factory.VLM") as vlm_factory,
            self.assertRaisesRegex(ValueError, "vector_store"),
        ):
            create_rag_agent(
                strategy_config=VanillaConfig(),
                llm_config=object(),
                vlm_config=object(),
            )

        llm_factory.assert_not_called()
        vlm_factory.assert_not_called()

    def test_gbc_requires_attested_tree_vdb_before_allocating_models(self):
        with (
            patch("retrieval_engine.rag.factory.LLM") as llm_factory,
            patch("retrieval_engine.rag.factory.VLM") as vlm_factory,
            self.assertRaisesRegex(ValueError, "tree_vdb"),
        ):
            create_rag_agent(
                strategy_config=GBCRAGConfig(),
                llm_config=object(),
                vlm_config=object(),
                gbc_index=object(),
            )

        llm_factory.assert_not_called()
        vlm_factory.assert_not_called()

    def test_gbc_factory_passes_both_graph_and_tree_indexes(self):
        llm = Mock()
        vlm = Mock()
        gbc = Mock()
        tree_vdb = Mock()
        agent = Mock()
        with (
            patch("retrieval_engine.rag.factory.LLM", return_value=llm),
            patch("retrieval_engine.rag.factory.VLM", return_value=vlm),
            patch("retrieval_engine.rag.factory.GBCRAG", return_value=agent) as factory,
        ):
            result = create_rag_agent(
                strategy_config=GBCRAGConfig(),
                llm_config=object(),
                vlm_config=object(),
                gbc_index=gbc,
                tree_vdb=tree_vdb,
            )

        self.assertIs(agent, result)
        factory.assert_called_once_with(
            llm=llm,
            vlm=vlm,
            config=ANY,
            gbc_index=gbc,
            tree_vdb=tree_vdb,
            asset_cas_root=None,
        )


class StrategyResourceOwnershipTests(unittest.TestCase):
    @staticmethod
    def _initialize_common(rag, *, llm=None, vlm=None):
        BaseRAG.__init__(rag, llm or Mock())
        if vlm is not None:
            rag.vlm = vlm

    def test_vanilla_closes_embedder_store_and_common_models_after_error(self):
        rag = object.__new__(VanillaRAG)
        llm = Mock()
        self._initialize_common(rag, llm=llm)
        rag.cfg = SimpleNamespace(retrieval_method="vector")
        embedder = Mock()
        embedder.close.side_effect = RuntimeError("embedder close failed")
        store = Mock()
        store.embedding_model = embedder
        rag.vdb = store

        rag.close()
        rag.close()

        embedder.close.assert_called_once_with()
        store.close.assert_called_once_with()
        llm.close.assert_called_once_with()

    def test_gbc_vanilla_closes_shared_embedder_and_both_stores_once(self):
        rag = object.__new__(GBCVanillaRAG)
        llm = Mock()
        vlm = Mock()
        self._initialize_common(rag, llm=llm, vlm=vlm)
        embedder = Mock()
        tree_store = Mock()
        graph_store = Mock()
        rag.embedder = embedder
        rag.tree_vdb = tree_store
        rag.graph_vdb = graph_store

        rag.close()
        rag.close()

        embedder.close.assert_called_once_with()
        tree_store.close.assert_called_once_with()
        graph_store.close.assert_called_once_with()
        vlm.close.assert_called_once_with()
        llm.close.assert_called_once_with()

    def test_mmr_closes_text_embedder_store_and_common_models(self):
        rag = object.__new__(MMRAG)
        llm = Mock()
        vlm = Mock()
        self._initialize_common(rag, llm=llm, vlm=vlm)
        embedder = Mock()
        store = Mock()
        store.embedding_model = embedder
        rag.vdb = store

        rag.close()
        rag.close()

        embedder.close.assert_called_once_with()
        store.close.assert_called_once_with()
        vlm.close.assert_called_once_with()
        llm.close.assert_called_once_with()

    def test_graph_closes_gbc_index_and_common_models(self):
        rag = object.__new__(GraphRAG)
        llm = Mock()
        vlm = Mock()
        self._initialize_common(rag, llm=llm, vlm=vlm)
        gbc = Mock()
        rag.gbc_index = gbc
        rag.embedder = Mock()

        rag.close()
        rag.close()

        gbc.close.assert_called_once_with()
        vlm.close.assert_called_once_with()
        llm.close.assert_called_once_with()

    def test_gbc_closes_reranker_index_and_common_models(self):
        rag = object.__new__(GBCRAG)
        llm = Mock()
        vlm = Mock()
        self._initialize_common(rag, llm=llm, vlm=vlm)
        reranker = Mock()
        gbc = Mock()
        tree_embedder = Mock()
        tree_vdb = Mock()
        rag.reranker = reranker
        rag.gbc_index = gbc
        rag.embedder = Mock()
        rag.tree_embedder = tree_embedder
        rag.tree_vdb = tree_vdb

        rag.close()
        rag.close()

        reranker.close.assert_called_once_with()
        tree_embedder.close.assert_called_once_with()
        tree_vdb.close.assert_called_once_with()
        gbc.close.assert_called_once_with()
        vlm.close.assert_called_once_with()
        llm.close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
