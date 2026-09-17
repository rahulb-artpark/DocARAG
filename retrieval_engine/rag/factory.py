"""Construct a RAG strategy and transfer ownership of its model resources."""

from typing import Any, Union
import logging
from retrieval_engine.configuration.language_model import LLMConfig
from retrieval_engine.configuration.vision_model import VLMConfig
from retrieval_engine.configuration.retrieval import RAGConfig
from retrieval_engine.rag.base_strategy import BaseRAG
from retrieval_engine.rag.session import RAGSession

# Import the specific strategy config classes and the Union type
from retrieval_engine.configuration.strategies import ALL_STRATEGY_CONFIGS
from retrieval_engine.configuration.strategies.traversal import TraverseRAGConfig
from retrieval_engine.configuration.strategies.multimodal import MMConfig 
from retrieval_engine.configuration.strategies.gbc import GBCRAGConfig
from retrieval_engine.configuration.strategies.graph import GraphRAGConfig
from retrieval_engine.configuration.strategies.vanilla import VanillaConfig
from retrieval_engine.configuration.strategies.gbc_vanilla import GBCVanillaConfig

from retrieval_engine.rag.strategies.traverse import TraverseAgent
from retrieval_engine.rag.gbc.agent import GBCRAG
from retrieval_engine.rag.strategies.multimodal import MMRAG
from retrieval_engine.rag.strategies.graph import GraphRAG
from retrieval_engine.rag.strategies.vanilla import VanillaRAG
from retrieval_engine.rag.strategies.gbc_vanilla import GBCVanillaRAG

from retrieval_engine.providers.language_model import LLM
from retrieval_engine.providers.vision_model import VLM


log = logging.getLogger(__name__)

# Define the type for the strategy_config parameter
StrategyConfig = Union[*ALL_STRATEGY_CONFIGS]


def _validate_strategy_dependencies(
    strategy_config: StrategyConfig,
    dependencies: dict[str, Any],
) -> bool:
    """Fail before model allocation and return whether the strategy needs VLM."""

    if isinstance(strategy_config, TraverseRAGConfig):
        if dependencies.get("tree_index") is None:
            raise ValueError("TraverseAgent requires a 'tree_index' in dependencies.")
        return True
    if isinstance(strategy_config, GBCRAGConfig):
        if dependencies.get("corpus_bundle") is not None:
            return True
        missing = [
            name
            for name in ("gbc_index", "tree_vdb")
            if dependencies.get(name) is None
        ]
        if missing:
            raise ValueError("GBCRAG requires dependencies: " + ", ".join(missing))
        return True
    if isinstance(strategy_config, GraphRAGConfig):
        if dependencies.get("gbc_index") is None:
            raise ValueError(
                f"{strategy_config.strategy} requires a 'gbc_index' in dependencies."
            )
        return True
    if isinstance(strategy_config, VanillaConfig):
        dependency = (
            "bm25" if strategy_config.retrieval_method == "bm25" else "vector_store"
        )
        if dependencies.get(dependency) is None:
            raise ValueError(
                f"VanillaRAG requires a '{dependency}' in dependencies."
            )
        return False
    if isinstance(strategy_config, GBCVanillaConfig):
        missing = [
            name
            for name in ("tree_vdb", "graph_vdb")
            if dependencies.get(name) is None
        ]
        if missing:
            raise ValueError(
                "GBCVanillaRAG requires dependencies: " + ", ".join(missing)
            )
        # This strategy currently generates from retrieved text only.
        return False
    if isinstance(strategy_config, MMConfig):
        if dependencies.get("vector_store") is None:
            raise ValueError("MMRAG requires a 'vector_store' in dependencies.")
        return True
    strategy_name = getattr(strategy_config, "strategy", "unknown")
    raise NotImplementedError(
        f"RAG agent for strategy '{strategy_name}' is not implemented."
    )


def _close_failed_dependencies(dependencies: dict[str, Any]) -> None:
    """Release dependency-loader resources when agent construction aborts.

    Vector stores do not own their embedding providers, and two stores may
    share one provider.  Close distinct embedders first, then each dependency;
    a store close also releases any vector-index read lease attached to it.
    """

    resources = tuple(dependencies.items())
    seen: set[int] = set()

    def close_once(label: str, resource: object) -> None:
        if resource is None or id(resource) in seen:
            return
        seen.add(id(resource))
        close = getattr(resource, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                log.exception("Failed to close partial RAG dependency %s", label)

    for name, resource in resources:
        close_once(f"{name}.embedding_model", getattr(resource, "embedding_model", None))
    for name, resource in resources:
        close_once(name, resource)


def create_rag_agent(
    # The first parameter is now the specific strategy config object
    strategy_config: StrategyConfig,
    llm_config: LLMConfig,
    vlm_config: VLMConfig,
    **dependencies: Any,
) -> BaseRAG:
    """
    Factory function to create a RAG agent based on the provided strategy configuration.
    """
    # Get the strategy name directly from the config object for logging
    strategy_name = strategy_config.strategy
    print(f"INFO: Creating RAG agent with strategy: '{strategy_name}'")

    needs_vlm = _validate_strategy_dependencies(strategy_config, dependencies)

    # 1. Initialize common dependencies.  Until a strategy is returned, this
    # factory owns both clients and must release them on every partial failure.
    llm_client = None
    vlm_client = None
    try:
        llm_client = LLM(llm_config)
        vlm_client = VLM(vlm_config) if needs_vlm else None

        return _create_rag_agent_from_clients(
            strategy_config,
            llm_client,
            vlm_client,
            dependencies,
        )
    except BaseException:
        for client in (vlm_client, llm_client):
            close = getattr(client, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    # Preserve the construction failure; cleanup is best effort
                    # but continues for the other already-created resource.
                    log.exception("Failed to close partially constructed model client")
        _close_failed_dependencies(dependencies)
        raise


def _create_rag_agent_from_clients(
    strategy_config: StrategyConfig,
    llm_client: LLM,
    vlm_client: VLM | None,
    dependencies: dict[str, Any],
) -> BaseRAG:
    """Dispatch a validated strategy after common clients are constructed."""

    strategy_name = strategy_config.strategy

    # 2. Use isinstance for type-safe dispatching
    if isinstance(strategy_config, TraverseRAGConfig):
        tree_index = dependencies.get("tree_index")
        if not tree_index:
            raise ValueError("TraverseAgent requires a 'tree_index' in dependencies.")

        # 3. Pass the specific config object to the agent's constructor
        return TraverseAgent(
            config=strategy_config,
            llm=llm_client,
            vlm=vlm_client,
            tree_index=tree_index,
        )
    elif isinstance(strategy_config, GBCRAGConfig):
        assert vlm_client is not None
        corpus_bundle = dependencies.get("corpus_bundle")
        if corpus_bundle is not None:
            return GBCRAG.from_corpus_bundle(
                llm=llm_client,
                vlm=vlm_client,
                config=strategy_config,
                corpus_bundle=corpus_bundle,
                asset_cas_root=dependencies.get("asset_cas_root"),
            )
        gbc_index = dependencies.get("gbc_index")
        tree_vdb = dependencies.get("tree_vdb")
        return GBCRAG(
            llm=llm_client,
            vlm=vlm_client,
            config=strategy_config,
            gbc_index=gbc_index,
            tree_vdb=tree_vdb,
            asset_cas_root=dependencies.get("asset_cas_root"),
        )
    elif isinstance(strategy_config, GraphRAGConfig):
        gbc_index = dependencies.get("gbc_index")
        assert vlm_client is not None
        return GraphRAG(
            llm=llm_client,
            vlm=vlm_client,
            config=strategy_config,
            gbc_index=gbc_index,
        )
    elif isinstance(strategy_config, VanillaConfig):
        if strategy_config.retrieval_method == "bm25":
            bm25 = dependencies.get("bm25")
            return VanillaRAG(
                config=strategy_config,
                llm=llm_client,
                bm25=bm25,
                vector_store=None,
            )
        else:
            vector_store = dependencies.get("vector_store")
            return VanillaRAG(
                config=strategy_config,
                llm=llm_client,
                vector_store=vector_store,
                bm25=None,
            )
    elif isinstance(strategy_config, GBCVanillaConfig):
        tree_vdb = dependencies.get("tree_vdb")
        graph_vdb = dependencies.get("graph_vdb")
        return GBCVanillaRAG(
            llm=llm_client,
            vlm=None,
            config=strategy_config,
            tree_vdb=tree_vdb,
            graph_vdb=graph_vdb,
        )
    elif isinstance(strategy_config, MMConfig):
        vector_store = dependencies.get("vector_store")
        if vector_store is None:
            raise ValueError("MMRAG requires a 'vector_store' in dependencies.")

        assert vlm_client is not None

        return MMRAG(
            config=strategy_config,
            llm=llm_client,
            vlm=vlm_client,
            vector_store=vector_store,
            topk=strategy_config.topk if hasattr(strategy_config, 'topk') else 3,
        )

    else:
        raise NotImplementedError(
            f"RAG agent for strategy '{strategy_name}' is not implemented."
        )
