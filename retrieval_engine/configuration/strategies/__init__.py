# retrieval_engine/configuration/strategies/__init__.py

from retrieval_engine.configuration.strategies.traversal import TraverseRAGConfig
from retrieval_engine.configuration.strategies.gbc import GBCRAGConfig
from retrieval_engine.configuration.strategies.multimodal import MMConfig
from retrieval_engine.configuration.strategies.graph import GraphRAGConfig
from retrieval_engine.configuration.strategies.gbc_vanilla import GBCVanillaConfig
from retrieval_engine.configuration.strategies.vanilla import VanillaConfig

ALL_STRATEGY_CONFIGS = (
    TraverseRAGConfig,
    GBCRAGConfig,
    MMConfig,
    GraphRAGConfig,
    VanillaConfig,
    GBCVanillaConfig,
)
