from retrieval_engine.configuration.strategies.base import BaseRAGStrategyConfig
from typing import Literal
from pydantic import Field
from retrieval_engine.configuration.vector_store import VDBConfig


class MMConfig(BaseRAGStrategyConfig):
    vdb_config: VDBConfig = Field(default_factory=VDBConfig)
    strategy: Literal["mmr"] = "mmr"
    topk: int = Field(
        default=5, description="The number of topk retrieval results for MMRAG."
    )
