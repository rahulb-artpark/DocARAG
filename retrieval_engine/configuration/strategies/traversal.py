# retrieval_engine/configuration/strategies/traversal.py
from pydantic import Field
from typing import Literal
from retrieval_engine.configuration.strategies.base import BaseRAGStrategyConfig

class TraverseRAGConfig(BaseRAGStrategyConfig):
    strategy: Literal["traverse"] = "traverse"
    max_depth: int = Field(
        default=5,
        description="The maximum depth for the document tree traversal."
    )
    