from pydantic import BaseModel, ConfigDict, Field
from typing import Union
from retrieval_engine.configuration.strategies import ALL_STRATEGY_CONFIGS

StrategyConfig = Union[*ALL_STRATEGY_CONFIGS]


class RAGConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strategy_config: StrategyConfig = Field(..., discriminator="strategy")
