from pydantic import BaseModel, ConfigDict

class BaseRAGStrategyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
