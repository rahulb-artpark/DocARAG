"""Import compatibility for saved indexes; use the retrieval_engine package."""

from retrieval_engine.indexes.graph import (
    Entity,
    Graph,
    Relationship,
    SetEncoder,
)

__all__ = ['Entity', 'Graph', 'Relationship', 'SetEncoder']
