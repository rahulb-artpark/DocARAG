"""Compatibility exports for previously serialized indexes.

New code should import from :mod:`retrieval_engine.indexes.graph`.
"""

from retrieval_engine.indexes.graph import (
    Entity,
    Graph,
    Relationship,
    SetEncoder,
)

__all__ = ['Entity', 'Graph', 'Relationship', 'SetEncoder']
