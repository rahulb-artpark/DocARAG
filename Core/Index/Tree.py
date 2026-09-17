"""Compatibility exports for previously serialized indexes.

New code should import from :mod:`retrieval_engine.indexes.tree`.
"""

from retrieval_engine.indexes.tree import (
    DocumentTree,
    FURNITURE_NODE_TYPES,
    MetaInfo,
    NodeType,
    TreeNode,
)

__all__ = ['DocumentTree', 'FURNITURE_NODE_TYPES', 'MetaInfo', 'NodeType', 'TreeNode']
