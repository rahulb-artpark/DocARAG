"""Import compatibility for saved indexes; use the retrieval_engine package."""

from retrieval_engine.indexes.tree import (
    DocumentTree,
    FURNITURE_NODE_TYPES,
    MetaInfo,
    NodeType,
    TreeNode,
)

__all__ = ['DocumentTree', 'FURNITURE_NODE_TYPES', 'MetaInfo', 'NodeType', 'TreeNode']
