"""Compatibility exports for previously serialized indexes.

New code should import from :mod:`retrieval_engine.indexes.bm25`.
"""

from retrieval_engine.indexes.bm25 import (
    BM25,
)

__all__ = ['BM25']
