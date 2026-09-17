"""Import compatibility for saved indexes; use the retrieval_engine package."""

from retrieval_engine.indexes.bm25 import (
    BM25,
)

__all__ = ['BM25']
