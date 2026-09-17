"""Compatibility exports for previously serialized indexes.

New code should import from :mod:`retrieval_engine.indexes.gbc`.
"""

from retrieval_engine.indexes.gbc import (
    GBC,
)

__all__ = ['GBC']
