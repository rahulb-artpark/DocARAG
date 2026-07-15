"""Explicit ownership for a constructed RAG agent and its resources."""

from __future__ import annotations

import logging
from threading import Lock
from typing import Generic, TypeVar

from Core.rag.base_rag import BaseRAG


log = logging.getLogger(__name__)

RAGT = TypeVar("RAGT", bound=BaseRAG)


class RAGSession(Generic[RAGT]):
    """Own or borrow one RAG agent through an idempotent context boundary.

    ``owns_agent=True`` means this session closes the agent when the context is
    left, including when query setup or execution fails.  ``owns_agent=False``
    is an explicit borrow: the caller retains lifecycle responsibility.  The
    session marks ownership as released *before* invoking ``close()``, ensuring
    a failing close operation is never repeated by another exit path.
    """

    def __init__(self, rag_agent: RAGT, *, owns_agent: bool = True) -> None:
        if rag_agent is None:
            raise ValueError("rag_agent is required")
        close = getattr(rag_agent, "close", None)
        if not callable(close):
            raise TypeError("rag_agent must provide close()")
        self._rag_agent: RAGT | None = rag_agent
        self._owns_agent = bool(owns_agent)
        self._closed = False
        self._close_lock = Lock()

    @property
    def rag_agent(self) -> RAGT:
        """Return the live agent or reject use after session shutdown."""

        with self._close_lock:
            if self._closed or self._rag_agent is None:
                raise RuntimeError("RAGSession is closed")
            return self._rag_agent

    @property
    def owns_agent(self) -> bool:
        return self._owns_agent

    @property
    def closed(self) -> bool:
        with self._close_lock:
            return self._closed

    def __enter__(self) -> "RAGSession[RAGT]":
        # Accessing the property catches accidental reuse after an earlier exit.
        self.rag_agent
        return self

    def __exit__(self, exc_type, _exc, _traceback) -> bool:
        try:
            self.close()
        except Exception:
            if exc_type is None:
                raise
            # Never hide the active parsing/query failure behind cleanup.  The
            # ownership bit was already consumed, so cleanup is not retried.
            log.exception("Failed to close RAG session after an active failure")
        return False

    def close(self) -> None:
        """Release an owned agent exactly once and break the session reference."""

        with self._close_lock:
            if self._closed:
                return
            self._closed = True
            rag_agent = self._rag_agent
            self._rag_agent = None
            should_close = self._owns_agent

        if should_close and rag_agent is not None:
            rag_agent.close()

