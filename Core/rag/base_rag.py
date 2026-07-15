from abc import ABC, abstractmethod
from Core.provider.llm import LLM
from typing import List, Tuple, Any
import logging


log = logging.getLogger(__name__)


class BaseRAG(ABC):
    def __init__(
        self,
        llm: LLM,
        name: str = "Base RAG",
        description: str = "Base Retrieval Augmented Generation",
    ):
        self.llm = llm
        self.name = name
        self.description = description
        self._common_resources_closed = False

    @abstractmethod
    def _retrieve(self, query: str, **kwargs):
        pass

    @abstractmethod
    def _create_augmented_prompt(self, query: str) -> str:
        pass

    @abstractmethod
    def generation(self, query: str, query_output_dir: str) -> Tuple[str, List[Any]]:
        """
        Generates an answer for a given query and returns the answer along with the context used.
        Returns:
            Tuple[str, List[Any]]: A tuple contains final answer and the retrieval ids
        """
        raise NotImplementedError

    def close(self):
        """Release common model clients once, continuing after close failures.

        Strategy implementations own their retrieval resources and should
        release those first, then call ``super().close()`` in a ``finally``
        block.  Cleanup errors are logged so they cannot mask an active query
        failure or prevent the remaining GPU/client resources from closing.
        """

        if getattr(self, "_common_resources_closed", False):
            return
        self._common_resources_closed = True
        for attribute in ("vlm", "llm"):
            resource = getattr(self, attribute, None)
            setattr(self, attribute, None)
            close = getattr(resource, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    log.exception("Failed to close RAG %s resource", attribute)

    def _close_owned_resources(self, *resources: tuple[str, object]) -> None:
        """Close distinct strategy-owned resources without aborting cleanup."""

        seen: set[int] = set()
        for label, resource in resources:
            if resource is None or id(resource) in seen:
                continue
            seen.add(id(resource))
            close = getattr(resource, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    log.exception("Failed to close RAG %s resource", label)
