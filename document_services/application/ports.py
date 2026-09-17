"""Dependency-inverted ports used by document extraction application logic."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import os
from typing import Protocol, runtime_checkable

from document_services.application.contracts import DocumentExtractionRequest
from document_services.domain.base import (
    ContractValidationError,
    require_non_empty,
    require_sha256,
)
from document_services.domain.evidence import CanonicalBlock, DocumentRef


@dataclass(frozen=True, slots=True)
class CanonicalExtractionPayload:
    """In-process handoff from a parser backend to the application use case."""

    document: DocumentRef
    canonical_blocks: tuple[CanonicalBlock, ...]
    canonical_artifact_path: str | os.PathLike[str]
    audit_artifact_path: str | os.PathLike[str]
    canonical_blocks_sha256: str

    def __post_init__(self) -> None:
        try:
            require_sha256(self.canonical_blocks_sha256, "canonical_blocks_sha256")
        except ContractValidationError as exc:
            raise ValueError(str(exc)) from exc


@dataclass(frozen=True, slots=True)
class ValidatedRoutedAudit:
    """Dependency-neutral facts returned by a full routed-audit replay."""

    projection_id: str
    parse_request_id: str
    routed_parsing_result_id: str
    document: DocumentRef
    canonical_blocks: tuple[CanonicalBlock, ...]

    def __post_init__(self) -> None:
        for field_name in (
            "projection_id",
            "parse_request_id",
            "routed_parsing_result_id",
        ):
            require_non_empty(getattr(self, field_name), field_name)
        if not isinstance(self.document, DocumentRef):
            raise ContractValidationError("document must be a DocumentRef")
        blocks = tuple(self.canonical_blocks)
        if not blocks or any(not isinstance(block, CanonicalBlock) for block in blocks):
            raise ContractValidationError(
                "canonical_blocks must contain CanonicalBlock values"
            )
        if any(
            block.document_version_id != self.document.document_version_id
            for block in blocks
        ):
            raise ContractValidationError(
                "validated audit blocks belong to another document"
            )
        object.__setattr__(self, "canonical_blocks", blocks)


@runtime_checkable
class RoutedAuditValidator(Protocol):
    """Replay a serialized routed audit without exposing composition DTOs."""

    def validate(self, payload: Mapping[str, object]) -> ValidatedRoutedAudit:
        ...


@runtime_checkable
class DocumentExtractionBackend(Protocol):
    """Replaceable parsing backend; vendor/model imports stay behind this port."""

    @property
    def configuration_sha256(self) -> str:
        ...

    def extract(
        self,
        request: DocumentExtractionRequest,
    ) -> CanonicalExtractionPayload:
        ...


__all__ = [
    "CanonicalExtractionPayload",
    "DocumentExtractionBackend",
    "RoutedAuditValidator",
    "ValidatedRoutedAudit",
]
