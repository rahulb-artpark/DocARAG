"""Adapter from the application extraction port to routed ingestion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from docarag.application.contracts import DocumentExtractionRequest
from docarag.application.ports import (
    CanonicalExtractionPayload,
    RoutedAuditValidator,
)
from docarag.domain.base import require_sha256


@runtime_checkable
class RoutedCanonicalProvider(Protocol):
    """Structural subset implemented by RoutedCanonicalIngestionProvider."""

    @property
    def configuration_sha256(self) -> str:
        ...

    def ingest(self, cfg: object, *, reforce: bool = False) -> object:
        ...


@dataclass(frozen=True, slots=True)
class _SourceConfiguration:
    display_name: str
    original_uri: str
    sha256: str
    snapshot_path: str


@dataclass(frozen=True, slots=True)
class _RoutedIngestionConfiguration:
    source: _SourceConfiguration
    pdf_path: str
    save_path: str


class RoutedDocumentExtractionBackend:
    """Translate an application request without exposing BookRAG config types."""

    def __init__(
        self,
        provider: RoutedCanonicalProvider,
        *,
        audit_validator: RoutedAuditValidator | None = None,
    ) -> None:
        if not isinstance(provider, RoutedCanonicalProvider):
            raise TypeError("provider must implement RoutedCanonicalProvider")
        try:
            configuration_sha256 = require_sha256(
                provider.configuration_sha256,
                "provider.configuration_sha256",
            )
        except (TypeError, ValueError) as exc:
            raise TypeError(str(exc)) from exc
        validator = audit_validator or getattr(provider, "audit_validator", None)
        if not isinstance(validator, RoutedAuditValidator):
            raise TypeError(
                "audit_validator must implement RoutedAuditValidator; pass one "
                "explicitly or use a provider that exposes audit_validator"
            )
        self._provider = provider
        self._audit_validator = validator
        self._configuration_sha256 = configuration_sha256

    @property
    def configuration_sha256(self) -> str:
        return self._configuration_sha256

    @property
    def audit_validator(self) -> RoutedAuditValidator:
        return self._audit_validator

    def extract(
        self,
        request: DocumentExtractionRequest,
    ) -> CanonicalExtractionPayload:
        if not isinstance(request, DocumentExtractionRequest):
            raise TypeError("request must be DocumentExtractionRequest")
        if self._provider.configuration_sha256 != self._configuration_sha256:
            raise RuntimeError(
                "routed provider configuration identity changed after composition"
            )
        cfg = _RoutedIngestionConfiguration(
            source=_SourceConfiguration(
                display_name=request.display_name,
                original_uri=request.source_uri,
                sha256=request.source_sha256,
                snapshot_path=request.source_path,
            ),
            pdf_path=request.source_path,
            save_path=request.output_root,
        )
        supplied = self._provider.ingest(cfg, reforce=request.force)
        try:
            return CanonicalExtractionPayload(
                document=supplied.document,
                canonical_blocks=tuple(supplied.canonical_blocks),
                canonical_artifact_path=supplied.canonical_artifact_path,
                audit_artifact_path=supplied.audit_artifact_path,
                canonical_blocks_sha256=supplied.canonical_blocks_sha256,
            )
        except (AttributeError, TypeError) as exc:
            raise RuntimeError(
                "routed provider returned an invalid canonical payload"
            ) from exc


__all__ = [
    "RoutedCanonicalProvider",
    "RoutedDocumentExtractionBackend",
]
