"""Dependency-light application use cases shared by CLI and API adapters."""

from .contracts import (
    DOCUMENT_EXTRACTION_REQUEST_SCHEMA_VERSION,
    DOCUMENT_EXTRACTION_RESULT_SCHEMA_VERSION,
    DOCUMENT_EXTRACTION_SCHEMA_VERSION,
    MAX_DOCUMENT_EXTRACTION_REQUEST_BYTES,
    MAX_DOCUMENT_EXTRACTION_RESULT_BYTES,
    DocumentExtractionRequest,
    DocumentExtractionResult,
)
from .extract_document import (
    DocumentExtractionArtifactError,
    DocumentExtractionBindingError,
    DocumentExtractionError,
    DocumentExtractionPolicy,
    ExtractDocumentUseCase,
)
from .ports import (
    CanonicalExtractionPayload,
    DocumentExtractionBackend,
    RoutedAuditValidator,
    ValidatedRoutedAudit,
)
from .routed import RoutedCanonicalProvider, RoutedDocumentExtractionBackend

__all__ = [
    "DOCUMENT_EXTRACTION_REQUEST_SCHEMA_VERSION",
    "DOCUMENT_EXTRACTION_RESULT_SCHEMA_VERSION",
    "DOCUMENT_EXTRACTION_SCHEMA_VERSION",
    "MAX_DOCUMENT_EXTRACTION_REQUEST_BYTES",
    "MAX_DOCUMENT_EXTRACTION_RESULT_BYTES",
    "CanonicalExtractionPayload",
    "DocumentExtractionArtifactError",
    "DocumentExtractionBackend",
    "DocumentExtractionBindingError",
    "DocumentExtractionError",
    "DocumentExtractionPolicy",
    "DocumentExtractionRequest",
    "DocumentExtractionResult",
    "ExtractDocumentUseCase",
    "RoutedAuditValidator",
    "RoutedCanonicalProvider",
    "RoutedDocumentExtractionBackend",
    "ValidatedRoutedAudit",
]
