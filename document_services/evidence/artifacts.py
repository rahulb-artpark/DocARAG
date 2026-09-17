"""Shared canonical-evidence artifact names and schema revisions.

Keeping these constants outside BookRAG Core lets dependency-light ingestion
providers emit the same envelope as the legacy MinerU binding without
importing Core.  Core may re-export these values for compatibility.
"""

from document_services.domain.base import CONTRACT_SCHEMA_VERSION


CANONICAL_EVIDENCE_SCHEMA_VERSION = "docarag.canonical-evidence.v1"
CANONICAL_EVIDENCE_FILE = "canonical_blocks.json"
CANONICAL_CONTRACT_SCHEMA_VERSION = CONTRACT_SCHEMA_VERSION
CANONICAL_EVIDENCE_ARTIFACT_ROLE = "canonical_ingestion"
ROUTED_PARSING_AUDIT_SCHEMA_VERSION = "docarag.routed-bookrag-composition.v1"
ROUTED_CANONICAL_INGESTION_SCHEMA_VERSION = (
    "docarag.routed-canonical-ingestion.v1"
)


__all__ = [
    "CANONICAL_CONTRACT_SCHEMA_VERSION",
    "CANONICAL_EVIDENCE_ARTIFACT_ROLE",
    "CANONICAL_EVIDENCE_FILE",
    "CANONICAL_EVIDENCE_SCHEMA_VERSION",
    "ROUTED_PARSING_AUDIT_SCHEMA_VERSION",
    "ROUTED_CANONICAL_INGESTION_SCHEMA_VERSION",
]
