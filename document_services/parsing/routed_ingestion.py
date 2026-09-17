"""Durable routed parsing provider for BookRAG's canonical ingestion seam.

This module is the application boundary between a cheap source-bound page
probe, routed parser orchestration, and BookRAG's structural provider protocol.
It deliberately imports neither ``Core`` nor parser-vendor SDKs.  A successful
run publishes two deterministic artifacts:

* ``routed_parsing_audit.json`` contains the complete replayable routing,
  invocation, reconciliation, and readiness audit; and
* ``canonical_blocks.json`` contains the strictly ready canonical evidence
  accepted by the legacy BookRAG handoff.

The canonical artifact is never published for partial, unresolved, review, or
failed output.  Existing artifacts are reusable only after the live source,
current policies/capabilities, schemas, counts, and content digests all bind.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import hmac
import json
import os
from pathlib import Path
import stat
import tempfile
from typing import cast
from urllib.parse import parse_qsl, urlsplit

from document_services.application.ports import RoutedAuditValidator, ValidatedRoutedAudit
from document_services.evidence.artifacts import (
    CANONICAL_CONTRACT_SCHEMA_VERSION,
    CANONICAL_EVIDENCE_ARTIFACT_ROLE,
    CANONICAL_EVIDENCE_FILE,
    CANONICAL_EVIDENCE_SCHEMA_VERSION,
    ROUTED_CANONICAL_INGESTION_SCHEMA_VERSION,
)
from document_services.parsing.bookrag_handoff import RoutedBookRAGFacade, RoutedBookRAGResult
from document_services.domain.base import (
    ContractValidationError,
    content_sha256,
    json_ready,
    reject_unknown_fields,
    require_mapping,
    require_non_empty,
    require_sequence,
    require_sha256,
    stable_id,
)
from document_services.domain.evidence import CanonicalBlock, DocumentRef
from document_services.domain.parsing import ParseOptions, ParseRequest
from document_services.ports.page_probe import PageProbe
from document_services.parsing.routing import PageFeatures
from document_services.evidence.sources import bind_source_material, local_source_material_binding


ROUTED_INGESTION_SCHEMA_VERSION = ROUTED_CANONICAL_INGESTION_SCHEMA_VERSION
ROUTED_AUDIT_FILE = "routed_parsing_audit.json"
LAST_ATTEMPT_AUDIT_FILE = "routed_parsing_last_attempt.json"
CANONICAL_ARTIFACT_FILE = CANONICAL_EVIDENCE_FILE
CANONICAL_ARTIFACT_ROLE = CANONICAL_EVIDENCE_ARTIFACT_ROLE


class RoutedIngestionError(RuntimeError):
    """Base failure at the routed canonical ingestion boundary."""


class RoutedIngestionConfigurationError(RoutedIngestionError):
    """The injected dependencies or duck-typed run configuration are invalid."""


class RoutedIngestionSourceBindingError(RoutedIngestionError):
    """The live PDF bytes do not match their configured immutable identity."""


class RoutedIngestionArtifactError(RoutedIngestionError):
    """A persisted audit/canonical artifact is missing, corrupt, or mismatched."""


class RoutedIngestionArtifactSecurityError(RoutedIngestionArtifactError):
    """An artifact path is a symlink or can escape the configured run root."""


class RoutedIngestionPageLimitError(RoutedIngestionError):
    """The probe emitted more pages than this bounded composition permits."""


class RoutedBookRAGAuditValidator:
    """Replay the complete composition audit behind the application port."""

    def validate(self, payload: Mapping[str, object]) -> ValidatedRoutedAudit:
        if not isinstance(payload, Mapping):
            raise ContractValidationError("routed audit must be a mapping")
        audit = RoutedBookRAGResult.from_dict(payload)
        # Re-run strict projection invariants in addition to reconstructing every
        # nested request, invocation, result, route, and reconciliation DTO.
        audit.to_bookrag_pdf_list()
        return ValidatedRoutedAudit(
            projection_id=audit.projection_id,
            parse_request_id=audit.request.request_id,
            routed_parsing_result_id=audit.parsing.result_id,
            document=audit.request.document,
            canonical_blocks=audit.parsing.canonical_blocks,
        )


@dataclass(frozen=True, slots=True)
class RoutedIngestionPolicy:
    """Resource and artifact bounds for one dependency-light ingestion run."""

    max_page_features: int = 10_000
    max_audit_bytes: int = 512 * 1024 * 1024
    max_canonical_artifact_bytes: int = 512 * 1024 * 1024
    sha256_chunk_bytes: int = 1024 * 1024
    probe_identity: str = "source_bound_pdf_page_probe.v1"

    def __post_init__(self) -> None:
        for field_name, minimum, maximum in (
            ("max_page_features", 1, 1_000_000),
            ("max_audit_bytes", 1024, 4 * 1024 * 1024 * 1024),
            (
                "max_canonical_artifact_bytes",
                1024,
                4 * 1024 * 1024 * 1024,
            ),
            ("sha256_chunk_bytes", 4096, 64 * 1024 * 1024),
        ):
            value = getattr(self, field_name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not minimum <= value <= maximum
            ):
                raise RoutedIngestionConfigurationError(
                    f"{field_name} must be an integer between {minimum} and {maximum}"
                )
        require_non_empty(self.probe_identity, "probe_identity")


@dataclass(frozen=True, slots=True)
class RoutedCanonicalIngestionPayload:
    """Immutable structural payload accepted by Core's provider protocol."""

    document: DocumentRef
    canonical_blocks: tuple[CanonicalBlock, ...]
    canonical_artifact_path: Path
    audit_artifact_path: Path
    canonical_blocks_sha256: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.document, DocumentRef):
            raise RoutedIngestionArtifactError("document must be a DocumentRef")
        blocks = tuple(self.canonical_blocks)
        if not blocks or any(not isinstance(block, CanonicalBlock) for block in blocks):
            raise RoutedIngestionArtifactError(
                "canonical_blocks must contain at least one CanonicalBlock"
            )
        if any(
            block.document_version_id != self.document.document_version_id
            for block in blocks
        ):
            raise RoutedIngestionArtifactError(
                "canonical blocks belong to another document version"
            )
        if not isinstance(self.canonical_artifact_path, Path):
            raise RoutedIngestionArtifactError("canonical_artifact_path must be a Path")
        if not isinstance(self.audit_artifact_path, Path):
            raise RoutedIngestionArtifactError("audit_artifact_path must be a Path")
        if self.canonical_blocks_sha256 is not None:
            try:
                require_sha256(
                    self.canonical_blocks_sha256,
                    "canonical_blocks_sha256",
                )
            except ContractValidationError as exc:
                raise RoutedIngestionArtifactError(str(exc)) from exc
        object.__setattr__(self, "canonical_blocks", blocks)


@dataclass(frozen=True, slots=True)
class _RunContext:
    source_path: Path
    output_root: Path
    audit_path: Path
    last_attempt_audit_path: Path
    canonical_path: Path
    base_document: DocumentRef


class RoutedCanonicalIngestionProvider:
    """Compose a page probe and routed facade behind BookRAG's provider seam.

    The public ``ingest(cfg, reforce=False)`` method intentionally relies only
    on the small attribute surface ``cfg.source``, ``cfg.pdf_path``, and
    ``cfg.save_path``.  This preserves dependency direction: BookRAG may inject
    this object through its structural protocol, while this module never needs
    to import BookRAG or Pydantic configuration types.
    """

    def __init__(
        self,
        page_probe: PageProbe[PageFeatures],
        facade: RoutedBookRAGFacade,
        *,
        parse_options: ParseOptions | None = None,
        policy: RoutedIngestionPolicy | None = None,
    ) -> None:
        if not isinstance(page_probe, PageProbe):
            raise TypeError("page_probe must implement PageProbe")
        if not isinstance(facade, RoutedBookRAGFacade):
            raise TypeError("facade must be RoutedBookRAGFacade")
        options = parse_options or ParseOptions()
        if not isinstance(options, ParseOptions):
            raise TypeError("parse_options must be ParseOptions")
        try:
            _assert_secret_free(options.to_dict(), path="parse_options")
        except RoutedIngestionArtifactError as exc:
            raise RoutedIngestionConfigurationError(str(exc)) from exc
        active_policy = policy or RoutedIngestionPolicy()
        if not isinstance(active_policy, RoutedIngestionPolicy):
            raise TypeError("policy must be RoutedIngestionPolicy")
        self._page_probe = page_probe
        self._facade = facade
        self._audit_validator = RoutedBookRAGAuditValidator()
        self._parse_options = options
        self._policy = active_policy
        self._configuration_sha256 = _configuration_sha256(
            facade,
            options,
            active_policy.probe_identity,
        )

    @property
    def parse_options(self) -> ParseOptions:
        return self._parse_options

    @property
    def policy(self) -> RoutedIngestionPolicy:
        return self._policy

    @property
    def configuration_sha256(self) -> str:
        return self._configuration_sha256

    @property
    def audit_validator(self) -> RoutedAuditValidator:
        return self._audit_validator

    def ingest(
        self,
        cfg: object,
        *,
        reforce: bool = False,
    ) -> RoutedCanonicalIngestionPayload:
        """Return strictly ready, persisted canonical evidence for one PDF."""

        if not isinstance(reforce, bool):
            raise RoutedIngestionConfigurationError("reforce must be a boolean")
        context = self._run_context(cfg)
        prior_ready = False
        if not reforce:
            restored = self._restore(context)
            if restored is not None:
                return restored
        else:
            # Validate and preserve an already-published ready pair.  A forced
            # attempt may replace it only after the new result passes strict
            # projection and both replacement files have been staged.
            prior_ready = self._restore(context) is not None

        features = self._probe_features(context)
        document = _document_with_page_count(context.base_document, len(features))
        request = self._request(document, features)
        result = self._facade.run(request, features)

        audit_payload = result.to_dict()
        _assert_secret_free(audit_payload)
        attempt_audit_path = (
            context.last_attempt_audit_path if prior_ready else context.audit_path
        )
        _atomic_write_json(
            attempt_audit_path,
            audit_payload,
            root=context.output_root,
            max_bytes=self._policy.max_audit_bytes,
        )

        # This method verifies readiness *and* the loss-aware BookRAG projection.
        # Persisting the full audit first ensures a blocked handoff is inspectable.
        result.to_bookrag_pdf_list()

        canonical_payload = _canonical_artifact_payload(
            result,
            configuration_sha256=self._configuration_sha256,
        )
        _assert_secret_free(canonical_payload)
        _atomic_replace_json_pair(
            (
                (
                    context.audit_path,
                    audit_payload,
                    self._policy.max_audit_bytes,
                ),
                (
                    context.canonical_path,
                    canonical_payload,
                    self._policy.max_canonical_artifact_bytes,
                ),
            ),
            root=context.output_root,
        )
        _durable_unlink(context.last_attempt_audit_path, context.output_root)
        return RoutedCanonicalIngestionPayload(
            document=document,
            canonical_blocks=result.parsing.canonical_blocks,
            canonical_artifact_path=context.canonical_path,
            audit_artifact_path=context.audit_path,
            canonical_blocks_sha256=cast(
                str,
                canonical_payload["canonical_blocks_sha256"],
            ),
        )

    def _run_context(self, cfg: object) -> _RunContext:
        source = _required_field(cfg, "source")
        raw_pdf_path = _required_text_field(cfg, "pdf_path")
        raw_save_path = _required_text_field(cfg, "save_path")
        source_path = _resolved_regular_source(Path(raw_pdf_path))

        snapshot_path = _optional_text_field(source, "snapshot_path")
        if snapshot_path is not None:
            try:
                configured_snapshot = Path(snapshot_path).expanduser().resolve(strict=True)
            except (OSError, RuntimeError) as exc:
                raise RoutedIngestionSourceBindingError(
                    f"configured source.snapshot_path is unavailable: {snapshot_path}"
                ) from exc
            if configured_snapshot != source_path:
                raise RoutedIngestionSourceBindingError(
                    "cfg.pdf_path and source.snapshot_path identify different files"
                )

        configured_sha256 = _required_text_field(source, "sha256")
        try:
            require_sha256(configured_sha256, "source.sha256")
        except ContractValidationError as exc:
            raise RoutedIngestionSourceBindingError(str(exc)) from exc
        observed_sha256 = _hash_stable_file(source_path, self._policy.sha256_chunk_bytes)
        if not hmac.compare_digest(configured_sha256, observed_sha256):
            raise RoutedIngestionSourceBindingError(
                "configured source SHA-256 does not match the live PDF bytes"
            )

        display_name = _required_text_field(source, "display_name")
        source_uri = _safe_source_uri(_required_text_field(source, "original_uri"))
        base_document = DocumentRef.identified(
            content_sha256=observed_sha256,
            mime_type="application/pdf",
            source_uri=source_uri,
            display_name=display_name,
        )
        # Keep the original URI authoritative for citations while giving
        # isolated workers a separate, exact locator for the immutable run
        # snapshot. Worker composition roots must still allow-list and verify
        # this path before any vendor service sees the bytes.
        base_document = bind_source_material(
            base_document,
            local_source_material_binding(
                source_path,
                sha256=observed_sha256,
                media_type="application/pdf",
            ),
        )
        output_root = _safe_output_root(Path(raw_save_path))
        audit_path = _artifact_path(output_root, ROUTED_AUDIT_FILE)
        last_attempt_audit_path = _artifact_path(
            output_root, LAST_ATTEMPT_AUDIT_FILE
        )
        canonical_path = _artifact_path(output_root, CANONICAL_ARTIFACT_FILE)
        return _RunContext(
            source_path=source_path,
            output_root=output_root,
            audit_path=audit_path,
            last_attempt_audit_path=last_attempt_audit_path,
            canonical_path=canonical_path,
            base_document=base_document,
        )

    def _probe_features(self, context: _RunContext) -> tuple[PageFeatures, ...]:
        iterator = self._page_probe.probe(
            context.base_document,
            source_path=context.source_path,
        )
        features: list[PageFeatures] = []
        try:
            for feature in iterator:
                if len(features) >= self._policy.max_page_features:
                    raise RoutedIngestionPageLimitError(
                        "page probe exceeded max_page_features "
                        f"({self._policy.max_page_features})"
                    )
                if not isinstance(feature, PageFeatures):
                    raise RoutedIngestionSourceBindingError(
                        "page probe must emit PageFeatures"
                    )
                features.append(feature)
        finally:
            close = getattr(iterator, "close", None)
            if callable(close):
                close()
        if not features:
            raise RoutedIngestionSourceBindingError(
                "page probe returned no pages; routed parsing requires a non-empty PDF"
            )
        expected_indices = list(range(len(features)))
        actual_indices = [feature.page.page_index for feature in features]
        if actual_indices != expected_indices:
            raise RoutedIngestionSourceBindingError(
                "page probe must emit every page exactly once in zero-based order"
            )
        if any(
            feature.page.document_version_id
            != context.base_document.document_version_id
            for feature in features
        ):
            raise RoutedIngestionSourceBindingError(
                "page probe emitted features for another document version"
            )
        if any(feature.mime_type != "application/pdf" for feature in features):
            raise RoutedIngestionSourceBindingError(
                "page probe emitted a non-PDF feature"
            )
        observed_after_probe = _hash_stable_file(
            context.source_path, self._policy.sha256_chunk_bytes
        )
        if not hmac.compare_digest(
            observed_after_probe, context.base_document.content_sha256
        ):
            raise RoutedIngestionSourceBindingError(
                "PDF bytes changed while page features were being materialized"
            )
        return tuple(features)

    def _request(
        self,
        document: DocumentRef,
        features: tuple[PageFeatures, ...],
    ) -> ParseRequest:
        feature_ids = tuple(feature.feature_id for feature in features)
        identity = (
            document.document_version_id,
            self._configuration_sha256,
            feature_ids,
        )
        return ParseRequest(
            document=document,
            pages=tuple(feature.page for feature in features),
            options=self._parse_options,
            run_id=stable_id("routed-ingestion-run", *identity),
            trace_id=stable_id("routed-ingestion-trace", *identity),
            idempotency_key=stable_id("routed-ingestion-idempotency", *identity),
        )

    def _restore(
        self,
        context: _RunContext,
    ) -> RoutedCanonicalIngestionPayload | None:
        audit_exists = _artifact_exists(context.audit_path, context.output_root)
        canonical_exists = _artifact_exists(
            context.canonical_path, context.output_root
        )
        if canonical_exists and not audit_exists:
            raise RoutedIngestionArtifactError(
                "canonical artifact exists without its routed parsing audit"
            )
        if not audit_exists:
            return None

        audit_mapping = _read_json_object(
            context.audit_path,
            root=context.output_root,
            max_bytes=self._policy.max_audit_bytes,
        )
        try:
            audit = RoutedBookRAGResult.from_dict(audit_mapping)
        except (ContractValidationError, TypeError, ValueError) as exc:
            raise RoutedIngestionArtifactError(
                f"routed parsing audit is invalid: {exc}"
            ) from exc
        _validate_audit_binding(
            audit,
            base_document=context.base_document,
            parse_options=self._parse_options,
            facade=self._facade,
        )
        if not audit.is_ready_for_bookrag_handoff:
            if canonical_exists:
                raise RoutedIngestionArtifactError(
                    "blocked routed audit must not have a canonical artifact"
                )
            # A valid failed/partial audit is retryable; only ready work is reused.
            return None
        if not canonical_exists:
            raise RoutedIngestionArtifactError(
                "ready routed audit is missing its canonical artifact"
            )
        try:
            audit.to_bookrag_pdf_list()
        except Exception as exc:
            raise RoutedIngestionArtifactError(
                f"routed audit cannot reproduce the strict BookRAG handoff: {exc}"
            ) from exc

        canonical_mapping = _read_json_object(
            context.canonical_path,
            root=context.output_root,
            max_bytes=self._policy.max_canonical_artifact_bytes,
        )
        blocks = _validate_canonical_artifact(
            canonical_mapping,
            audit=audit,
            configuration_sha256=self._configuration_sha256,
        )
        return RoutedCanonicalIngestionPayload(
            document=audit.request.document,
            canonical_blocks=blocks,
            canonical_artifact_path=context.canonical_path,
            audit_artifact_path=context.audit_path,
            canonical_blocks_sha256=cast(
                str,
                canonical_mapping["canonical_blocks_sha256"],
            ),
        )


def _document_with_page_count(document: DocumentRef, page_count: int) -> DocumentRef:
    return DocumentRef(
        document_id=document.document_id,
        document_version_id=document.document_version_id,
        content_sha256=document.content_sha256,
        mime_type=document.mime_type,
        source_uri=document.source_uri,
        display_name=document.display_name,
        page_count=page_count,
        metadata=document.metadata,
    )


def _configuration_sha256(
    facade: RoutedBookRAGFacade,
    options: ParseOptions,
    probe_identity: str,
) -> str:
    return content_sha256(
        {
            "schema_version": ROUTED_INGESTION_SCHEMA_VERSION,
            "probe_identity": probe_identity,
            "parse_options": options.to_dict(),
            "routing_policy": facade.routing_policy.to_dict(),
            "orchestration_policy": facade.orchestration_policy.to_dict(),
            "reconciliation_policy": facade.reconciliation_policy.to_dict(),
            "parser_capabilities": [
                manifest.to_dict() for manifest in facade.registry.manifests()
            ],
        }
    )


def _canonical_artifact_payload(
    result: RoutedBookRAGResult,
    *,
    configuration_sha256: str,
) -> dict[str, object]:
    blocks = tuple(result.parsing.canonical_blocks)
    serialized_blocks = [block.to_dict() for block in blocks]
    audit_payload = result.to_dict()
    return {
        "artifact_schema_version": CANONICAL_EVIDENCE_SCHEMA_VERSION,
        "contract_schema_version": CANONICAL_CONTRACT_SCHEMA_VERSION,
        "artifact_role": CANONICAL_ARTIFACT_ROLE,
        "producer_schema_version": ROUTED_INGESTION_SCHEMA_VERSION,
        "configuration_sha256": configuration_sha256,
        "audit_sha256": content_sha256(audit_payload),
        "parse_request_id": result.request.request_id,
        "projection_id": result.projection_id,
        "routed_parsing_result_id": result.parsing.result_id,
        "document_version_id": result.request.document.document_version_id,
        "source_sha256": result.request.document.content_sha256,
        "document": result.request.document.to_dict(),
        "block_count": len(blocks),
        "canonical_blocks_sha256": content_sha256(serialized_blocks),
        "canonical_blocks": serialized_blocks,
    }


def _validate_audit_binding(
    audit: RoutedBookRAGResult,
    *,
    base_document: DocumentRef,
    parse_options: ParseOptions,
    facade: RoutedBookRAGFacade,
) -> None:
    document = audit.request.document
    expected_fields = (
        "document_id",
        "document_version_id",
        "content_sha256",
        "mime_type",
        "source_uri",
        "display_name",
    )
    if any(
        getattr(document, field_name) != getattr(base_document, field_name)
        for field_name in expected_fields
    ):
        raise RoutedIngestionArtifactError(
            "routed audit belongs to another source document"
        )
    if document.page_count is None or document.page_count < 1:
        raise RoutedIngestionArtifactError(
            "routed audit document requires a positive page_count"
        )
    if tuple(page.page_index for page in audit.request.pages) != tuple(
        range(document.page_count)
    ):
        raise RoutedIngestionArtifactError(
            "routed audit does not cover every source page exactly once"
        )
    if audit.request.options != parse_options:
        raise RoutedIngestionArtifactError(
            "routed audit parse options differ from the current provider"
        )
    if audit.parsing.orchestration_policy != facade.orchestration_policy:
        raise RoutedIngestionArtifactError(
            "routed audit orchestration policy differs from the current provider"
        )
    if (
        audit.parsing.reconciliation_policy_sha256
        != facade.reconciliation_policy.policy_sha256
    ):
        raise RoutedIngestionArtifactError(
            "routed audit reconciliation policy differs from the current provider"
        )
    policy_id = facade.routing_policy.policy_id
    if any(page.route.policy_id != policy_id for page in audit.parsing.pages):
        raise RoutedIngestionArtifactError(
            "routed audit routing policy differs from the current provider"
        )
    capability_by_adapter = {
        manifest.adapter_name: manifest.capability_id
        for manifest in facade.registry.manifests()
    }
    if any(
        capability_by_adapter.get(invocation.adapter_name)
        != invocation.capability_id
        for invocation in audit.parsing.invocations
    ):
        raise RoutedIngestionArtifactError(
            "routed audit parser capabilities differ from the current provider"
        )


def _validate_canonical_artifact(
    payload: Mapping[str, object],
    *,
    audit: RoutedBookRAGResult,
    configuration_sha256: str,
) -> tuple[CanonicalBlock, ...]:
    allowed = {
        "artifact_schema_version",
        "contract_schema_version",
        "artifact_role",
        "producer_schema_version",
        "configuration_sha256",
        "audit_sha256",
        "parse_request_id",
        "projection_id",
        "routed_parsing_result_id",
        "document_version_id",
        "source_sha256",
        "document",
        "block_count",
        "canonical_blocks_sha256",
        "canonical_blocks",
    }
    try:
        reject_unknown_fields(payload, allowed, dto="CanonicalIngestionArtifact")
        if (
            payload.get("artifact_schema_version")
            != CANONICAL_EVIDENCE_SCHEMA_VERSION
        ):
            raise ContractValidationError(
                "unsupported canonical artifact schema_version"
            )
        if (
            payload.get("contract_schema_version")
            != CANONICAL_CONTRACT_SCHEMA_VERSION
        ):
            raise ContractValidationError(
                "unsupported canonical artifact contract_schema_version"
            )
        if payload.get("artifact_role") != CANONICAL_ARTIFACT_ROLE:
            raise ContractValidationError("canonical artifact role mismatch")
        if payload.get("producer_schema_version") != ROUTED_INGESTION_SCHEMA_VERSION:
            raise ContractValidationError("canonical artifact producer mismatch")
        require_sha256(
            payload.get("configuration_sha256"), "configuration_sha256"
        )
        require_sha256(payload.get("audit_sha256"), "audit_sha256")
        require_sha256(payload.get("source_sha256"), "source_sha256")
        require_sha256(
            payload.get("canonical_blocks_sha256"),
            "canonical_blocks_sha256",
        )
        document = DocumentRef.from_dict(
            require_mapping(payload.get("document"), "document")
        )
        raw_blocks = require_sequence(
            payload.get("canonical_blocks"), "canonical_blocks"
        )
        blocks = tuple(
            CanonicalBlock.from_dict(
                require_mapping(raw_block, f"canonical_blocks[{index}]")
            )
            for index, raw_block in enumerate(raw_blocks)
        )
    except (ContractValidationError, TypeError, ValueError) as exc:
        raise RoutedIngestionArtifactError(
            f"canonical artifact is invalid: {exc}"
        ) from exc

    count = payload.get("block_count")
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise RoutedIngestionArtifactError(
            "block_count must be a positive integer"
        )
    expected_blocks = tuple(audit.parsing.canonical_blocks)
    serialized_blocks = [block.to_dict() for block in blocks]
    expected_bindings = {
        "configuration_sha256": configuration_sha256,
        "audit_sha256": content_sha256(audit.to_dict()),
        "parse_request_id": audit.request.request_id,
        "projection_id": audit.projection_id,
        "routed_parsing_result_id": audit.parsing.result_id,
        "document_version_id": audit.request.document.document_version_id,
        "source_sha256": audit.request.document.content_sha256,
        "canonical_blocks_sha256": content_sha256(serialized_blocks),
    }
    for field_name, expected in expected_bindings.items():
        if payload.get(field_name) != expected:
            raise RoutedIngestionArtifactError(
                f"canonical artifact {field_name} binding mismatch"
            )
    if document != audit.request.document:
        raise RoutedIngestionArtifactError(
            "canonical artifact document differs from its routed audit"
        )
    if count != len(blocks) or count != len(expected_blocks):
        raise RoutedIngestionArtifactError(
            "canonical artifact block count mismatch"
        )
    if blocks != expected_blocks:
        raise RoutedIngestionArtifactError(
            "canonical artifact blocks differ from their routed audit"
        )
    return blocks


def _required_field(value: object, field_name: str) -> object:
    result = _field(value, field_name, missing=None)
    if result is None:
        raise RoutedIngestionConfigurationError(
            f"cfg.{field_name} is required for routed ingestion"
        )
    return result


def _required_text_field(value: object, field_name: str) -> str:
    result = _field(value, field_name, missing=None)
    try:
        return require_non_empty(result, field_name)
    except ContractValidationError as exc:
        raise RoutedIngestionConfigurationError(str(exc)) from exc


def _optional_text_field(value: object, field_name: str) -> str | None:
    result = _field(value, field_name, missing=None)
    if result is None:
        return None
    try:
        return require_non_empty(result, field_name)
    except ContractValidationError as exc:
        raise RoutedIngestionConfigurationError(str(exc)) from exc


def _field(value: object, field_name: str, *, missing: object) -> object:
    if isinstance(value, Mapping):
        return value.get(field_name, missing)
    return getattr(value, field_name, missing)


def _resolved_regular_source(path: Path) -> Path:
    try:
        resolved = path.expanduser().resolve(strict=True)
        status = resolved.stat()
    except (OSError, RuntimeError) as exc:
        raise RoutedIngestionSourceBindingError(
            f"PDF source is unavailable: {path}"
        ) from exc
    if not stat.S_ISREG(status.st_mode):
        raise RoutedIngestionSourceBindingError(
            f"PDF source is not a regular file: {resolved}"
        )
    return resolved


def _hash_stable_file(path: Path, chunk_bytes: int) -> str:
    try:
        before = path.stat()
        digest = hashlib.sha256()
        with path.open("rb") as source:
            while chunk := source.read(chunk_bytes):
                digest.update(chunk)
        after = path.stat()
    except OSError as exc:
        raise RoutedIngestionSourceBindingError(
            f"could not hash PDF source: {path}"
        ) from exc
    fingerprint_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    fingerprint_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if fingerprint_before != fingerprint_after:
        raise RoutedIngestionSourceBindingError(
            "PDF source changed while its SHA-256 was being calculated"
        )
    return digest.hexdigest()


_SENSITIVE_URI_KEYS = frozenset(
    {
        "access_token",
        "api_key",
        "apikey",
        "authorization",
        "credential",
        "password",
        "refresh_token",
        "secret",
        "signature",
        "token",
        "x_amz_credential",
        "x_amz_signature",
    }
)
_SENSITIVE_FIELDS = _SENSITIVE_URI_KEYS | {
    "cookie",
    "passwd",
    "private_key",
    "set_cookie",
}


def _normalized_secret_key(value: str) -> str:
    return value.lower().replace("-", "_").replace(" ", "_")


def _is_sensitive_key(value: str) -> bool:
    normalized = _normalized_secret_key(value)
    return normalized in _SENSITIVE_FIELDS or any(
        normalized.endswith(f"_{suffix}")
        for suffix in ("api_key", "password", "secret", "token")
    )


def _safe_source_uri(value: str) -> str:
    try:
        parsed = urlsplit(value)
    except ValueError as exc:
        raise RoutedIngestionConfigurationError("source.original_uri is invalid") from exc
    if parsed.username is not None or parsed.password is not None:
        raise RoutedIngestionConfigurationError(
            "source.original_uri must not contain credentials"
        )
    if any(_is_sensitive_key(key) for key, _ in parse_qsl(parsed.query)):
        raise RoutedIngestionConfigurationError(
            "source.original_uri must not contain secret query parameters"
        )
    return value


def _assert_secret_free(value: object, *, path: str = "artifact") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key)
            if _is_sensitive_key(key_text) and item not in (None, ""):
                raise RoutedIngestionArtifactError(
                    f"refusing to serialize secret-like field {path}.{key_text}"
                )
            _assert_secret_free(item, path=f"{path}.{key_text}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_secret_free(item, path=f"{path}[{index}]")
        return
    if isinstance(value, str) and "://" in value:
        try:
            parsed = urlsplit(value)
        except ValueError:
            return
        if parsed.username is not None or parsed.password is not None:
            raise RoutedIngestionArtifactError(
                f"refusing to serialize URI credentials at {path}"
            )
        if any(_is_sensitive_key(key) for key, _ in parse_qsl(parsed.query)):
            raise RoutedIngestionArtifactError(
                f"refusing to serialize secret URI query at {path}"
            )


def _safe_output_root(path: Path) -> Path:
    expanded = path.expanduser()
    try:
        if expanded.exists() or expanded.is_symlink():
            if expanded.is_symlink():
                raise RoutedIngestionArtifactSecurityError(
                    f"output root must not be a symlink: {expanded}"
                )
            resolved = expanded.resolve(strict=True)
            if not resolved.is_dir():
                raise RoutedIngestionArtifactSecurityError(
                    f"output root is not a directory: {resolved}"
                )
        else:
            expanded.mkdir(parents=True, exist_ok=False)
            resolved = expanded.resolve(strict=True)
        if resolved.is_symlink():  # defensive; resolve() normally eliminates it
            raise RoutedIngestionArtifactSecurityError(
                f"output root must not be a symlink: {resolved}"
            )
    except RoutedIngestionArtifactSecurityError:
        raise
    except OSError as exc:
        raise RoutedIngestionArtifactSecurityError(
            f"could not create a safe output root: {expanded}"
        ) from exc
    return resolved


def _artifact_path(root: Path, filename: str) -> Path:
    if Path(filename).name != filename or filename in {"", ".", ".."}:
        raise RoutedIngestionArtifactSecurityError("artifact filename may not escape root")
    candidate = root / filename
    if candidate.parent.resolve(strict=True) != root:
        raise RoutedIngestionArtifactSecurityError("artifact path escapes output root")
    if candidate.is_symlink():
        raise RoutedIngestionArtifactSecurityError(
            f"artifact must not be a symlink: {candidate}"
        )
    return candidate


def _artifact_exists(path: Path, root: Path) -> bool:
    _validate_artifact_location(path, root)
    if not path.exists():
        return False
    try:
        mode = path.stat(follow_symlinks=False).st_mode
    except OSError as exc:
        raise RoutedIngestionArtifactSecurityError(
            f"could not inspect artifact: {path}"
        ) from exc
    if not stat.S_ISREG(mode):
        raise RoutedIngestionArtifactSecurityError(
            f"artifact is not a regular file: {path}"
        )
    return True


def _validate_artifact_location(path: Path, root: Path) -> None:
    if path.parent != root or path.name not in {
        ROUTED_AUDIT_FILE,
        LAST_ATTEMPT_AUDIT_FILE,
        CANONICAL_ARTIFACT_FILE,
    }:
        raise RoutedIngestionArtifactSecurityError(
            f"artifact path escapes or is not owned by routed ingestion: {path}"
        )
    if root.is_symlink() or path.is_symlink():
        raise RoutedIngestionArtifactSecurityError(
            f"symlinked artifact location is not allowed: {path}"
        )


def _read_json_object(path: Path, *, root: Path, max_bytes: int) -> dict[str, object]:
    _validate_artifact_location(path, root)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise RoutedIngestionArtifactSecurityError(
            f"could not safely open artifact: {path}"
        ) from exc
    try:
        mode = os.fstat(descriptor).st_mode
        if not stat.S_ISREG(mode):
            raise RoutedIngestionArtifactSecurityError(
                f"artifact is not a regular file: {path}"
            )
        chunks: list[bytes] = []
        remaining = max_bytes + 1
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        encoded = b"".join(chunks)
    finally:
        os.close(descriptor)
    if len(encoded) > max_bytes:
        raise RoutedIngestionArtifactError(
            f"artifact exceeds configured byte limit: {path.name}"
        )
    try:
        payload = json.loads(encoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RoutedIngestionArtifactError(
            f"artifact is not valid UTF-8 JSON: {path.name}"
        ) from exc
    if not isinstance(payload, dict):
        raise RoutedIngestionArtifactError(
            f"artifact root must be a JSON object: {path.name}"
        )
    return cast(dict[str, object], payload)


def _atomic_write_json(
    path: Path,
    payload: Mapping[str, object],
    *,
    root: Path,
    max_bytes: int,
) -> None:
    _validate_artifact_location(path, root)
    if path.exists() and not _artifact_exists(path, root):  # pragma: no cover
        raise RoutedIngestionArtifactSecurityError(f"unsafe artifact target: {path}")
    encoded = _encode_json_artifact(payload, path=path, max_bytes=max_bytes)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=root,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(encoded)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
        _fsync_directory(root)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _atomic_replace_json_pair(
    items: tuple[tuple[Path, Mapping[str, object], int], ...],
    *,
    root: Path,
) -> None:
    """Stage and replace a small artifact set, rolling back normal failures.

    POSIX has no multi-file rename primitive.  Both new files are therefore
    fully written and fsynced first, existing files are moved to same-directory
    backups, and every promoted target is rolled back if a later operation
    fails.  This prevents a failed forced parse or ordinary I/O exception from
    destroying the last ready audit/canonical pair.
    """

    if not items:
        raise RoutedIngestionArtifactError("artifact replacement set is empty")
    paths = tuple(path for path, _, _ in items)
    if len(paths) != len(set(paths)):
        raise RoutedIngestionArtifactError(
            "artifact replacement set contains a duplicate path"
        )
    encoded_by_path: dict[Path, bytes] = {}
    for path, payload, max_bytes in items:
        _validate_artifact_location(path, root)
        if path.exists():
            _artifact_exists(path, root)
        encoded_by_path[path] = _encode_json_artifact(
            payload,
            path=path,
            max_bytes=max_bytes,
        )

    stages: dict[Path, Path] = {}
    backups: dict[Path, Path] = {}
    promoted: set[Path] = set()
    retain_backups = False
    try:
        for path in paths:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=f".{path.name}.",
                suffix=".stage",
                dir=root,
                delete=False,
            ) as temporary:
                stage = Path(temporary.name)
                stages[path] = stage
                temporary.write(encoded_by_path[path])
                temporary.flush()
                os.fsync(temporary.fileno())
        _fsync_directory(root)

        for path in paths:
            if not path.exists():
                continue
            descriptor, backup_name = tempfile.mkstemp(
                prefix=f".{path.name}.",
                suffix=".previous",
                dir=root,
            )
            os.close(descriptor)
            backup = Path(backup_name)
            backup.unlink()
            os.replace(path, backup)
            backups[path] = backup

        for path in paths:
            os.replace(stages[path], path)
            promoted.add(path)
        _fsync_directory(root)
    except BaseException as exc:
        rollback_errors: list[str] = []
        for path in promoted:
            try:
                path.unlink(missing_ok=True)
            except OSError as rollback_exc:
                rollback_errors.append(f"remove {path.name}: {rollback_exc}")
        for path, backup in backups.items():
            try:
                if path.exists():
                    path.unlink()
                if backup.exists():
                    os.replace(backup, path)
            except OSError as rollback_exc:
                rollback_errors.append(f"restore {path.name}: {rollback_exc}")
        try:
            _fsync_directory(root)
        except RoutedIngestionArtifactError as rollback_exc:
            rollback_errors.append(str(rollback_exc))
        if rollback_errors:
            retain_backups = True
            raise RoutedIngestionArtifactError(
                "artifact pair replacement failed and rollback was incomplete: "
                + "; ".join(rollback_errors)
            ) from exc
        if isinstance(exc, Exception):
            raise RoutedIngestionArtifactError(
                f"artifact pair replacement failed: {exc}"
            ) from exc
        raise
    finally:
        cleanup = list(stages.values())
        if not retain_backups:
            cleanup.extend(backups.values())
        for temporary in cleanup:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
    _fsync_directory(root)


def _encode_json_artifact(
    payload: Mapping[str, object],
    *,
    path: Path,
    max_bytes: int,
) -> bytes:
    encoded = (
        json.dumps(
            json_ready(payload),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    if len(encoded) > max_bytes:
        raise RoutedIngestionArtifactError(
            f"artifact exceeds configured byte limit: {path.name}"
        )
    return encoded


def _durable_unlink(path: Path, root: Path) -> None:
    _validate_artifact_location(path, root)
    if path.is_symlink():
        raise RoutedIngestionArtifactSecurityError(
            f"refusing to unlink symlinked artifact: {path}"
        )
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        raise RoutedIngestionArtifactSecurityError(
            f"could not remove stale artifact: {path}"
        ) from exc
    _fsync_directory(root)


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError as exc:
        raise RoutedIngestionArtifactError(
            f"could not durably synchronize artifact directory: {path}"
        ) from exc


__all__ = [
    "CANONICAL_ARTIFACT_FILE",
    "CANONICAL_ARTIFACT_ROLE",
    "LAST_ATTEMPT_AUDIT_FILE",
    "ROUTED_AUDIT_FILE",
    "ROUTED_INGESTION_SCHEMA_VERSION",
    "RoutedBookRAGAuditValidator",
    "RoutedCanonicalIngestionPayload",
    "RoutedCanonicalIngestionProvider",
    "RoutedIngestionArtifactError",
    "RoutedIngestionArtifactSecurityError",
    "RoutedIngestionConfigurationError",
    "RoutedIngestionError",
    "RoutedIngestionPageLimitError",
    "RoutedIngestionPolicy",
    "RoutedIngestionSourceBindingError",
]
