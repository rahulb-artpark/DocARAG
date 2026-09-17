"""Bounded canonical-lineage hydration for BookRAG retrieval evidence.

The retrieval index is a candidate generator, not the source of truth.  This
module resolves each candidate through the persisted BookRAG tree and then
through the canonical evidence artifact before it can enter an answer prompt.
It deliberately depends only on the standard library and DocARAG's domain
contracts; importing it never imports model, vector-store, or BookRAG runtime
dependencies.

Hydration is O(B + R + S): B canonical blocks are validated and indexed once,
R retrieved tree records are resolved in one batch, and S canonical source
references are checked with O(1) revision lookups.  Explicit limits bound each
of those terms.

This boundary validates asset references (URI, declared digest, media type,
and asset identity) as part of each canonical block, but deliberately does not
open or hash image/crop bytes.  ``asset_material_status=reference_only`` makes
that limitation machine-visible.  The opt-in :mod:`document_services.evidence.assets`
resolver can produce a bounded verified read snapshot; hydration does not call
it automatically because a later reopen of the pathname would reintroduce a
time-of-check/time-of-use gap.  A strict VLM path must consume verified bytes
or an immutable content-addressed copy before claiming byte-level integrity.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
import json
import os
from pathlib import Path
import re
import stat as stat_module
from typing import NoReturn

from document_services.evidence.projections.bookrag import project_canonical_blocks
from document_services.evidence.artifacts import (
    CANONICAL_CONTRACT_SCHEMA_VERSION,
    CANONICAL_EVIDENCE_SCHEMA_VERSION,
)
from document_services.domain import AssetRef, CanonicalBlock, DocumentRef, SourceGeometry
from document_services.domain.base import (
    ContractValidationError,
    content_sha256,
    require_mapping,
    stable_id,
)
from document_services.evidence.identity import (
    caption_footnote_text,
    derived_evidence_id,
    derived_identity_payload,
    derived_output_identity,
)


CITATION_SCHEMA_VERSION = "docarag.source-citation.v1"
ASSET_MATERIAL_REFERENCE_ONLY = "reference_only"
ASSET_MATERIAL_NOT_APPLICABLE = "not_applicable"


class EvidenceHydrationError(ValueError):
    """Retrieved lineage cannot be validated against tree and canonical catalog."""


def _fail(message: str) -> NoReturn:
    raise EvidenceHydrationError(message)


# Node types whose evidence lives in the projection's "text" channel, and
# those whose evidence lives in its image caption/footnote channels. Both sets
# mirror bookrag_projection's _NODE_TYPE_BY_BLOCK / _VISUAL_NODE_TYPE_BY_BLOCK:
# a type added there without being added here hydrates as "unsupported node
# type" rather than degrading quietly.
_TEXT_NODE_TYPES = frozenset(
    {
        "text",
        "list",
        "code",
        "algorithm",
        "abstract",
        "reference",
        # Page furniture. It gained its own projected node type so a page number
        # or a "Scanned with CamScanner" footer would stop being embedded and
        # summarised as document prose, but was never added here -- and its
        # evidence still lives in the same "text" channel as any other of these.
        # The gap stayed hidden because the only path that hydrates a whole
        # section was refusing on its evidence budget before it ever reached
        # hydration; a section containing two footers and two asides then failed
        # with "unsupported node type" the moment that budget was corrected.
        "header",
        "footer",
        "aside",
        "discarded",
    }
)
_VISUAL_NODE_TYPES = frozenset({"image", "chart", "diagram"})
# Types that may carry a materialized crop whose path must match the
# projection. Mirrors vdb_index._CROP_BEARING_NODE_TYPES.
_CROP_BEARING_NODE_TYPES = _VISUAL_NODE_TYPES | {"table", "equation", "list"}


def _non_empty_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(f"{field_name} must be a non-empty string")
    return value


def _positive_limit(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        _fail(f"{field_name} must be a positive integer")
    return value


def _json_value(value: object) -> object:
    """Normalize enums/tuples so trusted and retrieved records compare exactly."""

    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {
            str(key): _json_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return [_json_value(item) for item in value]
    return value


def _same(left: object, right: object) -> bool:
    return _json_value(left) == _json_value(right)


def _unique_strings(value: object, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        _fail(f"{field_name} must be an array")
    result: list[str] = []
    seen: set[str] = set()
    for index, item in enumerate(value):
        text = _non_empty_text(item, f"{field_name}[{index}]")
        if text in seen:
            _fail(f"{field_name} contains a duplicate identity")
        seen.add(text)
        result.append(text)
    return tuple(result)


def _ordered_unique(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def _semantic_substructure_union(
    blocks: Sequence[CanonicalBlock],
) -> list[dict[str, object]]:
    """Return separate source-bound graphs in stable identity order."""

    by_id: dict[str, dict[str, object]] = {}
    for block in blocks:
        for structure in block.semantic_substructures:
            payload = structure.to_dict()
            previous = by_id.get(structure.substructure_id)
            if previous is not None and not _same(previous, payload):
                _fail(
                    "one semantic substructure identity resolves to conflicting "
                    "canonical payloads"
                )
            by_id[structure.substructure_id] = payload
    return [by_id[substructure_id] for substructure_id in sorted(by_id)]


def _semantic_substructure_refs(
    block: CanonicalBlock,
) -> list[dict[str, object]]:
    """Return bounded semantic identity/digest/lineage receipts for one source."""

    return [
        {
            "substructure_id": structure.substructure_id,
            "semantic_sha256": structure.semantic_sha256,
            "comparison_sha256": structure.comparison_sha256,
            "source_observation_ids": list(structure.source_observation_ids),
            "source_substructure_ids": list(structure.source_substructure_ids),
        }
        for structure in block.semantic_substructures
    ]


def _load_json_object(path: Path, *, max_bytes: int) -> dict[str, object]:
    """Load a bounded UTF-8 JSON object while rejecting duplicate keys/NaN."""

    file_descriptor: int | None = None
    try:
        if path.is_symlink():
            _fail("canonical evidence artifact must not be a symbolic link")
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        file_descriptor = os.open(path, flags)
        before = os.fstat(file_descriptor)
    except OSError as exc:
        if file_descriptor is not None:
            os.close(file_descriptor)
        raise EvidenceHydrationError(
            f"canonical evidence artifact is unavailable: {path}"
        ) from exc
    try:
        if not stat_module.S_ISREG(before.st_mode):
            _fail(f"canonical evidence artifact is not a regular file: {path}")
        if before.st_size > max_bytes:
            _fail(
                "canonical evidence artifact exceeds max_artifact_bytes "
                f"({before.st_size} > {max_bytes})"
            )
        chunks: list[bytes] = []
        remaining = max_bytes + 1
        while remaining:
            chunk = os.read(file_descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        if len(raw) > max_bytes:
            _fail("canonical evidence artifact grew beyond max_artifact_bytes")
        after = os.fstat(file_descriptor)
        if len(raw) != after.st_size:
            _fail("canonical evidence artifact could not be read completely")
        stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if any(getattr(before, field) != getattr(after, field) for field in stable_fields):
            _fail("canonical evidence artifact changed while it was being read")
        current = os.stat(path, follow_symlinks=False)
        if (current.st_dev, current.st_ino) != (after.st_dev, after.st_ino):
            _fail("canonical evidence artifact path was replaced while it was being read")

        def object_hook(pairs: list[tuple[str, object]]) -> dict[str, object]:
            result: dict[str, object] = {}
            for key, value in pairs:
                if key in result:
                    _fail(f"canonical evidence JSON contains duplicate key: {key}")
                result[key] = value
            return result

        def reject_constant(value: str) -> NoReturn:
            _fail(f"canonical evidence JSON contains non-finite value: {value}")

        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=object_hook,
            parse_constant=reject_constant,
        )
    except EvidenceHydrationError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, OSError) as exc:
        raise EvidenceHydrationError(
            f"canonical evidence artifact is not valid UTF-8 JSON: {path}"
        ) from exc
    finally:
        if file_descriptor is not None:
            os.close(file_descriptor)
    if not isinstance(payload, dict):
        _fail("canonical evidence artifact root must be an object")
    return payload


_KNOWN_ARTIFACT_FIELDS = frozenset(
    {
        # Shared fields.
        "artifact_schema_version",
        "contract_schema_version",
        "document",
        "parse_request_id",
        "block_count",
        "canonical_blocks",
        # Legacy MinerU canonical binding.
        "parse_result_sha256",
        "parser_manifest",
        # Routed-ingestion canonical binding.
        "artifact_role",
        "producer_schema_version",
        "configuration_sha256",
        "audit_sha256",
        "projection_id",
        "routed_parsing_result_id",
        "document_version_id",
        "source_sha256",
        "canonical_blocks_sha256",
    }
)


class CanonicalEvidenceCatalog:
    """One validated canonical artifact with constant-time identity lookup."""

    __slots__ = ("artifact_path", "document", "blocks", "_by_block", "_by_revision")

    def __init__(
        self,
        *,
        artifact_path: Path,
        document: DocumentRef,
        blocks: tuple[CanonicalBlock, ...],
    ) -> None:
        self.artifact_path = artifact_path
        self.document = document
        self.blocks = blocks
        self._by_block = {block.block_id: block for block in blocks}
        self._by_revision = {block.revision_id: block for block in blocks}

    @classmethod
    def load(
        cls,
        artifact_path: str | os.PathLike[str],
        *,
        max_artifact_bytes: int = 64 * 1024 * 1024,
        max_canonical_blocks: int = 250_000,
    ) -> "CanonicalEvidenceCatalog":
        max_artifact_bytes = _positive_limit(
            max_artifact_bytes, "max_artifact_bytes"
        )
        max_canonical_blocks = _positive_limit(
            max_canonical_blocks, "max_canonical_blocks"
        )
        try:
            unresolved_path = Path(os.fspath(artifact_path)).expanduser()
            if unresolved_path.is_symlink():
                _fail("canonical evidence artifact must not be a symbolic link")
            path = unresolved_path.resolve(strict=True)
        except (OSError, TypeError, ValueError) as exc:
            raise EvidenceHydrationError(
                "canonical evidence artifact path is invalid or unavailable"
            ) from exc
        payload = _load_json_object(path, max_bytes=max_artifact_bytes)
        unknown = sorted(set(payload).difference(_KNOWN_ARTIFACT_FIELDS))
        if unknown:
            _fail(
                "canonical evidence artifact contains unsupported fields: "
                + ", ".join(unknown)
            )
        if payload.get("artifact_schema_version") != CANONICAL_EVIDENCE_SCHEMA_VERSION:
            _fail("canonical evidence artifact schema version is unsupported")
        if payload.get("contract_schema_version") != CANONICAL_CONTRACT_SCHEMA_VERSION:
            _fail("canonical evidence contract schema version is unsupported")

        try:
            document = DocumentRef.from_dict(
                require_mapping(payload.get("document"), "document")
            )
        except (ContractValidationError, TypeError, ValueError) as exc:
            raise EvidenceHydrationError(
                f"canonical evidence document is invalid: {exc}"
            ) from exc

        raw_blocks = payload.get("canonical_blocks")
        if not isinstance(raw_blocks, Sequence) or isinstance(
            raw_blocks, (str, bytes, bytearray)
        ):
            _fail("canonical_blocks must be an array")
        count = payload.get("block_count")
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            _fail("canonical block_count must be a positive integer")
        if count != len(raw_blocks):
            _fail("canonical block_count does not match canonical_blocks")
        if count > max_canonical_blocks:
            _fail(
                "canonical evidence exceeds max_canonical_blocks "
                f"({count} > {max_canonical_blocks})"
            )

        blocks: list[CanonicalBlock] = []
        try:
            for index, raw_block in enumerate(raw_blocks):
                block = CanonicalBlock.from_dict(
                    require_mapping(raw_block, f"canonical_blocks[{index}]")
                )
                if block.document_version_id != document.document_version_id:
                    _fail(
                        f"canonical_blocks[{index}] belongs to another document version"
                    )
                blocks.append(block)
        except EvidenceHydrationError:
            raise
        except (ContractValidationError, TypeError, ValueError) as exc:
            raise EvidenceHydrationError(
                f"canonical evidence block is invalid: {exc}"
            ) from exc

        block_ids = [block.block_id for block in blocks]
        revision_ids = [block.revision_id for block in blocks]
        if len(set(block_ids)) != len(block_ids):
            _fail("canonical artifact contains multiple current revisions for a block")
        if len(set(revision_ids)) != len(revision_ids):
            _fail("canonical artifact contains duplicate revision identities")

        for field_name, expected in (
            ("document_version_id", document.document_version_id),
            ("source_sha256", document.content_sha256),
        ):
            actual = payload.get(field_name)
            if actual is not None and actual != expected:
                _fail(f"canonical artifact {field_name} binding mismatch")
        declared_digest = payload.get("canonical_blocks_sha256")
        if declared_digest is not None:
            actual_digest = content_sha256([block.to_dict() for block in blocks])
            if declared_digest != actual_digest:
                _fail("canonical artifact canonical_blocks_sha256 mismatch")

        return cls(
            artifact_path=path,
            document=document,
            blocks=tuple(blocks),
        )

    def block_for_revision(self, revision_id: str) -> CanonicalBlock:
        try:
            return self._by_revision[revision_id]
        except KeyError as exc:
            raise EvidenceHydrationError(
                f"canonical revision is absent from source artifact: {revision_id}"
            ) from exc

    def validate_pair(self, block_id: str, revision_id: str) -> CanonicalBlock:
        block = self.block_for_revision(revision_id)
        if block.block_id != block_id or self._by_block.get(block_id) is not block:
            _fail("canonical block and revision identities do not refer to one block")
        return block


@dataclass(frozen=True, slots=True)
class CitationLocation:
    """One validated canonical block location used by an answer evidence item."""

    canonical_block_id: str
    canonical_revision_id: str
    block_type: str
    semantic_subtype: str | None
    page_index: int
    page_id: str
    geometry: SourceGeometry | None
    representation_ids: tuple[str, ...]
    assets: tuple[AssetRef, ...]

    @classmethod
    def from_block(cls, block: CanonicalBlock) -> "CitationLocation":
        return cls(
            canonical_block_id=block.block_id,
            canonical_revision_id=block.revision_id,
            block_type=block.block_type.value,
            semantic_subtype=block.semantic_subtype,
            page_index=block.page_index,
            page_id=block.page_id,
            geometry=block.geometry,
            representation_ids=tuple(
                representation.representation_id
                for representation in block.representations
            ),
            assets=tuple(block.assets),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "canonical_block_id": self.canonical_block_id,
            "canonical_revision_id": self.canonical_revision_id,
            "block_type": self.block_type,
            "semantic_subtype": self.semantic_subtype,
            "page_index": self.page_index,
            "page_id": self.page_id,
            "geometry": self.geometry.to_dict() if self.geometry else None,
            "representation_ids": list(self.representation_ids),
            "assets": [asset.to_dict() for asset in self.assets],
            "asset_material_status": (
                ASSET_MATERIAL_REFERENCE_ONLY
                if self.assets
                else ASSET_MATERIAL_NOT_APPLICABLE
            ),
        }


@dataclass(frozen=True, slots=True)
class SourceCitation:
    """Stable citation contract emitted after canonical-lineage hydration.

    This contract identifies canonical locations supplied to generation. Asset
    records remain references until a separate material resolver verifies their
    bytes. It also does not claim that a later language-model response cited the
    supplied evidence correctly.
    """

    evidence_id: str
    document_version_id: str
    source_sha256: str
    source_uri: str
    locations: tuple[CitationLocation, ...]
    transformations: tuple[str, ...] = ()
    citation_id: str = field(init=False)

    def __post_init__(self) -> None:
        _non_empty_text(self.evidence_id, "citation.evidence_id")
        _non_empty_text(
            self.document_version_id, "citation.document_version_id"
        )
        if (
            not isinstance(self.source_sha256, str)
            or len(self.source_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.source_sha256)
        ):
            _fail("citation.source_sha256 must be a lowercase SHA-256 digest")
        _non_empty_text(self.source_uri, "citation.source_uri")
        if not self.locations:
            _fail("citation requires at least one canonical source location")
        if any(not isinstance(location, CitationLocation) for location in self.locations):
            _fail("citation.locations must contain CitationLocation values")
        transformations = _unique_strings(
            self.transformations, "citation.transformations"
        )
        object.__setattr__(self, "transformations", transformations)
        identity = {
            "schema_version": CITATION_SCHEMA_VERSION,
            "evidence_id": self.evidence_id,
            "document_version_id": self.document_version_id,
            "source_sha256": self.source_sha256,
            "source_uri": self.source_uri,
            "canonical_revision_ids": [
                location.canonical_revision_id for location in self.locations
            ],
            "transformations": list(self.transformations),
        }
        object.__setattr__(self, "citation_id", stable_id("citation", identity))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": CITATION_SCHEMA_VERSION,
            "citation_id": self.citation_id,
            "evidence_id": self.evidence_id,
            "document_version_id": self.document_version_id,
            "source_sha256": self.source_sha256,
            "source_uri": self.source_uri,
            "page_indices": [location.page_index for location in self.locations],
            "page_ids": [location.page_id for location in self.locations],
            "canonical_block_ids": [
                location.canonical_block_id for location in self.locations
            ],
            "canonical_revision_ids": [
                location.canonical_revision_id for location in self.locations
            ],
            "transformations": list(self.transformations),
            "locations": [location.to_dict() for location in self.locations],
            "asset_material_status": (
                ASSET_MATERIAL_REFERENCE_ONLY
                if any(location.assets for location in self.locations)
                else ASSET_MATERIAL_NOT_APPLICABLE
            ),
        }


_AUTHORITATIVE_FIELDS = frozenset(
    {
        "index_id",
        "type",
        "content",
        "page",
        "page_id",
        "document_version_id",
        "canonical_block_id",
        "canonical_revision_id",
        "observation_id",
        "derived_evidence_id",
        "source_canonical_block_ids",
        "source_canonical_revision_ids",
        "source_observation_ids",
        "source_evidence_refs",
        "canonical_representation_ids",
        "canonical_asset_ids",
        "selected_representation_ids",
        "selected_asset_ids",
        "canonical_relations",
        "semantic_substructures",
        "transformations",
        "geometry",
        "source_geometry",
        "semantic_subtype",
        "parser_provenance",
        "quality_status",
        "reconciliation_method",
        "source_uri",
        "source_sha256",
        "snapshot_path",
        "img_path",
        "table_body",
        "caption",
        "footnote",
    }
)


class CanonicalEvidenceHydrator:
    """Resolve retrieval candidates against authoritative tree and canonical data."""

    __slots__ = (
        "catalog",
        "_node_resolver",
        "max_evidence",
        "max_sources_per_evidence",
        "max_total_sources",
    )

    def __init__(
        self,
        catalog: CanonicalEvidenceCatalog,
        *,
        node_resolver: Callable[[Sequence[int]], Sequence[Mapping[str, object]]],
        expected_document_version_id: str | None = None,
        expected_source_sha256: str | None = None,
        expected_source_uri: str | None = None,
        max_evidence: int = 64,
        max_sources_per_evidence: int = 64,
        max_total_sources: int = 512,
    ) -> None:
        if not isinstance(catalog, CanonicalEvidenceCatalog):
            _fail("catalog must be a CanonicalEvidenceCatalog")
        if not callable(node_resolver):
            _fail("node_resolver must be callable")
        expected = (
            ("document_version_id", expected_document_version_id, catalog.document.document_version_id),
            ("source_sha256", expected_source_sha256, catalog.document.content_sha256),
            ("source_uri", expected_source_uri, catalog.document.source_uri),
        )
        for field_name, actual, artifact_value in expected:
            if actual is not None and actual != artifact_value:
                _fail(f"tree {field_name} differs from canonical artifact")
        self.catalog = catalog
        self._node_resolver = node_resolver
        self.max_evidence = _positive_limit(max_evidence, "max_evidence")
        self.max_sources_per_evidence = _positive_limit(
            max_sources_per_evidence, "max_sources_per_evidence"
        )
        self.max_total_sources = _positive_limit(
            max_total_sources, "max_total_sources"
        )

    @classmethod
    def from_artifact(
        cls,
        artifact_path: str | os.PathLike[str],
        *,
        node_resolver: Callable[[Sequence[int]], Sequence[Mapping[str, object]]],
        expected_document_version_id: str | None = None,
        expected_source_sha256: str | None = None,
        expected_source_uri: str | None = None,
        max_artifact_bytes: int = 64 * 1024 * 1024,
        max_canonical_blocks: int = 250_000,
        max_evidence: int = 64,
        max_sources_per_evidence: int = 64,
        max_total_sources: int = 512,
    ) -> "CanonicalEvidenceHydrator":
        catalog = CanonicalEvidenceCatalog.load(
            artifact_path,
            max_artifact_bytes=max_artifact_bytes,
            max_canonical_blocks=max_canonical_blocks,
        )
        return cls(
            catalog,
            node_resolver=node_resolver,
            expected_document_version_id=expected_document_version_id,
            expected_source_sha256=expected_source_sha256,
            expected_source_uri=expected_source_uri,
            max_evidence=max_evidence,
            max_sources_per_evidence=max_sources_per_evidence,
            max_total_sources=max_total_sources,
        )

    def hydrate(
        self, retrieved_nodes: Sequence[Mapping[str, object]]
    ) -> list[dict[str, object]]:
        """Return deep-copied, canonical-lineage-validated records in input order."""

        if not isinstance(retrieved_nodes, Sequence) or isinstance(
            retrieved_nodes, (str, bytes, bytearray)
        ):
            _fail("retrieved_nodes must be an array")
        if len(retrieved_nodes) > self.max_evidence:
            _fail(
                "retrieval exceeds max_evidence "
                f"({len(retrieved_nodes)} > {self.max_evidence})"
            )
        if not retrieved_nodes:
            return []

        requested_ids: list[int] = []
        retrieval_by_id: dict[int, Mapping[str, object]] = {}
        for position, node in enumerate(retrieved_nodes):
            if not isinstance(node, Mapping):
                _fail(f"retrieved_nodes[{position}] must be an object")
            node_id = node.get("index_id")
            if isinstance(node_id, bool) or not isinstance(node_id, int) or node_id < 1:
                _fail(f"retrieved_nodes[{position}].index_id must be a positive integer")
            if node_id in retrieval_by_id:
                _fail(f"retrieval contains duplicate tree node id: {node_id}")
            if node.get("document_version_id") != self.catalog.document.document_version_id:
                _fail(
                    f"retrieved_nodes[{position}].document_version_id differs from "
                    "the canonical artifact"
                )
            if node.get("source_sha256") != self.catalog.document.content_sha256:
                _fail(
                    f"retrieved_nodes[{position}].source_sha256 differs from the "
                    "canonical artifact"
                )
            candidate_block_id = node.get("canonical_block_id")
            candidate_revision_id = node.get("canonical_revision_id")
            candidate_derived_id = node.get("derived_evidence_id")
            direct_identity = candidate_block_id is not None or candidate_revision_id is not None
            derived_identity = candidate_derived_id is not None
            if direct_identity == derived_identity:
                _fail(
                    f"retrieved_nodes[{position}] must have exactly one direct or "
                    "derived canonical identity"
                )
            if direct_identity:
                _non_empty_text(
                    candidate_block_id,
                    f"retrieved_nodes[{position}].canonical_block_id",
                )
                _non_empty_text(
                    candidate_revision_id,
                    f"retrieved_nodes[{position}].canonical_revision_id",
                )
            elif not isinstance(candidate_derived_id, str) or re.fullmatch(
                r"derived:[0-9a-f]{64}", candidate_derived_id
            ) is None:
                _fail(
                    f"retrieved_nodes[{position}].derived_evidence_id must use "
                    "derived:<64 lowercase hex>"
                )
            requested_ids.append(node_id)
            retrieval_by_id[node_id] = node

        try:
            resolved = self._node_resolver(tuple(requested_ids))
        except EvidenceHydrationError:
            raise
        except Exception as exc:
            raise EvidenceHydrationError(
                "authoritative tree node resolution failed"
            ) from exc
        if not isinstance(resolved, Sequence) or isinstance(
            resolved, (str, bytes, bytearray)
        ):
            _fail("node_resolver must return an array")
        tree_by_id: dict[int, Mapping[str, object]] = {}
        for position, node in enumerate(resolved):
            if not isinstance(node, Mapping):
                _fail(f"node_resolver result {position} must be an object")
            node_id = node.get("index_id")
            if isinstance(node_id, bool) or not isinstance(node_id, int):
                _fail(f"node_resolver result {position} has an invalid index_id")
            if node_id in tree_by_id:
                _fail(f"node_resolver returned duplicate tree node id: {node_id}")
            tree_by_id[node_id] = node
        if set(tree_by_id) != set(requested_ids):
            missing = sorted(set(requested_ids).difference(tree_by_id))
            unexpected = sorted(set(tree_by_id).difference(requested_ids))
            _fail(
                "node_resolver did not return exactly the requested tree nodes "
                f"(missing={missing}, unexpected={unexpected})"
            )

        hydrated: list[dict[str, object]] = []
        total_sources = 0
        for node_id in requested_ids:
            candidate = retrieval_by_id[node_id]
            tree_node = tree_by_id[node_id]
            self._validate_candidate(candidate, tree_node)
            blocks, transformations, evidence_id = self._resolve_lineage(tree_node)
            semantic_selection = self._resolve_semantic_selection(candidate, blocks)
            total_sources += len(blocks)
            if total_sources > self.max_total_sources:
                _fail(
                    "hydration exceeds max_total_sources "
                    f"({total_sources} > {self.max_total_sources})"
                )
            citation = SourceCitation(
                evidence_id=evidence_id,
                document_version_id=self.catalog.document.document_version_id,
                source_sha256=self.catalog.document.content_sha256,
                source_uri=self.catalog.document.source_uri,
                locations=tuple(CitationLocation.from_block(block) for block in blocks),
                transformations=transformations,
            )

            # The tree record is authoritative.  Retrieval-only ranking metadata
            # may be retained, but it can never replace evidence-bearing fields.
            result = deepcopy(dict(tree_node))
            for key, value in candidate.items():
                if key not in _AUTHORITATIVE_FIELDS:
                    result[key] = deepcopy(value)
            result["evidence_id"] = evidence_id
            result["citation_id"] = citation.citation_id
            result["citation"] = citation.to_dict()
            result["canonical_sources"] = [
                deepcopy(block.to_dict()) for block in blocks
            ]
            result["canonical_source_count"] = len(blocks)
            if semantic_selection is not None:
                result.update(semantic_selection)
            hydrated.append(result)
        return hydrated

    def _validate_candidate(
        self,
        candidate: Mapping[str, object],
        tree_node: Mapping[str, object],
    ) -> None:
        for field_name in _AUTHORITATIVE_FIELDS:
            if field_name in candidate and not _same(
                candidate.get(field_name), tree_node.get(field_name)
            ):
                _fail(
                    f"retrieved tree node {candidate.get('index_id')} has a "
                    f"mismatched authoritative field: {field_name}"
                )
        document = self.catalog.document
        expected_fields = (
            ("document_version_id", document.document_version_id),
            ("source_sha256", document.content_sha256),
            ("source_uri", document.source_uri),
        )
        for field_name, expected in expected_fields:
            if tree_node.get(field_name) != expected:
                _fail(
                    f"tree node {tree_node.get('index_id')} {field_name} "
                    "differs from canonical artifact"
                )
        page = tree_node.get("page")
        if isinstance(page, bool) or not isinstance(page, int) or page < 0:
            _fail(f"tree node {tree_node.get('index_id')} has an invalid page index")

    def _resolve_semantic_selection(
        self,
        candidate: Mapping[str, object],
        blocks: Sequence[CanonicalBlock],
    ) -> dict[str, object] | None:
        """Validate a vector-selected cell/symbol/diagram node against source truth."""

        substructure_id = candidate.get("semantic_substructure_id")
        element_id = candidate.get("semantic_element_id")
        if substructure_id is None and element_id is None:
            return None
        if not isinstance(substructure_id, str) or not substructure_id:
            _fail("semantic element retrieval requires semantic_substructure_id")
        if not isinstance(element_id, str) or not element_id:
            _fail("semantic element retrieval requires semantic_element_id")

        matches = []
        for block in blocks:
            for structure in block.semantic_substructures:
                if structure.substructure_id != substructure_id:
                    continue
                for element in structure.elements:
                    if element.element_id == element_id:
                        matches.append((block, structure, element))
        if len(matches) != 1:
            _fail(
                "retrieved semantic element does not resolve uniquely in canonical "
                f"sources: {substructure_id}/{element_id}"
            )
        block, structure, element = matches[0]

        expected_fields = (
            ("semantic_structure_kind", structure.kind.value),
            ("semantic_structure_availability", structure.availability.value),
            ("semantic_structure_producer_kind", structure.producer_kind.value),
            ("semantic_sha256", structure.semantic_sha256),
            ("semantic_comparison_sha256", structure.comparison_sha256),
            ("semantic_element_kind", element.kind.value),
            ("semantic_element_local_key", element.local_key),
            ("semantic_element_content_sha256", content_sha256(element.content)),
            ("semantic_element_role", element.role),
            ("semantic_element_row_index", element.row_index),
            ("semantic_element_column_index", element.column_index),
            ("semantic_element_row_span", element.row_span),
            ("semantic_element_column_span", element.column_span),
            ("semantic_element_confidence", element.confidence),
            (
                "semantic_element_source_selector_count",
                len(element.source_selectors),
            ),
        )
        for field_name, expected in expected_fields:
            if field_name in candidate and not _same(candidate[field_name], expected):
                _fail(
                    f"retrieved semantic selector has a mismatched field: {field_name}"
                )

        touching_relations = [
            relation.to_dict()
            for relation in structure.relations
            if element.element_id
            in (relation.source_element_id, relation.target_element_id)
        ]
        return {
            "selected_semantic_canonical_block_id": block.block_id,
            "selected_semantic_substructure": deepcopy(structure.to_dict()),
            "selected_semantic_element": deepcopy(element.to_dict()),
            "selected_semantic_relations": deepcopy(touching_relations),
        }

    def _resolve_lineage(
        self, tree_node: Mapping[str, object]
    ) -> tuple[tuple[CanonicalBlock, ...], tuple[str, ...], str]:
        direct_block_id = tree_node.get("canonical_block_id")
        direct_revision_id = tree_node.get("canonical_revision_id")
        derived_id = tree_node.get("derived_evidence_id")

        has_direct = direct_block_id is not None or direct_revision_id is not None
        has_derived = derived_id is not None
        if has_direct and has_derived:
            _fail("tree evidence cannot be both direct canonical and derived")
        if has_direct:
            block_id = _non_empty_text(direct_block_id, "canonical_block_id")
            revision_id = _non_empty_text(
                direct_revision_id, "canonical_revision_id"
            )
            block = self.catalog.validate_pair(block_id, revision_id)
            self._validate_direct_tree_node(tree_node, block)
            raw_transformations = tree_node.get("transformations")
            transformations = (
                _unique_strings(raw_transformations, "transformations")
                if raw_transformations is not None
                else ()
            )
            return (block,), transformations, revision_id
        if not has_derived:
            _fail(
                f"tree node {tree_node.get('index_id')} has no canonical or derived identity"
            )

        evidence_id = _non_empty_text(derived_id, "derived_evidence_id")
        refs = tree_node.get("source_evidence_refs")
        if not isinstance(refs, Sequence) or isinstance(
            refs, (str, bytes, bytearray)
        ) or not refs:
            _fail("derived evidence requires source_evidence_refs")
        if len(refs) > self.max_sources_per_evidence:
            _fail(
                "derived evidence exceeds max_sources_per_evidence "
                f"({len(refs)} > {self.max_sources_per_evidence})"
            )

        blocks: list[CanonicalBlock] = []
        for position, raw_ref in enumerate(refs):
            if not isinstance(raw_ref, Mapping):
                _fail(f"source_evidence_refs[{position}] must be an object")
            block_id = _non_empty_text(
                raw_ref.get("canonical_block_id"),
                f"source_evidence_refs[{position}].canonical_block_id",
            )
            revision_id = _non_empty_text(
                raw_ref.get("canonical_revision_id"),
                f"source_evidence_refs[{position}].canonical_revision_id",
            )
            block = self.catalog.validate_pair(block_id, revision_id)
            self._validate_source_ref(raw_ref, block, position=position)
            blocks.append(block)

        revision_ids = tuple(block.revision_id for block in blocks)
        block_ids = tuple(block.block_id for block in blocks)
        if len(set(revision_ids)) != len(revision_ids):
            _fail("derived evidence repeats a canonical revision")
        declared_revision_ids = _unique_strings(
            tree_node.get("source_canonical_revision_ids"),
            "source_canonical_revision_ids",
        )
        declared_block_ids = _unique_strings(
            tree_node.get("source_canonical_block_ids"),
            "source_canonical_block_ids",
        )
        if declared_revision_ids != revision_ids or declared_block_ids != block_ids:
            _fail("derived evidence source identity arrays differ from source references")

        transformations = _unique_strings(
            tree_node.get("transformations"), "transformations"
        )
        if not transformations:
            _fail("derived evidence requires at least one transformation")
        first = blocks[0]
        if tree_node.get("page") != first.page_index:
            _fail("derived tree node page differs from its primary source")
        if tree_node.get("page_id") != first.page_id:
            _fail("derived tree node page_id differs from its primary source")
        expected_geometry = first.geometry.to_dict() if first.geometry else None
        geometry = tree_node.get("geometry", tree_node.get("source_geometry"))
        if not _same(geometry, expected_geometry):
            _fail("derived tree node geometry differs from its primary source")
        expected_observations = _ordered_unique(
            tuple(
                observation_id
                for block in blocks
                for observation_id in block.source_observation_ids
            )
        )
        declared_observations = _unique_strings(
            tree_node.get("source_observation_ids"), "source_observation_ids"
        )
        if declared_observations != expected_observations:
            _fail("derived evidence observations differ from canonical sources")
        declared_semantic = tree_node.get("semantic_substructures")
        if declared_semantic is not None and not _same(
            declared_semantic, _semantic_substructure_union(blocks)
        ):
            _fail(
                "derived semantic substructures differ from the exact canonical "
                "source graphs"
            )

        source_document_version_ids = _ordered_unique(
            tuple(
                value
                for value in (
                    tree_node.get("document_version_id"),
                    *(reference.get("document_version_id") for reference in refs),
                )
                if isinstance(value, str) and value
            )
        )
        # The refiner seals this identity over the *block* it transformed, and
        # a block's type is not always the node's: the outline pass afterwards
        # promotes a text block that opens a section to "title". Reading the
        # node's type here made every promoted heading unciteable, so the type
        # comes from the same canonical projection the refiner was handed.
        primary = project_canonical_blocks(
            self.catalog.document, (first,), validate_relation_targets=False
        )[0]
        identity_payload = derived_identity_payload(
            output=derived_output_identity(
                content_type=primary.get("type", "unknown"),
                text=tree_node.get("content", ""),
                table_body=tree_node.get("table_body", ""),
                table_caption=tree_node.get("caption", ""),
                table_footnote=tree_node.get("footnote", ""),
            ),
            source_revision_ids=declared_revision_ids,
            source_document_version_ids=source_document_version_ids,
            source_block_ids=declared_block_ids,
            source_observation_ids=declared_observations,
            source_references=refs,
            transformations=transformations,
        )
        if derived_evidence_id(identity_payload) != evidence_id:
            _fail("derived evidence digest differs from its output or canonical lineage")
        return tuple(blocks), transformations, evidence_id

    def _validate_direct_tree_node(
        self, tree_node: Mapping[str, object], block: CanonicalBlock
    ) -> None:
        if tree_node.get("document_version_id") != block.document_version_id:
            _fail("direct tree node belongs to another document version")
        if tree_node.get("page") != block.page_index:
            _fail("direct tree node page differs from canonical block")
        if tree_node.get("page_id") != block.page_id:
            _fail("direct tree node page_id differs from canonical block")
        expected_geometry = block.geometry.to_dict() if block.geometry else None
        geometry = tree_node.get("geometry", tree_node.get("source_geometry"))
        if not _same(geometry, expected_geometry):
            _fail("direct tree node geometry differs from canonical block")
        declared_observations = tree_node.get("source_observation_ids")
        if declared_observations is not None and _unique_strings(
            declared_observations, "source_observation_ids"
        ) != tuple(block.source_observation_ids):
            _fail("direct tree node observations differ from canonical block")
        observation_id = tree_node.get("observation_id")
        if observation_id is not None and observation_id not in block.source_observation_ids:
            _fail("direct tree node observation_id differs from canonical block")
        source_block_ids = tree_node.get("source_canonical_block_ids")
        if source_block_ids is not None and _unique_strings(
            source_block_ids, "source_canonical_block_ids"
        ) != (block.block_id,):
            _fail("direct tree node source block identities are inconsistent")
        source_revision_ids = tree_node.get("source_canonical_revision_ids")
        if source_revision_ids is not None and _unique_strings(
            source_revision_ids, "source_canonical_revision_ids"
        ) != (block.revision_id,):
            _fail("direct tree node source revision identities are inconsistent")
        representation_ids = tree_node.get("canonical_representation_ids")
        if representation_ids is not None and set(
            _unique_strings(
                representation_ids, "canonical_representation_ids"
            )
        ) != {item.representation_id for item in block.representations}:
            _fail(
                "direct tree node representation identities are inconsistent"
            )
        asset_ids = tree_node.get("canonical_asset_ids")
        if asset_ids is not None and set(
            _unique_strings(asset_ids, "canonical_asset_ids")
        ) != {item.asset_id for item in block.assets}:
            _fail("direct tree node asset identities are inconsistent")
        canonical_relations = tree_node.get("canonical_relations")
        if canonical_relations is not None and not _same(
            canonical_relations,
            [relation.to_dict() for relation in block.relations],
        ):
            _fail("direct tree node structural relations differ from canonical block")
        semantic_substructures = tree_node.get("semantic_substructures")
        if semantic_substructures is not None and not _same(
            semantic_substructures,
            [structure.to_dict() for structure in block.semantic_substructures],
        ):
            _fail("direct tree node semantic substructures differ from canonical block")
        source_refs = tree_node.get("source_evidence_refs")
        if source_refs is not None:
            if not isinstance(source_refs, Sequence) or isinstance(
                source_refs, (str, bytes, bytearray)
            ) or len(source_refs) != 1:
                _fail("direct tree node must reference exactly one canonical source")
            source_ref = source_refs[0]
            if not isinstance(source_ref, Mapping):
                _fail("direct tree node source reference must be an object")
            if source_ref.get("canonical_block_id") != block.block_id or source_ref.get(
                "canonical_revision_id"
            ) != block.revision_id:
                _fail("direct tree node source reference identity is inconsistent")
            self._validate_source_ref(source_ref, block, position=0)

        # Re-project one canonical block through the same deterministic
        # compatibility contract and compare the evidence-bearing BookRAG view.
        projected = project_canonical_blocks(
            self.catalog.document, (block,), validate_relation_targets=False
        )[0]
        declared = tree_node.get("selected_representation_ids")
        if declared is not None and _unique_strings(
            declared, "selected_representation_ids"
        ) != tuple(projected.get("selected_representation_ids", [])):
            _fail(
                "direct tree node selected_representation_ids differs from "
                "canonical projection"
            )
        # Which asset became the row's image is decided by the index-time
        # materialization lease, and this hydrator is deliberately
        # reference-only, so that choice is not reproducible here.  Verify the
        # node cites only assets the canonical block actually owns.
        declared_assets = tree_node.get("selected_asset_ids")
        if declared_assets is not None and not set(
            _unique_strings(declared_assets, "selected_asset_ids")
        ) <= {item.asset_id for item in block.assets}:
            _fail("direct tree node selected_asset_ids are not canonical block assets")
        projected_type = projected.get("type")
        node_type = _json_value(tree_node.get("type"))
        # A node keeps the type the projection gave it. The outline pass may
        # additionally promote any text-carrying node to "title", so that one
        # substitution is allowed everywhere a text channel is expected.
        if projected_type in _TEXT_NODE_TYPES:
            if node_type not in {projected_type, "title"}:
                _fail("direct tree node type differs from canonical block")
            expected_content = projected.get("text", "")
        elif projected_type == "table":
            if node_type != "table":
                _fail("direct tree node type differs from canonical block")
            expected_content = caption_footnote_text(
                projected.get("table_caption"), projected.get("table_footnote")
            )
            if tree_node.get("table_body") != projected.get("table_body"):
                _fail("direct tree table body differs from canonical block")
        elif projected_type == "equation":
            if node_type != "equation":
                _fail("direct tree node type differs from canonical block")
            expected_content = projected.get("text", "")
        elif projected_type in _VISUAL_NODE_TYPES:
            if node_type != projected_type:
                _fail("direct tree node type differs from canonical block")
            # Caption and footnote are separate channels, and the tree writes
            # them as one string with a space between. Gluing them here instead
            # refused every figure and table that has both -- which is most of
            # the ones a question asks about, because a figure worth citing is
            # usually the one the document bothered to caption.
            expected_content = caption_footnote_text(
                projected.get("image_caption"), projected.get("image_footnote")
            )
        else:  # pragma: no cover - projector rejects unsupported block kinds.
            _fail("canonical projection produced an unsupported node type")
        if tree_node.get("content") != expected_content:
            _fail("direct tree node content differs from canonical block")
        projected_img_path = projected.get("img_path")
        if (
            projected_type in _CROP_BEARING_NODE_TYPES
            and projected_img_path
            and tree_node.get("img_path") != projected_img_path
        ):
            _fail("direct tree node crop differs from canonical block")

    def _validate_source_ref(
        self,
        reference: Mapping[str, object],
        block: CanonicalBlock,
        *,
        position: int,
    ) -> None:
        prefix = f"source_evidence_refs[{position}]"
        expected = (
            ("document_version_id", block.document_version_id),
            ("page_id", block.page_id),
            ("page_idx", block.page_index),
        )
        for field_name, expected_value in expected:
            if reference.get(field_name) != expected_value:
                _fail(f"{prefix}.{field_name} differs from canonical block")
        source_sha256 = reference.get("source_sha256")
        if source_sha256 is not None and source_sha256 != self.catalog.document.content_sha256:
            _fail(f"{prefix}.source_sha256 differs from canonical document")
        geometry = block.geometry.to_dict() if block.geometry else None
        if not _same(reference.get("geometry"), geometry):
            _fail(f"{prefix}.geometry differs from canonical block")
        observations = reference.get("source_observation_ids")
        if observations is not None and set(
            _unique_strings(observations, f"{prefix}.source_observation_ids")
        ) != set(block.source_observation_ids):
            _fail(f"{prefix}.source_observation_ids differs from canonical block")
        observation_id = reference.get("observation_id")
        if observation_id is not None and observation_id not in block.source_observation_ids:
            _fail(f"{prefix}.observation_id differs from canonical block")
        representation_ids = reference.get("representation_ids")
        if representation_ids is not None and set(
            _unique_strings(representation_ids, f"{prefix}.representation_ids")
        ) != {item.representation_id for item in block.representations}:
            _fail(f"{prefix}.representation_ids differs from canonical block")
        asset_ids = reference.get("asset_ids")
        if asset_ids is not None and set(
            _unique_strings(asset_ids, f"{prefix}.asset_ids")
        ) != {item.asset_id for item in block.assets}:
            _fail(f"{prefix}.asset_ids differs from canonical block")
        selected_representation_ids = reference.get(
            "selected_representation_ids"
        )
        if selected_representation_ids is not None and not set(
            _unique_strings(
                selected_representation_ids,
                f"{prefix}.selected_representation_ids",
            )
        ).issubset({item.representation_id for item in block.representations}):
            _fail(
                f"{prefix}.selected_representation_ids is not canonical material"
            )
        selected_asset_ids = reference.get("selected_asset_ids")
        if selected_asset_ids is not None and not set(
            _unique_strings(
                selected_asset_ids, f"{prefix}.selected_asset_ids"
            )
        ).issubset({item.asset_id for item in block.assets}):
            _fail(f"{prefix}.selected_asset_ids is not canonical material")
        canonical_relations = reference.get("canonical_relations")
        if canonical_relations is not None and not _same(
            canonical_relations,
            [relation.to_dict() for relation in block.relations],
        ):
            _fail(f"{prefix}.canonical_relations differs from canonical block")
        semantic_substructures = reference.get("semantic_substructures")
        if semantic_substructures is not None:
            _fail(
                f"{prefix}.semantic_substructures duplicates full canonical graphs"
            )
        semantic_substructure_refs = reference.get(
            "semantic_substructure_refs"
        )
        if semantic_substructure_refs is not None:
            expected_semantic_refs = _semantic_substructure_refs(block)
            legacy_semantic_refs = [
                {
                    key: value
                    for key, value in receipt.items()
                    if key != "comparison_sha256"
                }
                for receipt in expected_semantic_refs
            ]
            if not _same(
                semantic_substructure_refs, expected_semantic_refs
            ) and not _same(semantic_substructure_refs, legacy_semantic_refs):
                _fail(
                    f"{prefix}.semantic_substructure_refs differs from "
                    "canonical block"
                )


__all__ = [
    "ASSET_MATERIAL_NOT_APPLICABLE",
    "ASSET_MATERIAL_REFERENCE_ONLY",
    "CITATION_SCHEMA_VERSION",
    "CanonicalEvidenceCatalog",
    "CanonicalEvidenceHydrator",
    "CitationLocation",
    "EvidenceHydrationError",
    "SourceCitation",
]
