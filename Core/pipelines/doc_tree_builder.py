from collections.abc import Collection
from dataclasses import dataclass, replace
import hashlib
from importlib import metadata as importlib_metadata
import json
import logging
import os
from pathlib import Path
import stat
import tempfile
from typing import Protocol

from Core.Index.Tree import FURNITURE_NODE_TYPES, DocumentTree
from Core.configs.llm_config import indexing_llm_config
from Core.configs.system_config import SystemConfig
from Core.pipelines.outline_extractor import extract_pdf_outline_in_chunks
from Core.pipelines.pdf_refiner import pdf_info_refiner
from Core.pipelines.sharded_parsing import (
    offset_page_indices,
    page_count_from_blocks,
    partition_blocks_by_shard,
    pdf_page_count,
    shard_ranges,
)
from Core.pipelines.tree_node_builder import (
    _TEXT_CONTENT_TYPES,
    create_node_by_type,
)

# Projected types that may become a heading. Restricting this to "text" meant
# a code listing or an abstract could never open a section, and every type
# added to the projection silently lost the ability to be a title. Page
# furniture is excluded: a running header sits where a heading sits and reads
# like one, so left eligible it would open a section on every page.
_TITLE_CAPABLE_TYPES = frozenset(
    name
    for name, node_type in _TEXT_CONTENT_TYPES.items()
    if node_type not in FURNITURE_NODE_TYPES
)
from Core.pipelines.tree_node_summary import generate_tree_node_summary
from Core.provider.TokenTracker import TokenTracker
from Core.utils.phase_timing import PhaseTimings

#: canonical_bind is the tree stage's largest phase and its internals have never
#: been separated. Module-level because the function is called once per shard
#: and the interesting total is across all of them.
_BIND_TIMINGS = PhaseTimings("canonical bind")
from Core.provider.extract_pdf_info import merge_middle_content, parse_doc
from docarag.page_transcription import retranscribe_scanned_pages
from Core.provider.llm import LLM
from Core.provider.vlm import VLM
from docarag.adapters.mineru_compat import MinerUNormalizedAdapter
from docarag.artifact_store import LocalContentAddressedArtifactStore
from docarag.bookrag_projection import (
    BookRAGAssetMaterializationLease,
    project_to_bookrag_pdf_list,
)
from docarag.crop_materialization import materialize_visual_crops
from docarag.figure_describe import GroundedFigureDescriber
from docarag.figure_diagram_vlm import VLMDiagramExtractor
from docarag.figure_enrichment import VISUAL_BLOCK_TYPES, enrich_visual_blocks
from docarag.canonical_artifacts import (
    CANONICAL_EVIDENCE_FILE,
    CANONICAL_EVIDENCE_SCHEMA_VERSION,
    ROUTED_PARSING_AUDIT_SCHEMA_VERSION,
)
from docarag.domain import (
    BlockType,
    CanonicalBlock,
    DocumentRef,
    IssueSeverity,
    PageRef,
    ParseOptions,
    ParseRequest,
    ParseStatus,
    QualityStatus,
    RelationType,
)
from docarag.domain.base import CONTRACT_SCHEMA_VERSION, content_sha256
from docarag.ports import ParserRegistry
from docarag.persistence.stage_runtime import durable_stage
from docarag.reconciliation import ReconciliationError, reconcile_observations
from docarag.text_recovery import recover_and_prune_empty_text_blocks

log = logging.getLogger(__name__)


PARSER_OBSERVATION_SCHEMA_VERSION = "docarag.parser-observations.v1"
MAX_CORE_INGESTION_ARTIFACT_BYTES = 128 * 1024 * 1024

# The only relation kinds mineru_compat.py's _STRUCTURAL_FIELDS registers
# with allow_cross_page=True -- the sole kinds a per-shard bind_canonical_evidence
# call could see an UNRESOLVED_RELATION_TARGET issue for, purely because its
# target landed in a different shard, not because the reference is broken.
_CROSS_PAGE_RELATION_TYPE_VALUES = frozenset(
    {
        RelationType.REFERENCES.value,
        RelationType.REFERENCED_BY.value,
        RelationType.TABLE_CONTINUES.value,
        RelationType.TABLE_CONTINUED_FROM.value,
    }
)


class CanonicalEvidenceBindingError(RuntimeError):
    """MinerU output could not be bound losslessly to canonical evidence."""


class CanonicalIngestionBindingError(RuntimeError):
    """Injected canonical evidence cannot safely enter the BookRAG pipeline."""


class TreeConstructionError(RuntimeError):
    """Normalized BookRAG blocks cannot be assembled without evidence loss."""


@dataclass(frozen=True, slots=True)
class CanonicalIngestionPolicy:
    """Explicit quality gate for evidence entering legacy BookRAG stages."""

    allowed_quality_statuses: tuple[QualityStatus, ...] = (
        QualityStatus.ACCEPTED,
    )

    def __post_init__(self) -> None:
        try:
            statuses = tuple(
                sorted(
                    {QualityStatus(value) for value in self.allowed_quality_statuses},
                    key=lambda value: value.value,
                )
            )
        except (TypeError, ValueError) as exc:
            raise CanonicalIngestionBindingError(
                "canonical ingestion policy contains an invalid quality status"
            ) from exc
        if not statuses:
            raise CanonicalIngestionBindingError(
                "canonical ingestion policy must allow at least one quality status"
            )
        object.__setattr__(self, "allowed_quality_statuses", statuses)


@dataclass(frozen=True, slots=True)
class CanonicalIngestionResult:
    """Core-normalized snapshot of evidence supplied at the BookRAG boundary.

    Parser selection, page routing, reconciliation, persistence, and model
    lifecycle remain owned by the provider.  Core receives only the canonical
    result and its provider-declared artifact locations, then applies the shared
    loss-aware projection to BookRAG's historical ``pdf_list`` shape. Core
    verifies a stable byte receipt for an optional replay audit and carries it
    on tree/root metadata; schema/content validation remains the provider or
    application boundary's responsibility and is not inferred from a path.
    Providers are structurally typed and do not need to import or construct this
    Core-local normalization type.
    """

    document: DocumentRef
    canonical_blocks: tuple[CanonicalBlock, ...]
    canonical_artifact_path: str | os.PathLike[str]
    canonical_artifact_sha256: str | None = None
    canonical_artifact_size_bytes: int | None = None
    audit_artifact_path: str | os.PathLike[str] | None = None
    audit_artifact_sha256: str | None = None
    audit_artifact_size_bytes: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.document, DocumentRef):
            raise CanonicalIngestionBindingError(
                "canonical ingestion document must be a DocumentRef"
            )
        blocks = tuple(self.canonical_blocks)
        if not blocks or any(not isinstance(block, CanonicalBlock) for block in blocks):
            raise CanonicalIngestionBindingError(
                "canonical ingestion requires at least one CanonicalBlock"
            )
        if any(
            block.document_version_id != self.document.document_version_id
            for block in blocks
        ):
            raise CanonicalIngestionBindingError(
                "canonical ingestion blocks belong to another document version"
            )
        canonical_keys = [block.canonical_key for block in blocks]
        block_ids = [block.block_id for block in blocks]
        revision_ids = [block.revision_id for block in blocks]
        if len(block_ids) != len(set(block_ids)):
            raise CanonicalIngestionBindingError(
                "canonical ingestion contains multiple current revisions for a block"
            )
        if len(revision_ids) != len(set(revision_ids)):
            raise CanonicalIngestionBindingError(
                "canonical ingestion contains duplicate revision identities"
            )
        if len(canonical_keys) != len(set(canonical_keys)):
            raise CanonicalIngestionBindingError(
                "canonical ingestion contains duplicate canonical identities"
            )
        if not isinstance(self.canonical_artifact_path, (str, os.PathLike)):
            raise CanonicalIngestionBindingError(
                "canonical artifact path must be a filesystem path"
            )
        artifact_path, artifact_sha256, artifact_size = _stable_local_artifact_receipt(
            self.canonical_artifact_path,
            label="canonical evidence",
        )
        if self.canonical_artifact_sha256 not in (None, artifact_sha256):
            raise CanonicalIngestionBindingError(
                "canonical evidence artifact SHA-256 differs from provider receipt"
            )
        if self.canonical_artifact_size_bytes not in (None, artifact_size):
            raise CanonicalIngestionBindingError(
                "canonical evidence artifact size differs from provider receipt"
            )
        object.__setattr__(self, "canonical_blocks", blocks)
        object.__setattr__(self, "canonical_artifact_path", str(artifact_path))
        object.__setattr__(self, "canonical_artifact_sha256", artifact_sha256)
        object.__setattr__(self, "canonical_artifact_size_bytes", artifact_size)
        if self.audit_artifact_path is None:
            if self.audit_artifact_sha256 is not None or self.audit_artifact_size_bytes is not None:
                raise CanonicalIngestionBindingError(
                    "audit artifact digest/size require audit_artifact_path"
                )
            return
        audit_path, audit_sha256, audit_size = _stable_local_artifact_receipt(
            self.audit_artifact_path,
            label="routed audit",
        )
        if self.audit_artifact_sha256 not in (None, audit_sha256):
            raise CanonicalIngestionBindingError(
                "routed audit artifact SHA-256 differs from provider receipt"
            )
        if self.audit_artifact_size_bytes not in (None, audit_size):
            raise CanonicalIngestionBindingError(
                "routed audit artifact size differs from provider receipt"
            )
        object.__setattr__(self, "audit_artifact_path", str(audit_path))
        object.__setattr__(self, "audit_artifact_sha256", audit_sha256)
        object.__setattr__(self, "audit_artifact_size_bytes", audit_size)


class CanonicalIngestionPayload(Protocol):
    """Structural output contract implemented without importing BookRAG Core."""

    document: DocumentRef
    canonical_blocks: tuple[CanonicalBlock, ...]
    canonical_artifact_path: str | os.PathLike[str]


class CanonicalIngestionProvider(Protocol):
    """Opaque provider seam; routing and model policy must stay outside Core."""

    def ingest(
        self,
        cfg: SystemConfig,
        *,
        reforce: bool = False,
    ) -> CanonicalIngestionPayload:
        """Return persisted canonical evidence for one immutable source."""


def _stable_local_artifact_receipt(
    path_value: str | os.PathLike[str],
    *,
    label: str,
) -> tuple[Path, str, int]:
    if not isinstance(path_value, (str, os.PathLike)):
        raise CanonicalIngestionBindingError(f"{label} path must be a filesystem path")
    try:
        path = Path(os.fspath(path_value)).expanduser().resolve(strict=True)
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise CanonicalIngestionBindingError(f"{label} artifact is unavailable") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size < 1:
            raise CanonicalIngestionBindingError(
                f"{label} artifact must be a non-empty regular file"
            )
        if before.st_size > MAX_CORE_INGESTION_ARTIFACT_BYTES:
            raise CanonicalIngestionBindingError(
                f"{label} artifact exceeds {MAX_CORE_INGESTION_ARTIFACT_BYTES} bytes"
            )
        digest = hashlib.sha256()
        size = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > MAX_CORE_INGESTION_ARTIFACT_BYTES:
                raise CanonicalIngestionBindingError(
                    f"{label} artifact grew beyond its byte limit"
                )
            digest.update(chunk)
        after = os.fstat(descriptor)
        path_after = os.stat(path, follow_symlinks=False)
    except OSError as exc:
        raise CanonicalIngestionBindingError(f"{label} artifact could not be read") from exc
    finally:
        os.close(descriptor)
    before_fingerprint = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    after_fingerprint = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    path_fingerprint = (
        path_after.st_dev,
        path_after.st_ino,
        path_after.st_size,
        path_after.st_mtime_ns,
        path_after.st_ctime_ns,
    )
    if before_fingerprint != after_fingerprint or after_fingerprint != path_fingerprint or size != before.st_size:
        raise CanonicalIngestionBindingError(
            f"{label} artifact changed while its receipt was created"
        )
    return path, digest.hexdigest(), size


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    """Durably replace a deterministic JSON artifact in the target directory."""

    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(encoded)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _installed_mineru_version() -> str:
    """Resolve MinerU without importing its CUDA/ML dependency graph."""

    for distribution_name in ("mineru", "magic-pdf"):
        try:
            return importlib_metadata.version(distribution_name)
        except importlib_metadata.PackageNotFoundError:
            continue
    return "unknown"


def _mineru_model_identity(cfg: SystemConfig) -> str:
    """Return a secret-free identity for the configured MinerU execution path."""

    return f"mineru:{cfg.mineru.backend}:{cfg.mineru.method}"


def _normalized_block_native_key(block: dict[str, object]) -> str:
    """Identify a source block without run-local crop or artifact paths."""

    for field_name in ("parser_native_id", "block_id"):
        value = block.get(field_name)
        if value is not None and str(value).strip() and str(value).strip() != "-1":
            return content_sha256({"field": field_name, "value": str(value).strip()})

    middle = block.get("middle_json")
    middle_block = middle if isinstance(middle, dict) else {}
    for field_name in ("id", "block_id"):
        value = middle_block.get(field_name)
        if value is not None and str(value).strip():
            return content_sha256(
                {"field": f"middle_json.{field_name}", "value": str(value).strip()}
            )

    spatial_identity = {
        "page_idx": block.get("page_idx"),
        "type": block.get("type"),
        "bbox": middle_block.get("bbox"),
        "polygon": middle_block.get("polygon"),
        "reading_order": block.get("reading_order", middle_block.get("index")),
    }
    if not any(
        spatial_identity[field_name] is not None
        for field_name in ("bbox", "polygon", "reading_order")
    ):
        # Geometry-free legacy observations need a content fallback. Paths are
        # deliberately excluded because they are different in another run.
        spatial_identity["content"] = {
            "text": block.get("text"),
            "table_body": block.get("table_body"),
            "image_caption": block.get("image_caption"),
            "table_caption": block.get("table_caption"),
        }
    return content_sha256(spatial_identity)


def _make_parse_request(
    cfg: SystemConfig, pdf_list: list[dict], *, document_page_count: int | None = None
) -> ParseRequest:
    source = cfg.source
    if source is None:
        raise CanonicalEvidenceBindingError(
            "canonical evidence requires cfg.source with immutable source identity"
        )
    if not source.sha256:
        raise CanonicalEvidenceBindingError(
            "canonical evidence requires cfg.source.sha256"
        )
    page_indices = sorted(
        {
            block.get("page_idx")
            for block in pdf_list
            if isinstance(block, dict)
            and isinstance(block.get("page_idx"), int)
            and not isinstance(block.get("page_idx"), bool)
            and block.get("page_idx") >= 0
        }
    )
    if not page_indices:
        raise CanonicalEvidenceBindingError(
            "MinerU produced no blocks bound to a valid page"
        )

    document = DocumentRef.identified(
        content_sha256=source.sha256,
        mime_type="application/pdf",
        source_uri=source.original_uri,
        display_name=source.display_name,
        # A shard only ever sees its own blocks, so page_indices[-1] + 1
        # truncates page_count for every shard but the one containing the
        # document's last page. document_version_id doesn't depend on
        # page_count (only on source_uri/content_sha256), so this was never
        # an identity bug -- just wrong descriptive metadata. The per-shard
        # caller passes the document's real total; the non-sharded caller
        # (or a test) still gets today's derived-from-blocks behavior.
        page_count=(
            document_page_count if document_page_count is not None
            else page_indices[-1] + 1
        ),
        metadata={
            "snapshot_path": source.snapshot_path or cfg.pdf_path,
            "parser_input_path": cfg.pdf_path,
        },
    )
    if (
        getattr(source, "document_id", None) is not None
        and source.document_id != document.document_id
    ) or (
        getattr(source, "document_version_id", None) is not None
        and source.document_version_id != document.document_version_id
    ):
        raise CanonicalEvidenceBindingError(
            "configured source document IDs differ from the parser DocumentRef"
        )
    language = cfg.mineru.lang.strip() if cfg.mineru.lang else ""
    return ParseRequest(
        document=document,
        pages=tuple(
            PageRef(document.document_version_id, index) for index in page_indices
        ),
        options=ParseOptions(
            languages=(language,) if language else (),
            extra={
                "backend": cfg.mineru.backend,
                "method": cfg.mineru.method,
            },
        ),
        run_id=Path(cfg.save_path).name,
        idempotency_key=f"mineru-normalized:{source.sha256}",
    )


def _enrich_figures(
    cfg: SystemConfig,
    blocks: tuple[CanonicalBlock, ...],
    crop_store: LocalContentAddressedArtifactStore,
) -> tuple[CanonicalBlock, ...]:
    """Recover the text and meaning MinerU leaves inside an image region."""

    from docarag.formula_transcription import (
        VLMFormulaTranscriber,
        has_inline_formula_candidates,
        transcribe_equation_blocks,
        transcribe_inline_equations,
    )

    from docarag.table_transcription import (
        VLMTableTranscriber,
        transcribe_table_blocks,
    )

    has_visual = any(block.block_type in VISUAL_BLOCK_TYPES for block in blocks)
    has_equations = any(
        block.block_type is BlockType.EQUATION for block in blocks
    )
    has_tables = any(block.block_type is BlockType.TABLE for block in blocks)
    run_figures = cfg.figures.ocr and has_visual
    run_formulas = cfg.figures.transcribe_formulas and has_equations
    run_tables = cfg.figures.transcribe_tables and has_tables
    # MinerU crops individual equation spans even inside blocks it never
    # types as EQUATION (a formula written inline in a paragraph, say) --
    # transcribe_equation_blocks only ever looks at BlockType.EQUATION, so
    # this is a second, separate pass over the blocks it skips.
    run_inline_formulas = cfg.figures.transcribe_formulas and (
        has_inline_formula_candidates(blocks)
    )
    if (
        not run_figures
        and not run_formulas
        and not run_inline_formulas
        and not run_tables
    ):
        return blocks

    wants_vlm = run_formulas or run_inline_formulas or run_tables or (
        run_figures and (cfg.figures.describe or cfg.figures.diagrams)
    )
    vlm = VLM(cfg.vlm) if wants_vlm else None
    enriched = tuple(blocks)
    try:
        if run_figures:
            from docarag.adapters.mineru_figure_ocr import MinerUFigureOCR

            enriched, stats = enrich_visual_blocks(
                enriched,
                resolver=crop_store.make_asset_resolver(),
                ocr_engine=MinerUFigureOCR(lang=cfg.figures.ocr_lang),
                describer=(
                    GroundedFigureDescriber(vlm)
                    if vlm is not None and cfg.figures.describe
                    else None
                ),
                diagram_extractor=(
                    VLMDiagramExtractor(vlm)
                    if vlm is not None and cfg.figures.diagrams
                    else None
                ),
                min_confidence=cfg.figures.min_confidence,
            )
        if run_formulas:
            enriched, formula_stats = transcribe_equation_blocks(
                enriched,
                resolver=crop_store.make_asset_resolver(),
                transcriber=VLMFormulaTranscriber(vlm),
            )
            log.info(
                "formula transcription: %d equations, %d transcribed, "
                "%d skipped, %d failed",
                formula_stats.equation_blocks,
                formula_stats.transcribed_blocks,
                formula_stats.skipped_blocks,
                formula_stats.failed_blocks,
            )
        if run_tables:
            enriched, table_stats = transcribe_table_blocks(
                enriched,
                resolver=crop_store.make_asset_resolver(),
                transcriber=VLMTableTranscriber(vlm),
            )
            log.info(
                "table transcription: %d tables, %d transcribed, %d skipped, "
                "%d failed, %d rejected on shape",
                table_stats.table_blocks,
                table_stats.transcribed_blocks,
                table_stats.skipped_blocks,
                table_stats.failed_blocks,
                table_stats.shape_mismatch_blocks,
            )
        if run_inline_formulas:
            enriched, inline_stats = transcribe_inline_equations(
                enriched,
                resolver=crop_store.make_asset_resolver(),
                transcriber=VLMFormulaTranscriber(vlm),
            )
            log.info(
                "inline formula correction: %d candidates, %d corrected, "
                "%d skipped, %d failed",
                inline_stats.candidate_blocks,
                inline_stats.corrected_blocks,
                inline_stats.skipped_blocks,
                inline_stats.failed_blocks,
            )
    finally:
        if vlm is not None:
            vlm.close()
    if run_figures:
        log.info(
            "figure enrichment: %d visual, %d with text (%d spans), %d described, "
            "%d graphs (%d nodes, %d edges), %d skipped, %d failed",
            stats.visual_blocks,
            stats.ocr_blocks,
            stats.ocr_spans,
            stats.described_blocks,
            stats.graph_blocks,
            stats.graph_nodes,
            stats.graph_edges,
            stats.skipped_blocks,
            stats.failed_blocks,
        )
    return enriched


def bind_canonical_evidence(
    cfg: SystemConfig,
    pdf_list: list[dict],
    *,
    merged_content_path: str | os.PathLike[str],
    document_page_count: int | None = None,
    debug_artifact_suffix: str = "",
    canonical_artifact_filename: str = CANONICAL_EVIDENCE_FILE,
) -> tuple[ParseRequest, tuple[CanonicalBlock, ...], bool]:
    """Translate MinerU blocks and attach immutable evidence IDs one-to-one.

    The legacy dictionaries remain in their original order.  The adapter may
    sort its typed observations by page/reading order, so an injected stable
    native ID is used for the lossless join back to each dictionary.

    ``document_page_count`` lets a per-shard caller report the document's
    real total instead of the truncated one this function would otherwise
    derive from just the blocks it was given (see ``_make_parse_request``).

    ``debug_artifact_suffix`` is folded into this call's debug/audit dump
    name (``..._parser_observations.json``) so a per-shard caller's dumps
    don't silently overwrite each other under the same document-derived
    name -- the same bug class already found and fixed for ``do_parse``'s
    own dumps when the parse stage was sharded.

    ``canonical_artifact_filename`` overrides where this call's canonical
    blocks are written. Left at its default, behavior is unchanged: one
    call publishes the complete, final ``canonical_blocks.json``. A
    per-shard caller instead points this at a small per-shard fragment
    filename and calls ``merge_canonical_evidence_shards`` once, after every
    shard is bound, to produce the one complete artifact every existing
    reader (``evidence_projection.py``, ``evidence_persist.py``,
    ``docarag.provenance``) still expects -- this function's own writing
    logic doesn't need to know which mode it's in beyond the filename.

    Returns ``(request, canonical_blocks, has_unresolved_cross_page_relation)``.
    The third value is ``True`` only if some block's relation is one of the
    kinds that can legitimately point across a page (a table continuation or
    an explicit reference) and this call's own blocks didn't include its
    target -- for a per-shard caller, that means the target is probably in a
    different shard, not that the reference is genuinely broken, and a
    whole-document call would resolve it. A per-shard caller should treat
    this as a signal to fall back to one whole-document call for full
    correctness rather than trust the per-shard result for this document.
    """

    request = _make_parse_request(
        cfg, pdf_list, document_page_count=document_page_count
    )
    occurrence_by_digest: dict[str, int] = {}
    prepared_blocks: list[dict[str, object]] = []
    native_ids: list[str] = []
    for block in pdf_list:
        if not isinstance(block, dict):
            raise CanonicalEvidenceBindingError(
                "MinerU normalized output must contain JSON objects only"
            )
        digest = _normalized_block_native_key(block)
        occurrence = occurrence_by_digest.get(digest, 0) + 1
        occurrence_by_digest[digest] = occurrence
        native_token = f"normalized:{digest}:{occurrence}"
        # The compatibility adapter names the source key as part of its native
        # ID, preventing collisions with a vendor-owned ``block_id``.
        native_id = f"parser_native_id:{native_token}"
        prepared = dict(block)
        prepared["parser_native_id"] = native_token
        prepared_blocks.append(prepared)
        native_ids.append(native_id)

    parser_config = {
        "backend": cfg.mineru.backend,
        "method": cfg.mineru.method,
        "language": cfg.mineru.lang,
    }
    merged_path = Path(merged_content_path).expanduser().resolve()
    adapter = MinerUNormalizedAdapter(
        lambda _request: prepared_blocks,
        parser_version=_installed_mineru_version(),
        model_version=_mineru_model_identity(cfg),
        config=parser_config,
        raw_artifact_uri=merged_path.as_uri(),
    )
    result = ParserRegistry((adapter,)).parse("mineru-normalized", request)
    manifest = adapter.manifest()
    observations_by_native_id = {
        observation.provenance.parser_native_id: observation
        for observation in result.observations
    }
    deterministic_result = result.to_dict()
    deterministic_result["duration_ms"] = None
    observation_payload = {
        "artifact_schema_version": PARSER_OBSERVATION_SCHEMA_VERSION,
        "contract_schema_version": CONTRACT_SCHEMA_VERSION,
        "document": request.document.to_dict(),
        "request": request.to_dict(),
        "parser_manifest": manifest.to_dict(),
        "result": deterministic_result,
        "legacy_order_observation_ids": [
            observations_by_native_id[native_id].observation_id
            for native_id in native_ids
            if native_id in observations_by_native_id
        ],
    }
    observation_path = (
        Path(cfg.save_path)
        / cfg.mineru.method
        / f"{Path(cfg.pdf_path).stem}{debug_artifact_suffix}_parser_observations.json"
    )
    # Failed/partial translations remain inspectable even though no canonical
    # evidence or tree is allowed to publish from them.
    _atomic_write_json(observation_path, observation_payload)

    error_issues = tuple(
        issue for issue in result.issues if issue.severity == IssueSeverity.ERROR
    )
    if result.status != ParseStatus.SUCCEEDED or error_issues:
        messages = [issue.message for issue in error_issues]
        details = "; ".join(messages[:5]) or result.status.value
        if len(messages) > 5:
            details += f"; ... and {len(messages) - 5} more error issue(s)"
        raise CanonicalEvidenceBindingError(
            "MinerU parser-contract translation failed; inspect "
            f"{observation_path}: {details}"
        )
    if len(result.observations) != len(pdf_list):
        raise CanonicalEvidenceBindingError(
            "MinerU parser-contract translation was not one-to-one: "
            f"{len(pdf_list)} blocks, {len(result.observations)} observations"
        )

    if len(observations_by_native_id) != len(result.observations):
        raise CanonicalEvidenceBindingError(
            "MinerU parser-contract translation emitted duplicate native IDs"
        )

    try:
        with _BIND_TIMINGS.time("reconcile"):
            reconciled = reconcile_observations(result.observations)
    except (ReconciliationError, TypeError, ValueError) as exc:
        raise CanonicalEvidenceBindingError(
            "MinerU complete-observation reconciliation failed"
        ) from exc

    # A relation whose kind can legitimately point across a page
    # (TABLE_CONTINUES/TABLE_CONTINUED_FROM/REFERENCES/REFERENCED_BY -- the
    # only kinds mineru_compat.py registers allow_cross_page=True) is
    # resolved at translation time (mineru_compat.py's _attach_structural_relations),
    # inside this same call's own ParserRegistry.parse(request) above -- not
    # by reconcile_observations. A per-shard caller only ever passes one
    # shard's blocks into that translation call, so a relation whose target
    # lives in a different shard is never even found there: it never becomes
    # an ObservationRelation at all (confirmed live -- checked
    # reconcile_observations' own issues first, found nothing, because
    # nothing was created for reconciliation to flag). It's recorded only in
    # each observation's own "mineru_structural_ref_resolution" metadata
    # with status "unresolved" -- the actual, verified signal to check.
    has_unresolved_cross_page_relation = any(
        entry.get("status") == "unresolved"
        and entry.get("relation_type") in _CROSS_PAGE_RELATION_TYPE_VALUES
        for observation in result.observations
        for entry in observation.metadata.get("mineru_structural_ref_resolution", ())
    )

    crop_store = LocalContentAddressedArtifactStore(
        Path(cfg.save_path) / "asset_cas", create=True
    )
    with _BIND_TIMINGS.time("crop_materialization"):
        reconciled_blocks, cas_crop_assets = materialize_visual_crops(
        tuple(reconciled.canonical_blocks), crop_store, cfg.save_path
    )
    with _BIND_TIMINGS.time("figure_enrichment"):
        reconciled_blocks = _enrich_figures(cfg, reconciled_blocks, crop_store)

    canonical_by_observation_id: dict[str, CanonicalBlock] = {}
    for canonical in reconciled_blocks:
        for observation_id in canonical.source_observation_ids:
            if observation_id in canonical_by_observation_id:
                raise CanonicalEvidenceBindingError(
                    "MinerU reconciliation assigned one observation to multiple "
                    "canonical blocks"
                )
            canonical_by_observation_id[observation_id] = canonical
    if set(canonical_by_observation_id) != {
        observation.observation_id for observation in result.observations
    }:
        raise CanonicalEvidenceBindingError(
            "MinerU reconciliation did not retain every parser observation"
        )

    canonical_blocks: list[CanonicalBlock] = []
    attachments: list[dict[str, object]] = []
    for block, native_id in zip(pdf_list, native_ids, strict=True):
        observation = observations_by_native_id.get(native_id)
        if observation is None:
            raise CanonicalEvidenceBindingError(
                f"MinerU block lost its parser observation: {native_id}"
            )
        reconciliation = block.get("_reconciliation")
        reconciliation_status = (
            reconciliation.get("status")
            if isinstance(reconciliation, dict)
            else None
        )
        requires_review = (
            isinstance(reconciliation_status, str)
            and reconciliation_status.startswith("unmatched_")
        )
        base_canonical = canonical_by_observation_id[observation.observation_id]
        canonical = replace(
            base_canonical,
            reconciliation_method=(
                f"mineru_page_reconciliation:{reconciliation_status}"
                if reconciliation_status
                else "mineru_page_reconciliation"
            ),
            quality_status=(
                QualityStatus.NEEDS_REVIEW
                if requires_review
                or base_canonical.quality_status == QualityStatus.NEEDS_REVIEW
                else QualityStatus.ACCEPTED
            ),
        )
        canonical_blocks.append(canonical)
        attachments.append(
            {
                "parser_native_id": native_id,
                "observation_id": observation.observation_id,
                "canonical_block_id": canonical.block_id,
                "canonical_revision_id": canonical.revision_id,
                "document_version_id": canonical.document_version_id,
                "source_sha256": request.document.content_sha256,
                "page_id": canonical.page_id,
                "geometry": (
                    canonical.geometry.to_dict() if canonical.geometry else None
                ),
                "semantic_subtype": canonical.semantic_subtype,
                "parser_provenance": observation.provenance.to_dict(),
                "quality_status": canonical.quality_status.value,
                "reconciliation_method": canonical.reconciliation_method,
                "source_observation_ids": list(canonical.source_observation_ids),
                "canonical_representation_ids": [
                    item.representation_id for item in canonical.representations
                ],
                "canonical_asset_ids": [item.asset_id for item in canonical.assets],
                "canonical_relations": [
                    relation.to_dict() for relation in canonical.relations
                ],
            }
        )

    crop_lease = (
        BookRAGAssetMaterializationLease(
            crop_store.make_asset_resolver(), cas_crop_assets
        )
        if cas_crop_assets
        else None
    )
    if crop_lease is not None:
        with crop_lease:
            projected_rows = list(
                project_to_bookrag_pdf_list(
                    request.document,
                    canonical_blocks,
                    asset_materialization_lease=crop_lease,
                )
            )
    else:
        projected_rows = list(
            project_to_bookrag_pdf_list(request.document, canonical_blocks)
        )
    projected_by_block_id = {
        row["canonical_block_id"]: row for row in projected_rows
    }
    for attachment in attachments:
        projected = projected_by_block_id.get(attachment["canonical_block_id"])
        if projected is None:
            raise CanonicalEvidenceBindingError(
                "MinerU canonical projection lost a reconciled block"
            )
        attachment["selected_representation_ids"] = list(
            projected.get("selected_representation_ids", [])
        )
        attachment["selected_asset_ids"] = list(
            projected.get("selected_asset_ids", [])
        )
        attachment["selected_representation_id"] = projected.get(
            "selected_representation_id"
        )
        attachment["selected_asset_id"] = projected.get("selected_asset_id")
        attachment["source_evidence_refs"] = projected.get(
            "source_evidence_refs"
        )
        attachment["semantic_substructures"] = projected.get(
            "semantic_substructures"
        )
        # The parser's native label is not always the tree's normalized type:
        # MinerU emits ``code_body`` and ``ref_text``, while the canonical
        # projection deliberately maps them to ``code`` and ``reference``.
        # Keeping the native label here makes create_node_by_type downgrade
        # both to generic TEXT even though canonical evidence is correctly
        # typed.
        attachment["type"] = projected.get("type", attachment.get("type"))
        # BookRAG builds its tree nodes from these legacy rows while the
        # evidence hydrator re-derives node content from the canonical
        # projection.  MinerU leaves a figure's caption list empty, so without
        # this the recovered text and description stay in the canonical layer
        # and every figure reaches retrieval as an empty node.
        # The adapter calls a block a heading on any of three signals -- the
        # content-list type, the middle-json type, or a text_level -- while the
        # tree only ever reads text_level. A heading MinerU declares but does
        # not level is then a title canonically and body text in the tree, and
        # the section it opens is lost. Carry the canonical decision across.
        if projected.get("text_level") is not None:
            attachment["text_level"] = projected["text_level"]
        if projected.get("is_document_title"):
            attachment["is_document_title"] = True

        if projected.get("type") in {"image", "chart", "diagram"}:
            attachment["image_caption"] = list(projected.get("image_caption", []))
            attachment["image_footnote"] = list(
                projected.get("image_footnote", [])
            )
        # Every text-bearing row must follow the canonical projection's
        # selection. MinerU list blocks carry ``list_items`` but no ``text``;
        # without this handoff they become empty LIST nodes even though their
        # canonical Markdown representation is populated. Equations likewise
        # may prefer a vision transcription over recognizer output.
        if projected.get("type") in {
            "text",
            "list",
            "code",
            "algorithm",
            "abstract",
            "reference",
            "equation",
        }:
            attachment["text"] = projected.get("text", "")
            attachment["text_format"] = projected.get("text_format", "")
        # A table row's channels must follow the projection for the same
        # reason: MinerU's raw caption and footnote strings differ from the
        # canonical ones in whitespace, and the hydrator compares exactly.
        if projected.get("type") == "table":
            attachment["table_caption"] = list(projected.get("table_caption", []))
            attachment["table_footnote"] = list(projected.get("table_footnote", []))
            attachment["table_body"] = projected.get("table_body", "")

    canonical_payload = {
        "artifact_schema_version": CANONICAL_EVIDENCE_SCHEMA_VERSION,
        "contract_schema_version": CONTRACT_SCHEMA_VERSION,
        "document": request.document.to_dict(),
        "parse_request_id": request.request_id,
        "parse_result_sha256": result.result_sha256,
        "parser_manifest": manifest.to_dict(),
        "block_count": len(canonical_blocks),
        "canonical_blocks": [item.to_dict() for item in canonical_blocks],
    }
    canonical_path = Path(cfg.save_path) / canonical_artifact_filename
    with _BIND_TIMINGS.time("canonical_write"):
        _atomic_write_json(canonical_path, canonical_payload)
    for block, attachment in zip(pdf_list, attachments, strict=True):
        block.update(attachment)
    log.info(
        "Bound %d MinerU blocks to canonical evidence at %s",
        len(canonical_blocks),
        canonical_path,
    )
    if has_unresolved_cross_page_relation:
        log.info(
            "shard has a relation whose target is outside this call's own "
            "blocks (table continuation or reference) -- a whole-document "
            "bind is needed for a fully correct result"
        )
    return request, tuple(canonical_blocks), has_unresolved_cross_page_relation


def merge_canonical_evidence_shards(
    cfg: SystemConfig,
    *,
    shard_fragment_filenames: list[str],
) -> Path:
    """Publish the one complete canonical-evidence artifact every existing
    reader expects, from per-shard fragments written by repeated
    ``bind_canonical_evidence`` calls.

    Every reader of ``canonical_blocks.json``
    (``evidence_projection.py``, ``evidence_persist.py``,
    ``docarag.provenance``, ``docarag.evidence_hydration``) expects exactly
    one file with this same shape -- this does not change that shape or add
    a field; it only changes how the file is produced. Each fragment is read
    (bounded to one shard's blocks), its blocks streamed straight into the
    output file, and discarded before the next fragment is opened -- peak
    memory is one shard's blocks, not the whole document's.

    Every fragment's ``document`` is identical (shard-invariant since the
    per-shard ``document_page_count`` fix). ``parse_request_id``/
    ``parse_result_sha256``/``parser_manifest`` are genuinely per-shard-call
    values (each shard has its own ``ParseRequest``/``ParseResult``), but
    nothing downstream depends on which shard's values end up in the merged
    artifact (confirmed: ``evidence_hydration.py`` only checks these fields
    are present in its known-fields allowlist, never their content) -- so
    this takes them from whichever fragment is opened first, requiring
    nothing extra from the caller beyond the list of fragment filenames.
    """

    if not shard_fragment_filenames:
        raise CanonicalEvidenceBindingError(
            "cannot merge canonical evidence from zero shards"
        )
    canonical_path = Path(cfg.save_path) / CANONICAL_EVIDENCE_FILE
    canonical_path.parent.mkdir(parents=True, exist_ok=True)

    def _encode(value: object) -> str:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

    block_count = 0
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{canonical_path.name}.",
            suffix=".tmp",
            dir=canonical_path.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            # Hand-assembled, not one json.dumps(whole_payload) call, so
            # canonical_blocks never has to exist as one in-memory list --
            # each fragment's blocks are read, written, and dropped before
            # the next fragment is opened. Every reader parses this as JSON
            # by key, not by relying on a particular key order, so writing
            # header fields before canonical_blocks (matching the field
            # order bind_canonical_evidence itself already writes) is a
            # readability choice, not a correctness requirement.
            temporary.write("{")
            header_written = False
            first_block = True
            for filename in shard_fragment_filenames:
                fragment_path = Path(cfg.save_path) / filename
                with open(fragment_path, "r", encoding="utf-8") as fragment_file:
                    fragment = json.load(fragment_file)
                if not header_written:
                    for key in (
                        "artifact_schema_version",
                        "contract_schema_version",
                        "document",
                        "parse_request_id",
                        "parse_result_sha256",
                        "parser_manifest",
                    ):
                        temporary.write(f"{_encode(key)}:{_encode(fragment[key])},")
                    temporary.write(f'{_encode("canonical_blocks")}:[')
                    header_written = True
                for block in fragment["canonical_blocks"]:
                    if not first_block:
                        temporary.write(",")
                    temporary.write(_encode(block))
                    first_block = False
                    block_count += 1
                del fragment
            temporary.write(f'],{_encode("block_count")}:{block_count}}}')
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, canonical_path)
        directory_fd = os.open(canonical_path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    log.info(
        "Merged %d canonical blocks from %d shard(s) into %s",
        block_count,
        len(shard_fragment_filenames),
        canonical_path,
    )
    return canonical_path


def _normalize_canonical_ingestion(value: object) -> CanonicalIngestionResult:
    """Copy a structural provider value into Core's strict boundary DTO."""

    missing = tuple(
        name
        for name in ("document", "canonical_blocks", "canonical_artifact_path")
        if not hasattr(value, name)
    )
    if missing:
        raise CanonicalIngestionBindingError(
            "canonical ingestion provider result is missing: " + ", ".join(missing)
        )
    try:
        blocks = tuple(getattr(value, "canonical_blocks"))
    except TypeError as exc:
        raise CanonicalIngestionBindingError(
            "canonical ingestion canonical_blocks must be iterable"
        ) from exc
    return CanonicalIngestionResult(
        document=getattr(value, "document"),
        canonical_blocks=blocks,
        canonical_artifact_path=getattr(value, "canonical_artifact_path"),
        canonical_artifact_sha256=getattr(
            value, "canonical_artifact_sha256", None
        ),
        canonical_artifact_size_bytes=getattr(
            value, "canonical_artifact_size_bytes", None
        ),
        audit_artifact_path=getattr(value, "audit_artifact_path", None),
        audit_artifact_sha256=getattr(value, "audit_artifact_sha256", None),
        audit_artifact_size_bytes=getattr(
            value, "audit_artifact_size_bytes", None
        ),
    )


def _ingest_canonical_provider(
    cfg: SystemConfig,
    provider: CanonicalIngestionProvider,
    *,
    reforce: bool,
    policy: CanonicalIngestionPolicy,
) -> tuple[list[dict[str, object]], CanonicalIngestionResult]:
    """Validate and project provider-owned evidence into the legacy boundary."""

    if not isinstance(policy, CanonicalIngestionPolicy):
        raise TypeError("ingestion_policy must be CanonicalIngestionPolicy")
    source = cfg.source
    if source is None or not source.sha256:
        raise CanonicalIngestionBindingError(
            "canonical ingestion requires cfg.source.sha256"
        )
    supplied = _normalize_canonical_ingestion(
        provider.ingest(cfg, reforce=reforce)
    )
    if supplied.document.content_sha256 != source.sha256:
        raise CanonicalIngestionBindingError(
            "canonical ingestion source SHA-256 differs from cfg.source.sha256"
        )
    disallowed = tuple(
        block
        for block in supplied.canonical_blocks
        if block.quality_status not in policy.allowed_quality_statuses
    )
    if disallowed:
        statuses = ", ".join(
            sorted({block.quality_status.value for block in disallowed})
        )
        raise CanonicalIngestionBindingError(
            "canonical ingestion contains quality statuses blocked by policy: "
            f"{statuses}"
        )
    pdf_list = project_to_bookrag_pdf_list(
        supplied.document,
        supplied.canonical_blocks,
    )
    if not pdf_list:
        raise CanonicalIngestionBindingError(
            "canonical ingestion projected no BookRAG content"
        )
    return pdf_list, supplied


def _validate_cached_tree_source(cfg: SystemConfig, tree: DocumentTree) -> None:
    """Fail closed when a cached tree is reused for another logical source."""

    source = cfg.source
    if source is None:
        return
    if not source.sha256:
        raise CanonicalIngestionBindingError(
            "cached tree reuse requires cfg.source.sha256"
        )
    expected = {
        "source_sha256": source.sha256,
        "source_uri": source.original_uri,
    }
    for label, meta in (
        ("tree", tree.meta_info),
        ("tree root", tree.root_node.meta_info if tree.root_node else None),
    ):
        if meta is None:
            raise CanonicalIngestionBindingError(
                f"cached {label} is missing source metadata"
            )
        for field_name, expected_value in expected.items():
            if getattr(meta, field_name, None) != expected_value:
                raise CanonicalIngestionBindingError(
                    f"cached {label} {field_name} differs from cfg.source"
                )


def _validate_cached_provider_tree(
    tree: DocumentTree,
    pdf_list: list[dict[str, object]],
    supplied: CanonicalIngestionResult,
) -> None:
    """Bind a reused tree to the provider's currently validated evidence."""

    if tree.meta_info.document_version_id != supplied.document.document_version_id:
        raise CanonicalIngestionBindingError(
            "cached tree belongs to another canonical document version"
        )
    try:
        cached_artifact = Path(tree.meta_info.canonical_artifact_path).resolve(
            strict=True
        )
        current_artifact = Path(supplied.canonical_artifact_path).resolve(strict=True)
    except (OSError, TypeError, ValueError) as exc:
        raise CanonicalIngestionBindingError(
            "cached tree canonical artifact is unavailable"
        ) from exc
    if cached_artifact != current_artifact:
        raise CanonicalIngestionBindingError(
            "cached tree references a different canonical artifact"
        )
    expected_canonical = {
        "canonical_artifact_sha256": supplied.canonical_artifact_sha256,
        "canonical_artifact_size_bytes": supplied.canonical_artifact_size_bytes,
    }
    for label, meta in (
        ("tree", tree.meta_info),
        ("tree root", tree.root_node.meta_info),
    ):
        for field_name, expected in expected_canonical.items():
            if getattr(meta, field_name, None) != expected:
                raise CanonicalIngestionBindingError(
                    f"cached {label} {field_name} differs from current provider"
                )
    for label, meta in (
        ("tree", tree.meta_info),
        ("tree root", tree.root_node.meta_info),
    ):
        expected_audit = {
            "routed_audit_artifact_path": supplied.audit_artifact_path,
            "routed_audit_artifact_sha256": supplied.audit_artifact_sha256,
            "routed_audit_artifact_size_bytes": supplied.audit_artifact_size_bytes,
        }
        for field_name, expected in expected_audit.items():
            if getattr(meta, field_name, None) != expected:
                raise CanonicalIngestionBindingError(
                    f"cached {label} {field_name} differs from current provider"
                )

    expected_revisions = {
        revision
        for row in pdf_list
        if isinstance(row, dict)
        for revision in (row.get("canonical_revision_id"),)
        if isinstance(revision, str) and revision
    }
    if len(expected_revisions) != len(pdf_list):
        raise CanonicalIngestionBindingError(
            "provider projection lacks one unique canonical revision per block"
        )
    observed_revisions: set[str] = set()
    for node in tree.get_nodes():
        meta = node.meta_info
        if meta.canonical_revision_id:
            observed_revisions.add(meta.canonical_revision_id)
        observed_revisions.update(meta.source_canonical_revision_ids or ())
        for reference in meta.source_evidence_refs or ():
            revision = reference.get("canonical_revision_id")
            if isinstance(revision, str) and revision:
                observed_revisions.add(revision)
    if observed_revisions != expected_revisions:
        raise CanonicalIngestionBindingError(
            "cached tree canonical lineage differs from the provider's current "
            "validated evidence"
        )


def construct_tree_index(
    tree_index: DocumentTree,
    pdf_list: list[dict],
    title_outline: list[dict],
    rejected_title_ids: Collection[int] = (),
) -> DocumentTree:
    """Attach every normalized source block exactly once in document order.

    ``pdf_list`` is the evidence-bearing source of truth.  The outline may only
    contribute hierarchy (``text_level`` and ``parent_id``); it cannot replace
    text, geometry, or canonical provenance copied from the source block.

    ``rejected_title_ids`` are the blocks the outline pass read and explicitly
    ruled out as section headings.  Without them the pass can only ever add
    headings, so a verdict of "this is not a heading" is silently discarded.

    Identity corruption is rejected because silently guessing would break
    traceability.  Semantic hierarchy errors are recoverable: a title with an
    invalid, missing, self, or forward parent is attached to the root and its
    following body blocks still remain under that title.  Documents with no
    detected titles become a shallow root tree instead of losing all content.

    The two indexed passes are O(B + T) time and O(B + T) auxiliary space for
    B blocks and T outline entries.
    """

    if tree_index.root_node is None:
        raise TreeConstructionError("tree construction requires a root node")
    if len(tree_index.nodes) != 1 or tree_index.root_node.children:
        raise TreeConstructionError(
            "tree construction requires a fresh tree containing only its root"
        )
    if tree_index.pdf_id_to_index_id:
        raise TreeConstructionError(
            "tree construction requires an empty PDF-ID index"
        )
    if not isinstance(pdf_list, list):
        raise TreeConstructionError("pdf_list must be a list")
    if not isinstance(title_outline, list):
        raise TreeConstructionError("title_outline must be a list")

    blocks_by_id: dict[int, dict] = {}
    ordered_block_ids: list[int] = []
    for position, block in enumerate(pdf_list, start=1):
        if not isinstance(block, dict):
            raise TreeConstructionError(
                f"pdf_list entry {position} must be a dictionary"
            )
        pdf_id = block.get("pdf_id")
        if isinstance(pdf_id, bool) or not isinstance(pdf_id, int) or pdf_id < 1:
            raise TreeConstructionError(
                f"pdf_list entry {position} has an invalid pdf_id: {pdf_id!r}"
            )
        if pdf_id in blocks_by_id:
            raise TreeConstructionError(f"duplicate pdf_id in pdf_list: {pdf_id}")
        blocks_by_id[pdf_id] = block
        ordered_block_ids.append(pdf_id)

    expected_ids = list(range(1, len(pdf_list) + 1))
    if ordered_block_ids != expected_ids:
        raise TreeConstructionError(
            "pdf_list IDs must be contiguous and ordered from 1 through "
            f"{len(pdf_list)}; received {ordered_block_ids!r}"
        )

    outline_by_id: dict[int, dict] = {}
    for position, outline_entry in enumerate(title_outline, start=1):
        if not isinstance(outline_entry, dict):
            raise TreeConstructionError(
                f"title_outline entry {position} must be a dictionary"
            )
        pdf_id = outline_entry.get("pdf_id")
        if isinstance(pdf_id, bool) or not isinstance(pdf_id, int) or pdf_id < 1:
            raise TreeConstructionError(
                "title_outline entry "
                f"{position} has an invalid pdf_id: {pdf_id!r}"
            )
        if pdf_id in outline_by_id:
            raise TreeConstructionError(
                f"duplicate pdf_id in title_outline: {pdf_id}"
            )
        if pdf_id not in blocks_by_id:
            raise TreeConstructionError(
                f"title_outline references missing source pdf_id: {pdf_id}"
            )
        outline_by_id[pdf_id] = outline_entry

    # A title must be a textual source block.  A malformed outline annotation
    # never gets to turn an image/table/equation into a synthetic title node.
    title_ids: set[int] = set()
    title_levels: dict[int, int] = {}

    # The parser's own headings are headings unless the outline pass read one
    # and said otherwise.  Seeding from the parser first means a document keeps
    # the structure MinerU already found -- previously a heading existed only if
    # the LLM independently renominated it, so a five-page chapter whose parser
    # reported 14 headings was built from the 4 the outline pass happened to
    # agree on, and one mid-document heading adopted three quarters of the
    # document.  Seeding *unconditionally* has the opposite failure: the pass
    # marked "↓ Correction" and "atmospheric refraction." invalid and both
    # headed sections anyway, the second dragging a later page's content into
    # an earlier page's section.  Only an explicit verdict overrides the parser;
    # a pass that stayed silent about a block leaves it exactly as it was.
    rejected = frozenset(rejected_title_ids)
    for pdf_id, source_block in blocks_by_id.items():
        if source_block.get("type", "unknown") not in _TITLE_CAPABLE_TYPES:
            continue
        parser_level = source_block.get("text_level")
        if isinstance(parser_level, bool) or not isinstance(parser_level, int):
            continue
        if parser_level < 0:
            continue
        if pdf_id in rejected:
            log.info(
                "Outline pass ruled out parser heading pdf_id=%d; keeping it as body text",
                pdf_id,
            )
            continue
        title_ids.add(pdf_id)
        title_levels[pdf_id] = parser_level

    for pdf_id, outline_entry in outline_by_id.items():
        if pdf_id in rejected:
            # A rejection and a hierarchy entry for the same block contradict
            # each other; the rejection is the narrower, later claim.
            log.warning(
                "Outline pass both rejected and levelled pdf_id=%d; honouring "
                "the rejection",
                pdf_id,
            )
            continue
        if blocks_by_id[pdf_id].get("type", "unknown") in _TITLE_CAPABLE_TYPES:
            title_ids.add(pdf_id)
            raw_level = outline_entry.get(
                "text_level", blocks_by_id[pdf_id].get("text_level", 0)
            )
            try:
                if isinstance(raw_level, bool):
                    raise ValueError
                text_level = int(raw_level)
                if text_level < 0:
                    raise ValueError
            except (TypeError, ValueError):
                text_level = 0
                log.warning(
                    "Invalid text_level=%r for title pdf_id=%d; using level 0",
                    raw_level,
                    pdf_id,
                )
            title_levels[pdf_id] = text_level
        else:
            log.warning(
                "Ignoring non-text title outline entry for pdf_id=%d type=%r",
                pdf_id,
                blocks_by_id[pdf_id].get("type"),
            )

    nodes_by_id = {}
    for pdf_id in ordered_block_ids:
        source_block = blocks_by_id[pdf_id]
        is_title = pdf_id in title_ids
        node_input = source_block
        if is_title:
            # Only structural fields may flow back from the LLM-created outline.
            # All source/evidence fields stay bound to the normalized block.
            node_input = dict(source_block)
            node_input["text_level"] = title_levels[pdf_id]
        node = create_node_by_type(pdf_content=node_input, isTitle=is_title)
        tree_index.add_node(node)
        nodes_by_id[pdf_id] = node

    attached_title_ids: set[int] = set()
    active_title_node = None
    # Open headings, shallowest first, for depth-based parenting.
    open_titles: list[tuple[int, object]] = []
    for pdf_id in ordered_block_ids:
        node = nodes_by_id[pdf_id]
        if pdf_id not in title_ids:
            parent = active_title_node or tree_index.root_node
            parent.add_child(node)
            continue

        outline_entry = outline_by_id.get(pdf_id)
        text_level = title_levels[pdf_id]

        if outline_entry is not None:
            # The outline pass claimed this heading, so its parent decision
            # stands -- including the documented fallback to root when that
            # decision is unusable. Inventing a different parent from a level
            # the same pass produced would not be more trustworthy.
            parent_node = tree_index.root_node
            raw_parent_id = outline_entry.get("parent_id")
            if text_level > 0 and raw_parent_id not in (None, 0):
                try:
                    if isinstance(raw_parent_id, bool):
                        raise ValueError
                    parent_id = int(raw_parent_id)
                except (TypeError, ValueError):
                    parent_id = -1
                if parent_id in attached_title_ids and parent_id < pdf_id:
                    parent_node = nodes_by_id[parent_id]
                else:
                    log.warning(
                        "Invalid title parent_id=%r for pdf_id=%d; attaching to root",
                        raw_parent_id,
                        pdf_id,
                    )
            elif text_level > 0:
                log.warning(
                    "Missing title parent for pdf_id=%d level=%d; attaching to root",
                    pdf_id,
                    text_level,
                )
        else:
            # A heading the parser found and the outline pass never mentioned.
            # It has no parent to inherit, so nest by depth: it belongs under
            # the nearest heading above it that is shallower than itself.
            # Attaching these to the root instead would flatten exactly the
            # structure the parser had already worked out.
            while open_titles and open_titles[-1][0] >= text_level:
                open_titles.pop()
            parent_node = open_titles[-1][1] if open_titles else tree_index.root_node

        parent_node.add_child(node)
        open_titles.append((text_level, node))
        attached_title_ids.add(pdf_id)
        active_title_node = node

    tree_index.max_depth = -1
    log.info(
        "Total %d source nodes plus root added to the tree index.",
        len(pdf_list),
    )
    return tree_index


def build_tree_from_pdf(
    cfg: SystemConfig,
    reforce: bool = False,
    *,
    parsing_provider: CanonicalIngestionProvider | None = None,
    ingestion_policy: CanonicalIngestionPolicy | None = None,
) -> DocumentTree:
    """Build the legacy BookRAG tree from MinerU or injected canonical evidence.

    ``parsing_provider=None`` preserves the historical MinerU/cache path.  An
    injected provider owns all parsing, routing, reconciliation, and artifact
    persistence; this function only validates source identity and projects its
    canonical blocks into the unchanged ``pdf_list`` pipeline.  Injected
    evidence is accepted-only unless ``ingestion_policy`` explicitly allows
    another quality status.
    """

    tree_index_path = DocumentTree.get_save_path(cfg.save_path)
    if os.path.exists(tree_index_path) and not reforce:
        source_sha256 = cfg.source.sha256 if cfg.source else None
        with durable_stage(
            "parse",
            payload={"routed_provider": parsing_provider is not None},
            source_sha256=source_sha256,
        ) as parse_stage:
            # Loading a completed immutable revision is still validated and
            # traced; the durable work item is observed as a replay.
            log.info(f"Loading existing tree index from {tree_index_path}...")
            tree_index = DocumentTree.load_from_file(tree_index_path)
            _validate_cached_tree_source(cfg, tree_index)
            if parsing_provider is not None:
                active_ingestion_policy = (
                    CanonicalIngestionPolicy()
                    if ingestion_policy is None
                    else ingestion_policy
                )
                pdf_list, supplied = _ingest_canonical_provider(
                    cfg,
                    parsing_provider,
                    reforce=False,
                    policy=active_ingestion_policy,
                )
                _validate_cached_provider_tree(
                    tree_index,
                    pdf_list,
                    supplied,
                )
            elif ingestion_policy is not None:
                raise ValueError(
                    "ingestion_policy applies only when parsing_provider is supplied"
                )
            canonical_path = tree_index.meta_info.canonical_artifact_path
            if canonical_path:
                parse_stage.record_artifact(
                    canonical_path,
                    artifact_role="canonical-ingestion",
                    artifact_schema=CANONICAL_EVIDENCE_SCHEMA_VERSION,
                    media_type="application/json",
                    source_sha256=tree_index.meta_info.source_sha256,
                )
            audit_path = tree_index.meta_info.routed_audit_artifact_path
            if audit_path:
                parse_stage.record_artifact(
                    audit_path,
                    artifact_role="routed-parsing-audit",
                    artifact_schema=ROUTED_PARSING_AUDIT_SCHEMA_VERSION,
                    media_type="application/json",
                    source_sha256=tree_index.meta_info.source_sha256,
                )
            parse_stage.checkpoint("cache.validated")
        with durable_stage(
            "tree",
            payload={"use_vlm": bool(cfg.tree.use_vlm)},
            source_sha256=tree_index.meta_info.source_sha256,
        ) as tree_stage:
            tree_stage.checkpoint(
                "cache.validated", {"node_count": len(tree_index.nodes)}
            )
        with durable_stage(
            "summary",
            payload={"enabled": bool(cfg.tree.node_summary)},
            source_sha256=tree_index.meta_info.source_sha256,
        ) as summary_stage:
            summary_stage.record_artifact(
                tree_index_path,
                artifact_role="bookrag-tree",
                artifact_schema="bookrag.document-tree.pickle.v1",
                media_type="application/x-python-pickle",
                source_sha256=tree_index.meta_info.source_sha256,
                metadata={"node_count": len(tree_index.nodes)},
            )
            tree_json_path = Path(cfg.save_path) / "tree.json"
            if tree_json_path.is_file():
                summary_stage.record_artifact(
                    tree_json_path,
                    artifact_role="bookrag-tree-summary",
                    artifact_schema="bookrag.document-tree-summary.v1",
                    artifact_key="json",
                    media_type="application/json",
                    source_sha256=tree_index.meta_info.source_sha256,
                    metadata={"node_count": len(tree_index.nodes)},
                )
        log.info("Tree index loaded successfully.")
        return tree_index
    else:
        # Create a new tree index
        log.info("Creating a new tree index...")

    source = cfg.source
    meta_dict = {
        "file_name": source.display_name if source else os.path.basename(cfg.pdf_path),
        "file_path": source.original_uri if source else cfg.pdf_path,
        "source_uri": source.original_uri if source else cfg.pdf_path,
        "source_sha256": source.sha256 if source else None,
        "snapshot_path": source.snapshot_path if source else cfg.pdf_path,
    }

    os.makedirs(cfg.save_path, exist_ok=True)

    tree_index = DocumentTree(meta_dict=meta_dict, cfg=cfg)
    timings = PhaseTimings("tree stage")

    source_sha256 = cfg.source.sha256 if cfg.source else None
    with durable_stage(
        "parse",
        payload={"routed_provider": parsing_provider is not None},
        source_sha256=source_sha256,
    ) as parse_stage:
        if parsing_provider is not None:
            active_ingestion_policy = (
                CanonicalIngestionPolicy()
                if ingestion_policy is None
                else ingestion_policy
            )
            pdf_list, supplied = _ingest_canonical_provider(
                cfg,
                parsing_provider,
                reforce=reforce,
                policy=active_ingestion_policy,
            )
            document = supplied.document
            canonical_path = str(supplied.canonical_artifact_path)
            canonical_artifact_sha256 = supplied.canonical_artifact_sha256
            canonical_artifact_size_bytes = supplied.canonical_artifact_size_bytes
            audit_artifact_path = supplied.audit_artifact_path
            audit_artifact_sha256 = supplied.audit_artifact_sha256
            audit_artifact_size_bytes = supplied.audit_artifact_size_bytes
            log.info(
                "Projected %d canonical blocks from injected provider %s",
                len(pdf_list),
                type(parsing_provider).__name__,
            )
        else:
            if ingestion_policy is not None:
                raise ValueError(
                    "ingestion_policy applies only when parsing_provider is supplied"
                )
            backend = cfg.mineru.backend
            server_url = cfg.mineru.server_url
            method = cfg.mineru.method
            base_file_name = Path(cfg.pdf_path).stem
            tmp_save_path = os.path.join(
                cfg.save_path, method, f"{base_file_name}_merged_content.json"
            )

            if os.path.exists(tmp_save_path) and not reforce:
                # tmp load pdf_list
                with open(tmp_save_path, "rb") as f:
                    pdf_list = json.load(f)
                print(f"Loaded content from {tmp_save_path}")
            else:
                # Extract content one bounded page-range shard at a time. A
                # whole document parsed in a single MinerU call renders, OCRs
                # and layout-analyzes every page at once -- that is what
                # drives peak memory for a large PDF, and it is bounded here
                # to cfg.mineru.shard_pages pages regardless of document size.
                log.info(f"Extracting content from {cfg.pdf_path}...")
                file_name = str(Path(cfg.pdf_path).stem)
                save_dir = os.path.join(cfg.save_path, method)
                page_count = pdf_page_count(cfg.pdf_path)
                shards = list(shard_ranges(page_count, cfg.mineru.shard_pages))
                pdf_list = []
                transcription_pages: list[dict] = []
                transcription_failures: list[str] = []
                for shard_index, (start_page_id, end_page_id) in enumerate(shards):
                    with timings.time("mineru_parse"):
                        middle_json, content_list = parse_doc(
                            cfg.pdf_path,
                            output_dir=cfg.save_path,
                            backend=backend,
                            method=method,
                            server_url=server_url,
                            lang=cfg.mineru.lang,
                            effort=cfg.mineru.effort,
                            start_page_id=start_page_id,
                            end_page_id=end_page_id,
                            # Without this, do_parse's own .md/_middle.json/
                            # _content_list.json debug dumps use the same
                            # document-derived name every shard and each call
                            # silently overwrites the previous shard's dump.
                            pdf_file_name=f"{file_name}_shard{shard_index:03d}",
                        )
                    middle_json, content_list = offset_page_indices(
                        middle_json, content_list, start_page_id
                    )
                    # reconcile_middle_content matches middle_json against
                    # content_list one page at a time with no cross-page
                    # state (confirmed by reading it), so reconciling shard
                    # by shard produces the same result as reconciling the
                    # whole document at once.
                    with timings.time("merge_content"):
                        shard_pdf_list = merge_middle_content(
                            middle_json,
                            content_list,
                            parse_dir=save_dir,
                            save_dir=save_dir,
                            file_name=f"{file_name}_shard{shard_index:03d}",
                        )
                    # Re-read scanned pages the recogniser got wrong. After
                    # reconciliation, never before: the two parser views are
                    # matched by text, so rewriting one and not the other would
                    # stop them matching and emit every block on the page twice.
                    if cfg.mineru.retranscribe_scanned:
                        with timings.time("page_transcription"):
                            outcome = retranscribe_scanned_pages(
                                shard_pdf_list,
                                pdf_path=cfg.pdf_path,
                                model=cfg.vlm.model_name,
                                api_base=cfg.vlm.api_base,
                                min_validity=cfg.mineru.retranscribe_min_validity,
                            )
                        transcription_pages.extend(outcome.pages)
                        transcription_failures.extend(outcome.failures)
                        if outcome.pages_transcribed:
                            log.info(
                                "shard %d: re-transcribed %d/%d pages, "
                                "%d blocks replaced, %d left as parsed",
                                shard_index, outcome.pages_transcribed,
                                outcome.pages_examined, outcome.blocks_replaced,
                                outcome.blocks_unmatched,
                            )
                    pdf_list.extend(shard_pdf_list)
                    log.info(
                        "shard %d/%d (pages %d-%d): %d items (%d total so far)",
                        shard_index + 1, len(shards), start_page_id, end_page_id,
                        len(shard_pdf_list), len(pdf_list),
                    )

                os.makedirs(save_dir, exist_ok=True)
                with open(tmp_save_path, "w", encoding="utf-8") as f:
                    json.dump(pdf_list, f, ensure_ascii=False, indent=4)

                # tmp pdf_list save for fast test
                log.info(f"Content extracted and saved to {tmp_save_path}")

            pdf_list, recovered_blocks, dropped_blocks = (
                recover_and_prune_empty_text_blocks(cfg.pdf_path, pdf_list)
            )
            if recovered_blocks or dropped_blocks:
                log.info(
                    "Native text-layer recovery: recovered %d, dropped %d empty block(s)",
                    recovered_blocks,
                    dropped_blocks,
                )

            # Canonical evidence binding -- translation, reconciliation, crop
            # materialization, VLM figure enrichment -- is the second-largest
            # memory cost after MinerU's own rendering (bounded above). Bind
            # it in the same page-range shards parsing already used: one
            # shard's blocks, crops and VLM calls in memory at a time,
            # instead of the whole document's. recover_and_prune_empty_text_blocks
            # still needs pdf_list whole (it's a lightweight text match, not
            # a memory concern) so blocks are regrouped by page_idx back into
            # the same shard boundaries afterward.
            #
            # The page count comes from pdf_list itself (every block already
            # carries its page_idx, and pdf_list is the whole document's
            # here, whether freshly parsed or loaded from the merged-content
            # cache), not from reopening cfg.pdf_path -- the cache-hit branch
            # above intentionally never touches the source file again once
            # parsing has completed once.
            canonical_page_count = page_count_from_blocks(pdf_list)
            canonical_shard_ranges = list(
                shard_ranges(canonical_page_count, cfg.mineru.shard_pages)
            )
            canonical_shard_fragments: list[str] = []
            parse_request = None
            needs_whole_document_fallback = False
            for shard_index, shard_blocks in enumerate(
                partition_blocks_by_shard(pdf_list, canonical_shard_ranges)
            ):
                if not shard_blocks:
                    continue
                fragment_filename = f"canonical_blocks_shard{shard_index:03d}.json"
                with timings.time("canonical_bind"):
                    (
                        parse_request,
                        _shard_canonical,
                        shard_has_cross_page_gap,
                    ) = bind_canonical_evidence(
                        cfg,
                        shard_blocks,
                        merged_content_path=tmp_save_path,
                        document_page_count=canonical_page_count,
                        debug_artifact_suffix=f"_shard{shard_index:03d}",
                        canonical_artifact_filename=fragment_filename,
                    )
                canonical_shard_fragments.append(fragment_filename)
                needs_whole_document_fallback = (
                    needs_whole_document_fallback or shard_has_cross_page_gap
                )

            if needs_whole_document_fallback:
                # Rare: a table continuation or explicit reference has its
                # target in a different shard than its source, so shard-local
                # reconciliation couldn't resolve it -- confirmed by
                # bind_canonical_evidence's own issue list, not guessed.
                # Falling back to one whole-document call (the pre-sharding,
                # already-proven-correct path) costs the memory bound for
                # this document -- and, since crop/VLM work isn't cached
                # across calls, repeats that work -- but a correct answer
                # matters more than a shard-scoped guess for a case this
                # narrow. merge_canonical_evidence_shards is skipped: this
                # call publishes canonical_blocks.json directly, superseding
                # the (now-stale, left on disk as ordinary debug fragments)
                # per-shard files written above.
                log.info(
                    "canonical evidence has a relation spanning shard "
                    "boundaries; falling back to one whole-document bind "
                    "for full correctness"
                )
                with timings.time("canonical_bind_whole_document"):
                    parse_request, _canonical, _ = bind_canonical_evidence(
                        cfg,
                        pdf_list,
                        merged_content_path=tmp_save_path,
                        document_page_count=canonical_page_count,
                    )
            else:
                merge_canonical_evidence_shards(
                    cfg, shard_fragment_filenames=canonical_shard_fragments
                )
            document = parse_request.document
            (
                canonical_artifact_path,
                canonical_artifact_sha256,
                canonical_artifact_size_bytes,
            ) = _stable_local_artifact_receipt(
                Path(cfg.save_path) / CANONICAL_EVIDENCE_FILE,
                label="canonical evidence",
            )
            canonical_path = str(canonical_artifact_path)
            audit_artifact_path = None
            audit_artifact_sha256 = None
            audit_artifact_size_bytes = None

        parse_stage.record_artifact(
            canonical_path,
            artifact_role="canonical-ingestion",
            artifact_schema=CANONICAL_EVIDENCE_SCHEMA_VERSION,
            media_type="application/json",
            source_sha256=document.content_sha256,
        )
        if audit_artifact_path:
            parse_stage.record_artifact(
                audit_artifact_path,
                artifact_role="routed-parsing-audit",
                artifact_schema=ROUTED_PARSING_AUDIT_SCHEMA_VERSION,
                media_type="application/json",
                source_sha256=document.content_sha256,
            )
        else:
            parse_stage.record_not_emitted(
                artifact_role="routed-parsing-audit",
                artifact_schema=ROUTED_PARSING_AUDIT_SCHEMA_VERSION,
                media_type="application/json",
                source_sha256=document.content_sha256,
                metadata={"reason": "legacy-single-parser-ingestion"},
            )

    for tree_meta in (tree_index.meta_info, tree_index.root_node.meta_info):
        tree_meta.file_name = document.display_name
        tree_meta.source_uri = document.source_uri
        tree_meta.source_sha256 = document.content_sha256
        tree_meta.document_version_id = document.document_version_id
        tree_meta.canonical_artifact_path = canonical_path
        tree_meta.canonical_artifact_sha256 = canonical_artifact_sha256
        tree_meta.canonical_artifact_size_bytes = canonical_artifact_size_bytes
        tree_meta.routed_audit_artifact_path = audit_artifact_path
        tree_meta.routed_audit_artifact_sha256 = audit_artifact_sha256
        tree_meta.routed_audit_artifact_size_bytes = audit_artifact_size_bytes

    llm = LLM(indexing_llm_config(cfg.llm))
    vlm = VLM(cfg.vlm) if cfg.tree.use_vlm else None

    with durable_stage(
        "tree",
        payload={"use_vlm": bool(cfg.tree.use_vlm)},
        source_sha256=document.content_sha256,
    ) as tree_stage:
        with timings.time("layout_recheck"):
            pdf_list = pdf_info_refiner(pdf_list, llm)
        with timings.time("outline"):
            title_outline, rejected_title_ids = extract_pdf_outline_in_chunks(
                pdf_list, llm
            )
        with timings.time("construct"):
            tree_index = construct_tree_index(
                tree_index=tree_index,
                pdf_list=pdf_list,
                title_outline=title_outline,
                rejected_title_ids=rejected_title_ids,
            )
        token_tracker = TokenTracker.get_instance()
        tree_index_cost = token_tracker.record_stage("tree_index_construction")
        log.info(f"Tree index construction cost: {tree_index_cost}")
        tree_stage.checkpoint(
            "structure.constructed", {"node_count": len(tree_index.nodes)}
        )

    with durable_stage(
        "summary",
        payload={"enabled": bool(cfg.tree.node_summary)},
        source_sha256=document.content_sha256,
    ) as summary_stage:
        if cfg.tree.node_summary:
            # Generate summaries for each node
            with timings.time("summary"):
                tree_index = generate_tree_node_summary(
                    tree_index=tree_index,
                    llm=llm,
                    use_VLM=cfg.tree.use_vlm,
                    vlm=vlm,
                )
            token_tracker = TokenTracker.get_instance()
            summary_cost = token_tracker.record_stage("tree_node_summary")
            log.info(f"Tree node summary generation cost: {summary_cost}")
            summary_stage.checkpoint(
                "nodes.summarized", {"node_count": len(tree_index.nodes)}
            )
        else:
            summary_stage.checkpoint("disabled")

        # The final tree is published only after all requested summaries exist.
        tree_index.save_to_file()
        summary_stage.record_artifact(
            tree_index_path,
            artifact_role="bookrag-tree",
            artifact_schema="bookrag.document-tree.pickle.v1",
            media_type="application/x-python-pickle",
            source_sha256=document.content_sha256,
            metadata={"node_count": len(tree_index.nodes)},
        )
        summary_stage.record_artifact(
            Path(cfg.save_path) / "tree.json",
            artifact_role="bookrag-tree-summary",
            artifact_schema="bookrag.document-tree-summary.v1",
            artifact_key="json",
            media_type="application/json",
            source_sha256=document.content_sha256,
            metadata={"node_count": len(tree_index.nodes)},
        )
    timings.log_summary()
    _BIND_TIMINGS.log_summary()
    return tree_index
