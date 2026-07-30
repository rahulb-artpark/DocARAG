"""Recover the evidence baked into figure rasters.

MinerU treats an image region as an atomic picture: it crops the region and
emits a single ``image`` span, discarding the text its own OCR detected inside.
Rendered tables, charts and labelled diagrams therefore reach the canonical
layer carrying no textual evidence at all.  This re-reads the verified crop and
attaches what it finds: exact strings from an OCR engine, and a description
from a vision model grounded on those strings.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
import logging

from docarag.asset_material import AssetMaterialResolver, AssetMaterializationError
from docarag.domain.base import ContractValidationError
from docarag.domain.evidence import (
    AssetKind,
    AssetRef,
    BlockType,
    CanonicalBlock,
    Representation,
    RepresentationKind,
    SemanticSubstructure,
)
from docarag.figure_diagram import DiagramExtractor, diagram_substructure

log = logging.getLogger(__name__)

MAX_OCR_TEXT_CHARS = 16_000
MAX_DESCRIPTION_CHARS = 8_000


@dataclass(frozen=True, slots=True)
class OCRSpan:
    """One recognized text run in crop-pixel coordinates."""

    text: str
    confidence: float
    bbox: tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class FigureOCRResult:
    spans: tuple[OCRSpan, ...]
    width: int
    height: int


FigureOCREngine = Callable[[bytes], FigureOCRResult]
FigureDescriber = Callable[[bytes, str], str | None]


@dataclass(frozen=True, slots=True)
class FigureEnrichmentStats:
    visual_blocks: int = 0
    ocr_blocks: int = 0
    ocr_spans: int = 0
    described_blocks: int = 0
    graph_blocks: int = 0
    graph_nodes: int = 0
    graph_edges: int = 0
    skipped_blocks: int = 0
    failed_blocks: int = 0


VISUAL_BLOCK_TYPES = frozenset(
    {BlockType.FIGURE, BlockType.CHART, BlockType.DIAGRAM}
)
_IMAGE_ASSET_KINDS = frozenset(
    {AssetKind.REGION_CROP, AssetKind.EXTRACTED_IMAGE}
)


def _verified_image_asset(block: CanonicalBlock) -> AssetRef | None:
    for asset in block.assets:
        if not isinstance(asset.uri, str) or not asset.uri.startswith("cas://"):
            continue
        media_type = asset.media_type or ""
        if asset.kind in _IMAGE_ASSET_KINDS or media_type.startswith("image/"):
            return asset
    return None


def _already_enriched(block: CanonicalBlock) -> bool:
    """True only for evidence produced by this enrichment pass.

    MinerU 3.4.x may emit its own short ``VISUAL_DESCRIPTION``. That is useful
    parser evidence, but it is not the richer Qwen description requested by
    ``figures.describe``. Treating any visual description as an idempotence
    marker skipped the VLM entirely for every image in a document.
    """

    return any(
        (
            representation.kind is RepresentationKind.OCR_TEXT
            and representation.metadata.get("producer")
            == "mineru.pytorch_paddle_ocr"
        )
        or (
            representation.kind is RepresentationKind.VISUAL_DESCRIPTION
            and representation.metadata.get("producer") == "vlm"
        )
        for representation in block.representations
    )


def _extract_graph(
    block: CanonicalBlock,
    asset: AssetRef,
    data: bytes,
    text: str,
    extractor: DiagramExtractor,
) -> SemanticSubstructure | None:
    try:
        graph = extractor(data, text)
        if graph is None:
            return None
        return diagram_substructure(
            graph, asset=asset, source_observation_ids=block.source_observation_ids
        )
    except (ContractValidationError, OSError, ValueError, RuntimeError):
        log.warning(
            "figure graph extraction failed for block %s",
            block.block_id,
            exc_info=True,
        )
        return None


def _reading_order(spans: Sequence[OCRSpan], height: int) -> list[OCRSpan]:
    band = max(height / 64.0, 1.0)
    return sorted(spans, key=lambda span: (int(span.bbox[1] / band), span.bbox[0]))


def _lines(spans: Sequence[OCRSpan], height: int) -> str:
    band = max(height / 64.0, 1.0)
    rows: list[list[OCRSpan]] = []
    current_key: int | None = None
    for span in _reading_order(spans, height):
        key = int(span.bbox[1] / band)
        if current_key is None or key != current_key:
            rows.append([span])
            current_key = key
        else:
            rows[-1].append(span)
    return "\n".join(
        " ".join(span.text for span in row).strip() for row in rows
    ).strip()


def enrich_visual_blocks(
    blocks: Sequence[CanonicalBlock],
    *,
    resolver: AssetMaterialResolver,
    ocr_engine: FigureOCREngine,
    describer: FigureDescriber | None = None,
    diagram_extractor: DiagramExtractor | None = None,
    min_confidence: float = 0.5,
    min_chars: int = 2,
) -> tuple[tuple[CanonicalBlock, ...], FigureEnrichmentStats]:
    """Return blocks whose figures carry recovered text, meaning and topology."""

    visual = ocr_done = spans_total = described = skipped = failed = 0
    graphs = graph_nodes = graph_edges = 0
    enriched: list[CanonicalBlock] = []
    for block in blocks:
        if block.block_type not in VISUAL_BLOCK_TYPES:
            enriched.append(block)
            continue
        visual += 1
        asset = _verified_image_asset(block)
        if asset is None or _already_enriched(block):
            skipped += 1
            enriched.append(block)
            continue
        try:
            material = resolver.resolve(asset, include_bytes=True)
            data = material.data
            if not data:
                raise AssetMaterializationError(
                    f"verified asset {asset.asset_id} yielded no bytes"
                )
            result = ocr_engine(data)
        except (AssetMaterializationError, OSError, ValueError, RuntimeError):
            log.warning(
                "figure OCR failed for block %s", block.block_id, exc_info=True
            )
            failed += 1
            enriched.append(block)
            continue

        kept = [
            span
            for span in result.spans
            if span.confidence >= min_confidence
            and len(span.text.strip()) >= min_chars
        ]
        text = _lines(kept, result.height)[:MAX_OCR_TEXT_CHARS] if kept else ""
        added: list[Representation] = []
        if text:
            confidence = sum(span.confidence for span in kept) / len(kept)
            added.append(
                Representation(
                    kind=RepresentationKind.OCR_TEXT,
                    content=text,
                    media_type="text/plain",
                    confidence=round(min(confidence, 1.0), 4),
                    metadata={
                        "producer": "mineru.pytorch_paddle_ocr",
                        "span_count": len(kept),
                        "asset_sha256": asset.sha256,
                    },
                )
            )
            ocr_done += 1
            spans_total += len(kept)

        if describer is not None:
            try:
                description = describer(data, text)
            except (OSError, ValueError, RuntimeError):
                log.warning(
                    "figure description failed for block %s",
                    block.block_id,
                    exc_info=True,
                )
                description = None
            if description and description.strip():
                added.append(
                    Representation(
                        kind=RepresentationKind.VISUAL_DESCRIPTION,
                        content=description.strip()[:MAX_DESCRIPTION_CHARS],
                        media_type="text/plain",
                        metadata={
                            "producer": "vlm",
                            "grounded_by_ocr": bool(text),
                            "asset_sha256": asset.sha256,
                        },
                    )
                )
                described += 1

        substructure = (
            None
            if diagram_extractor is None
            else _extract_graph(block, asset, data, text, diagram_extractor)
        )
        if substructure is not None:
            graphs += 1
            graph_nodes += len(substructure.elements)
            graph_edges += len(substructure.relations)

        if not added and substructure is None:
            skipped += 1
            enriched.append(block)
            continue
        enriched.append(
            replace(
                block,
                representations=tuple(block.representations) + tuple(added),
                semantic_substructures=(
                    tuple(block.semantic_substructures)
                    if substructure is None
                    else tuple(block.semantic_substructures) + (substructure,)
                ),
            )
        )

    stats = FigureEnrichmentStats(
        visual_blocks=visual,
        ocr_blocks=ocr_done,
        ocr_spans=spans_total,
        described_blocks=described,
        graph_blocks=graphs,
        graph_nodes=graph_nodes,
        graph_edges=graph_edges,
        skipped_blocks=skipped,
        failed_blocks=failed,
    )
    return tuple(enriched), stats


__all__ = [
    "FigureDescriber",
    "FigureEnrichmentStats",
    "FigureOCREngine",
    "FigureOCRResult",
    "OCRSpan",
    "enrich_visual_blocks",
]
