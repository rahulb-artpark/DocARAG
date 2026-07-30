from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from docarag.artifact_store import LocalContentAddressedArtifactStore
from docarag.bookrag_projection import (
    BookRAGAssetMaterializationLease,
    project_canonical_blocks,
)
from docarag.domain import (
    AssetKind,
    BlockType,
    BoundingBox,
    CanonicalBlock,
    CoordinateSpace,
    DocumentRef,
    PageRef,
    QualityStatus,
    Representation,
    RepresentationKind,
    SourceGeometry,
)
from docarag.figure_describe import GroundedFigureDescriber
from docarag.figure_enrichment import (
    FigureOCRResult,
    OCRSpan,
    enrich_visual_blocks,
)

SOURCE_SHA256 = "b" * 64
CROP_BYTES = b"\x89PNG\r\n\x1a\n rendered-table-crop"


def make_document() -> DocumentRef:
    return DocumentRef.identified(
        content_sha256=SOURCE_SHA256,
        mime_type="application/pdf",
        source_uri="s3://truth/deck.pdf",
        display_name="deck.pdf",
        page_count=2,
    )


def make_block(
    document: DocumentRef,
    key: str,
    *,
    block_type: BlockType = BlockType.FIGURE,
    representations: tuple[Representation, ...] = (),
    assets: tuple = (),
) -> CanonicalBlock:
    return CanonicalBlock(
        canonical_key=key,
        revision=1,
        document_version_id=document.document_version_id,
        page_id=PageRef(
            document_version_id=document.document_version_id, page_index=0
        ).page_id,
        page_index=0,
        block_type=block_type,
        geometry=SourceGeometry(
            coordinate_space=CoordinateSpace.PAGE_POINTS,
            bbox=BoundingBox(10.0, 20.0, 300.0, 180.0),
            page_width=612.0,
            page_height=792.0,
        ),
        reading_order=1.0,
        representations=representations,
        assets=assets,
        source_observation_ids=(f"observation-{key}",),
        reconciliation_method="fixture",
        quality_status=QualityStatus.ACCEPTED,
    )


def table_spans() -> FigureOCRResult:
    return FigureOCRResult(
        spans=(
            OCRSpan(text="Dimension", confidence=1.0, bbox=(0, 0, 60, 10)),
            OCRSpan(text="Records", confidence=1.0, bbox=(80, 0, 130, 10)),
            OCRSpan(text="Purchase Orders", confidence=0.97, bbox=(0, 40, 70, 50)),
            OCRSpan(text="126", confidence=0.99, bbox=(80, 40, 110, 50)),
        ),
        width=200,
        height=64,
    )


def representation(kind: RepresentationKind, text: str) -> Representation:
    return Representation(kind=kind, content=text, media_type="text/plain")


class RecordingDescriber:
    def __init__(self, reply: str | None = "A table of purchase orders.") -> None:
        self.reply = reply
        self.calls: list[tuple[bytes, str]] = []

    def __call__(self, data: bytes, ocr_text: str) -> str | None:
        self.calls.append((data, ocr_text))
        return self.reply


class FigureEnrichmentTest(unittest.TestCase):
    def setUp(self) -> None:
        self.document = make_document()
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.store = LocalContentAddressedArtifactStore(
            Path(self._tmp.name) / "asset_cas", create=True
        )
        stored = self.store.publish(CROP_BYTES, media_type="image/png")
        self.asset = stored.to_asset_ref(kind=AssetKind.REGION_CROP)
        self.resolver = self.store.make_asset_resolver()

    def enrich(self, blocks, *, ocr_engine=None, describer=None, **kwargs):
        return enrich_visual_blocks(
            blocks,
            resolver=self.resolver,
            ocr_engine=ocr_engine or (lambda _data: table_spans()),
            describer=describer,
            **kwargs,
        )

    def test_rendered_table_gains_exact_ocr_text_in_reading_order(self) -> None:
        block = make_block(self.document, "figure-1", assets=(self.asset,))
        enriched, stats = self.enrich((block,))

        texts = {
            item.kind: item.content for item in enriched[0].representations
        }
        self.assertEqual(
            texts[RepresentationKind.OCR_TEXT],
            "Dimension Records\nPurchase Orders 126",
        )
        self.assertEqual(stats.ocr_blocks, 1)
        self.assertEqual(stats.ocr_spans, 4)
        self.assertEqual(stats.failed_blocks, 0)

    def test_ocr_text_records_producer_and_asset_digest(self) -> None:
        block = make_block(self.document, "figure-1", assets=(self.asset,))
        enriched, _ = self.enrich((block,))

        found = next(
            item
            for item in enriched[0].representations
            if item.kind is RepresentationKind.OCR_TEXT
        )
        self.assertEqual(found.metadata["producer"], "mineru.pytorch_paddle_ocr")
        self.assertEqual(found.metadata["asset_sha256"], self.asset.sha256)
        self.assertEqual(found.metadata["span_count"], 4)

    def test_describer_is_grounded_on_recovered_text(self) -> None:
        block = make_block(self.document, "figure-1", assets=(self.asset,))
        describer = RecordingDescriber()
        enriched, stats = self.enrich((block,), describer=describer)

        self.assertEqual(len(describer.calls), 1)
        data, ocr_text = describer.calls[0]
        self.assertEqual(data, CROP_BYTES)
        self.assertIn("Purchase Orders 126", ocr_text)
        self.assertEqual(stats.described_blocks, 1)
        kinds = {item.kind for item in enriched[0].representations}
        self.assertIn(RepresentationKind.VISUAL_DESCRIPTION, kinds)
        self.assertIn(RepresentationKind.OCR_TEXT, kinds)

    def test_mineru_description_does_not_suppress_richer_vlm_enrichment(self) -> None:
        parser_description = Representation(
            kind=RepresentationKind.VISUAL_DESCRIPTION,
            content="Labels: cornea, retina, iris.",
            media_type="text/plain",
            metadata={"producer": "mineru", "source_field": "image_content"},
        )
        block = make_block(
            self.document,
            "figure-1",
            representations=(parser_description,),
            assets=(self.asset,),
        )
        describer = RecordingDescriber(
            "A labelled cross-section of the human eye showing its optical path."
        )

        enriched, stats = self.enrich((block,), describer=describer)

        self.assertEqual(1, len(describer.calls))
        self.assertEqual(1, stats.described_blocks)
        descriptions = [
            item
            for item in enriched[0].representations
            if item.kind is RepresentationKind.VISUAL_DESCRIPTION
        ]
        self.assertEqual(2, len(descriptions))
        self.assertEqual("vlm", descriptions[-1].metadata["producer"])

    def test_photograph_without_text_is_described_but_gains_no_ocr_text(self) -> None:
        block = make_block(self.document, "photo-1", assets=(self.asset,))
        describer = RecordingDescriber(reply="A photograph of a factory floor.")
        enriched, stats = self.enrich(
            (block,),
            ocr_engine=lambda _data: FigureOCRResult(spans=(), width=64, height=64),
            describer=describer,
        )

        kinds = {item.kind for item in enriched[0].representations}
        self.assertNotIn(RepresentationKind.OCR_TEXT, kinds)
        self.assertIn(RepresentationKind.VISUAL_DESCRIPTION, kinds)
        self.assertEqual(describer.calls[0][1], "")
        self.assertEqual(stats.ocr_blocks, 0)
        self.assertEqual(stats.described_blocks, 1)

    def test_low_confidence_and_single_character_noise_is_discarded(self) -> None:
        block = make_block(self.document, "photo-1", assets=(self.asset,))
        noisy = FigureOCRResult(
            spans=(
                OCRSpan(text="Z", confidence=0.99, bbox=(0, 0, 5, 5)),
                OCRSpan(text="blurred", confidence=0.12, bbox=(0, 20, 40, 30)),
            ),
            width=64,
            height=64,
        )
        enriched, stats = self.enrich((block,), ocr_engine=lambda _data: noisy)

        self.assertEqual(enriched[0].representations, ())
        self.assertEqual(stats.ocr_blocks, 0)
        self.assertEqual(stats.skipped_blocks, 1)

    def test_non_visual_blocks_are_untouched(self) -> None:
        block = make_block(
            self.document,
            "paragraph-1",
            block_type=BlockType.PARAGRAPH,
            representations=(representation(RepresentationKind.TEXT, "Body text."),),
            assets=(self.asset,),
        )
        enriched, stats = self.enrich((block,))

        self.assertEqual(enriched[0], block)
        self.assertEqual(stats.visual_blocks, 0)

    def test_figure_without_verified_asset_is_skipped(self) -> None:
        block = make_block(
            self.document,
            "figure-1",
            representations=(
                representation(RepresentationKind.CAPTION, "Figure 1: overview."),
            ),
        )
        enriched, stats = self.enrich((block,))

        self.assertEqual(enriched[0], block)
        self.assertEqual(stats.skipped_blocks, 1)
        self.assertEqual(stats.ocr_blocks, 0)

    def test_enrichment_is_idempotent(self) -> None:
        block = make_block(self.document, "figure-1", assets=(self.asset,))
        once, _ = self.enrich((block,))
        twice, stats = self.enrich(once)

        self.assertEqual(once, twice)
        self.assertEqual(stats.skipped_blocks, 1)
        self.assertEqual(stats.ocr_blocks, 0)

    def test_ocr_failure_leaves_the_block_intact(self) -> None:
        block = make_block(self.document, "figure-1", assets=(self.asset,))

        def broken(_data: bytes) -> FigureOCRResult:
            raise RuntimeError("model unavailable")

        with self.assertLogs("docarag.figure_enrichment", level="WARNING"):
            enriched, stats = self.enrich((block,), ocr_engine=broken)

        self.assertEqual(enriched[0], block)
        self.assertEqual(stats.failed_blocks, 1)

    def test_describer_failure_keeps_recovered_ocr_text(self) -> None:
        block = make_block(self.document, "figure-1", assets=(self.asset,))

        def broken(_data: bytes, _text: str) -> str:
            raise RuntimeError("vision model unavailable")

        with self.assertLogs("docarag.figure_enrichment", level="WARNING"):
            enriched, stats = self.enrich((block,), describer=broken)

        kinds = {item.kind for item in enriched[0].representations}
        self.assertIn(RepresentationKind.OCR_TEXT, kinds)
        self.assertEqual(stats.described_blocks, 0)


class EnrichedFigureProjectionTest(unittest.TestCase):
    """The recovered text must reach the BookRAG rows the tree is built from."""

    def setUp(self) -> None:
        self.document = make_document()
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.store = LocalContentAddressedArtifactStore(
            Path(self._tmp.name) / "asset_cas", create=True
        )
        stored = self.store.publish(CROP_BYTES, media_type="image/png")
        self.asset = stored.to_asset_ref(kind=AssetKind.REGION_CROP)

    def test_recovered_text_and_description_reach_the_projected_row(self) -> None:
        block = make_block(self.document, "figure-1", assets=(self.asset,))
        enriched, _ = enrich_visual_blocks(
            (block,),
            resolver=self.store.make_asset_resolver(),
            ocr_engine=lambda _data: table_spans(),
            describer=RecordingDescriber("A table of purchase orders."),
        )

        lease = BookRAGAssetMaterializationLease(
            self.store.make_asset_resolver(), (self.asset,)
        )
        with lease:
            row = project_canonical_blocks(
                self.document, enriched, asset_materialization_lease=lease
            )[0]

        self.assertEqual("image", row["type"])
        caption = " ".join(row["image_caption"])
        self.assertIn("Purchase Orders 126", caption)
        self.assertIn("A table of purchase orders.", caption)

    def test_a_figure_with_no_recovered_text_still_projects(self) -> None:
        # Before enrichment a caption-less figure could not be projected at all:
        # with no text and no leased crop the projector refused the block.
        block = make_block(self.document, "photo-1", assets=(self.asset,))
        enriched, _ = enrich_visual_blocks(
            (block,),
            resolver=self.store.make_asset_resolver(),
            ocr_engine=lambda _data: FigureOCRResult(spans=(), width=64, height=64),
            describer=RecordingDescriber("A photograph of a factory floor."),
        )

        row = project_canonical_blocks(self.document, enriched)[0]

        self.assertEqual(
            ["A photograph of a factory floor."], row["image_caption"]
        )


class GroundedFigureDescriberTest(unittest.TestCase):
    class FakeVLM:
        def __init__(self, reply: str = "A bar chart.") -> None:
            self.reply = reply
            self.prompts: list[str] = []
            self.images: list[list[str]] = []

        def generate(self, prompt_or_memory: str, images=None) -> str:
            self.prompts.append(prompt_or_memory)
            self.images.append(list(images or ()))
            return self.reply

    def test_recovered_text_is_supplied_as_authoritative(self) -> None:
        vlm = self.FakeVLM()
        describer = GroundedFigureDescriber(vlm)

        result = describer(CROP_BYTES, "Purchase Orders 126")

        self.assertEqual(result, "A bar chart.")
        self.assertIn("authoritative", vlm.prompts[0])
        self.assertIn("Purchase Orders 126", vlm.prompts[0])
        self.assertEqual(len(vlm.images[0]), 1)

    def test_missing_text_switches_to_a_visual_only_prompt(self) -> None:
        vlm = self.FakeVLM(reply="A photograph.")
        describer = GroundedFigureDescriber(vlm)

        result = describer(CROP_BYTES, "")

        self.assertEqual(result, "A photograph.")
        self.assertIn("OCR recovered no text", vlm.prompts[0])

    def test_blank_reply_is_not_evidence(self) -> None:
        describer = GroundedFigureDescriber(self.FakeVLM(reply="   "))

        self.assertIsNone(describer(CROP_BYTES, "Purchase Orders 126"))


if __name__ == "__main__":
    unittest.main()
