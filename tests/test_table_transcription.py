"""Re-reading a table crop, and refusing to when the reading disagrees.

The recognizer returned ``1|2`` for a half and ``9|~ -9`` for root three over
two on a trigonometric ratios table -- a grid whose every interesting value is
a fraction or a radical, with none of them readable. The crop is re-read, but a
correction here has to claim the html primary slot to reach retrieval, so it is
admitted only when its shape matches: an invented table replacing a damaged one
is worse than the damage.
"""

from __future__ import annotations

import unittest

from document_services.domain.evidence import (
    AssetKind,
    AssetRef,
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
from document_services.enrichment.table_transcription import (
    TableTranscriptionStats,
    clean_table_html,
    table_shape,
    transcribe_table_blocks,
)

MANGLED = (
    "<table><tr><td>A</td><td>0</td><td>30</td></tr>"
    "<tr><td>sin A</td><td>0</td><td>1|2</td></tr></table>"
)
CORRECTED = (
    "<table><tr><td>A</td><td>0</td><td>30</td></tr>"
    "<tr><td>sin A</td><td>0</td><td>$\\frac{1}{2}$</td></tr></table>"
)


class _Material:
    def __init__(self, data: bytes) -> None:
        self.data = data


class _Resolver:
    def __init__(self, data: bytes = b"crop") -> None:
        self.data = data
        self.calls = 0

    def resolve(self, asset, *, include_bytes=False):
        self.calls += 1
        return _Material(self.data)


CROP_SHA = "a" * 64

DOCUMENT = DocumentRef.identified(
    content_sha256="b" * 64,
    mime_type="application/pdf",
    source_uri="file:///trigonometry.pdf",
    display_name="trigonometry.pdf",
    page_count=1,
)

_COUNTER = iter(range(1, 1000))


def make_block(*, block_type=BlockType.TABLE, representations=(), assets=(), metadata=None):
    key = f"block-{next(_COUNTER)}"
    return CanonicalBlock(
        canonical_key=key,
        revision=1,
        document_version_id=DOCUMENT.document_version_id,
        page_id=PageRef(
            document_version_id=DOCUMENT.document_version_id, page_index=0
        ).page_id,
        page_index=0,
        block_type=block_type,
        geometry=SourceGeometry(
            coordinate_space=CoordinateSpace.PAGE_POINTS,
            bbox=BoundingBox(10.0, 20.0, 400.0, 300.0),
            page_width=612.0,
            page_height=792.0,
        ),
        reading_order=1.0,
        representations=representations,
        assets=assets,
        source_observation_ids=(f"observation-{key}",),
        reconciliation_method="fixture",
        quality_status=QualityStatus.ACCEPTED,
        metadata=metadata or {},
    )


def table_block(html: str = MANGLED, *, with_crop: bool = True) -> CanonicalBlock:
    representation = Representation(
        kind=RepresentationKind.HTML,
        content=html,
        media_type="text/html",
        confidence=0.9,
    )
    assets = ()
    if with_crop:
        assets = (
            AssetRef(
                kind=AssetKind.REGION_CROP,
                uri=f"cas://sha256/{CROP_SHA}",
                media_type="image/jpeg",
                sha256=CROP_SHA,
            ),
        )
    return make_block(
        representations=(representation,),
        assets=assets,
        metadata={
            "primary_representations": {
                RepresentationKind.HTML.value: representation.representation_id
            }
        },
    )


class ShapeTests(unittest.TestCase):
    def test_shape_is_cells_per_row(self):
        self.assertEqual((3, 3), table_shape(MANGLED))

    def test_markup_that_is_not_a_table_has_no_shape(self):
        for text in ("", "I cannot read this image.", "<p>hello</p>", "<table></table>"):
            with self.subTest(text=text):
                self.assertIsNone(table_shape(text))

    def test_header_cells_count_as_cells(self):
        self.assertEqual((2,), table_shape("<table><tr><th>a</th><th>b</th></tr></table>"))


class CleanTests(unittest.TestCase):
    def test_a_fenced_reply_is_unwrapped(self):
        self.assertEqual(MANGLED, clean_table_html(f"```html\n{MANGLED}\n```"))

    def test_commentary_around_the_table_is_discarded(self):
        self.assertEqual(
            MANGLED, clean_table_html(f"Here is the table:\n{MANGLED}\nHope that helps.")
        )

    def test_a_refusal_yields_nothing(self):
        self.assertIsNone(clean_table_html("I'm sorry, I cannot read this image."))

    def test_an_enormous_reply_is_refused(self):
        huge = "<table>" + "<tr><td>x</td></tr>" * 5000 + "</table>"
        self.assertIsNone(clean_table_html(huge))


class TranscriptionTests(unittest.TestCase):
    def test_a_matching_reading_becomes_the_primary_html(self):
        block = table_block()
        resolver = _Resolver()

        out, stats = transcribe_table_blocks(
            [block], resolver=resolver, transcriber=lambda data: CORRECTED
        )

        self.assertEqual(1, stats.transcribed_blocks)
        corrected = out[0]
        primary_id = corrected.metadata["primary_representations"][
            RepresentationKind.HTML.value
        ]
        selected = next(
            item
            for item in corrected.representations
            if item.representation_id == primary_id
        )
        self.assertEqual(CORRECTED, selected.content)
        self.assertEqual("vlm.table_transcription", selected.metadata["producer"])

    def test_the_recognizers_own_reading_is_kept(self):
        block = table_block()

        out, _stats = transcribe_table_blocks(
            [block], resolver=_Resolver(), transcriber=lambda data: CORRECTED
        )

        self.assertIn(MANGLED, [item.content for item in out[0].representations])

    def test_a_differently_shaped_reading_is_refused(self):
        # A model that returns another shape read another table, or invented
        # one. Cell contents may be wrong; the grid is what the recognizer
        # gets right.
        block = table_block()
        wrong = "<table><tr><td>a</td><td>b</td></tr></table>"

        out, stats = transcribe_table_blocks(
            [block], resolver=_Resolver(), transcriber=lambda data: wrong
        )

        self.assertEqual(1, stats.shape_mismatch_blocks)
        self.assertEqual(0, stats.transcribed_blocks)
        self.assertEqual(block.metadata, out[0].metadata)
        self.assertEqual(1, len(out[0].representations))

    def test_an_unusable_reply_leaves_the_block_alone(self):
        block = table_block()

        out, stats = transcribe_table_blocks(
            [block], resolver=_Resolver(), transcriber=lambda data: None
        )

        self.assertEqual(1, stats.skipped_blocks)
        self.assertEqual(block, out[0])

    def test_an_identical_reading_is_not_added_twice(self):
        block = table_block()

        out, stats = transcribe_table_blocks(
            [block], resolver=_Resolver(), transcriber=lambda data: MANGLED
        )

        self.assertEqual(1, stats.skipped_blocks)
        self.assertEqual(1, len(out[0].representations))

    def test_a_table_without_a_crop_is_not_re_read(self):
        block = table_block(with_crop=False)
        resolver = _Resolver()

        out, stats = transcribe_table_blocks(
            [block], resolver=resolver, transcriber=lambda data: CORRECTED
        )

        self.assertEqual(0, resolver.calls)
        self.assertEqual(1, stats.skipped_blocks)
        self.assertEqual(block, out[0])

    def test_a_second_pass_does_not_re_transcribe(self):
        block = table_block()
        once, _ = transcribe_table_blocks(
            [block], resolver=_Resolver(), transcriber=lambda data: CORRECTED
        )

        twice, stats = transcribe_table_blocks(
            once, resolver=_Resolver(), transcriber=lambda data: CORRECTED
        )

        self.assertEqual(1, stats.skipped_blocks)
        self.assertEqual(len(once[0].representations), len(twice[0].representations))

    def test_a_resolver_failure_is_counted_and_survivable(self):
        class _Broken:
            def resolve(self, asset, *, include_bytes=False):
                raise OSError("crop is gone")

        block = table_block()

        out, stats = transcribe_table_blocks(
            [block], resolver=_Broken(), transcriber=lambda data: CORRECTED
        )

        self.assertEqual(1, stats.failed_blocks)
        self.assertEqual(block, out[0])

    def test_non_table_blocks_pass_through_untouched(self):
        block = table_block()
        other = make_block(
            block_type=BlockType.PARAGRAPH,
            representations=(
                Representation(
                    kind=RepresentationKind.TEXT,
                    content="words",
                    media_type="text/plain",
                ),
            ),
        )

        out, stats = transcribe_table_blocks(
            [other, block], resolver=_Resolver(), transcriber=lambda data: CORRECTED
        )

        self.assertEqual(other, out[0])
        self.assertEqual(1, stats.table_blocks)

    def test_no_tables_means_no_work(self):
        out, stats = transcribe_table_blocks(
            [], resolver=_Resolver(), transcriber=lambda data: CORRECTED
        )

        self.assertEqual((), out)
        self.assertEqual(TableTranscriptionStats(), stats)


if __name__ == "__main__":
    unittest.main()
