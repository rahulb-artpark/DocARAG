import unittest

from docarag.domain.evidence import (
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
from docarag.formula_transcription import (
    VLMFormulaTranscriber,
    _find_unambiguous_equation_span,
    clean_latex,
    has_inline_formula_candidates,
    transcribe_equation_blocks,
    transcribe_inline_equations,
)

CROP_SHA = "a" * 64

DOCUMENT = DocumentRef.identified(
    content_sha256="b" * 64,
    mime_type="application/pdf",
    source_uri="file:///physics.pdf",
    display_name="physics.pdf",
    page_count=1,
)

_COUNTER = iter(range(1, 1000))


def make_block(*, block_type=BlockType.EQUATION, representations=(), assets=()):
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
            bbox=BoundingBox(10.0, 20.0, 200.0, 60.0),
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


def equation_block(*, with_crop=True, representations=()):
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
        representations=(
            Representation(
                kind=RepresentationKind.LATEX,
                content="$$\nx = 1\n$$",
                media_type="application/x-latex",
            ),
            *representations,
        ),
        assets=assets,
    )


def inline_block(
    *, text, block_type=BlockType.PARAGRAPH, with_crop=True, confidence=0.9325
):
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
        block_type=block_type,
        representations=(
            Representation(
                kind=RepresentationKind.TEXT,
                content=text,
                media_type="text/plain",
                confidence=confidence,
            ),
        ),
        assets=assets,
    )


class _Resolver:
    def __init__(self, data=b"jpeg-bytes"):
        self.data = data
        self.resolved = []

    def resolve(self, asset, include_bytes=False):
        self.resolved.append(asset.sha256)

        class Material:
            data = self.data

        return Material()


class CleanLatexTests(unittest.TestCase):
    def test_fences_and_dollars_are_stripped(self):
        self.assertEqual(
            "v_2 = -\\frac{m}{M} v_1",
            clean_latex("```latex\n$$v_2 = -\\frac{m}{M} v_1$$\n```"),
        )

    def test_prose_without_math_is_rejected(self):
        self.assertIsNone(clean_latex("The image shows an equation."))

    def test_prose_words_inside_real_latex_survive(self):
        kept = clean_latex("E = mc^2 \\quad \\text{the mass–energy relation}")
        self.assertIn("E = mc^2", kept)

    def test_empty_and_oversized_are_rejected(self):
        self.assertIsNone(clean_latex("   "))
        self.assertIsNone(clean_latex("x" * 5000))


class TranscriberTests(unittest.TestCase):
    def test_reply_is_cleaned(self):
        class VLM:
            def generate(self, prompt, images=None):
                assert images and len(images) == 1
                return "$$ a = b $$"

        self.assertEqual("a = b", VLMFormulaTranscriber(VLM())(b"img"))

    def test_non_string_reply_is_dropped(self):
        class VLM:
            def generate(self, prompt, images=None):
                return None

        self.assertIsNone(VLMFormulaTranscriber(VLM())(b"img"))


class TranscribeBlocksTests(unittest.TestCase):
    def transcriber(self, latex="v_2 = -\\frac{m}{M} v_1"):
        class VLM:
            def generate(self, prompt, images=None):
                return latex

        return VLMFormulaTranscriber(VLM())

    def test_equation_gains_a_vlm_latex_channel(self):
        blocks, stats = transcribe_equation_blocks(
            [equation_block()],
            resolver=_Resolver(),
            transcriber=self.transcriber(),
        )

        kinds = [item.kind for item in blocks[0].representations]
        self.assertIn(RepresentationKind.VLM_LATEX, kinds)
        added = blocks[0].representations[-1]
        self.assertEqual("$$\nv_2 = -\\frac{m}{M} v_1\n$$", added.content)
        self.assertEqual("vlm.formula_transcription", added.metadata["producer"])
        self.assertEqual(CROP_SHA, added.metadata["asset_sha256"])
        self.assertEqual(1, stats.transcribed_blocks)

    def test_recognizer_output_is_kept_beside_the_transcription(self):
        blocks, _ = transcribe_equation_blocks(
            [equation_block()],
            resolver=_Resolver(),
            transcriber=self.transcriber(),
        )

        kinds = [item.kind for item in blocks[0].representations]
        self.assertIn(RepresentationKind.LATEX, kinds)

    def test_equation_without_crop_is_skipped(self):
        blocks, stats = transcribe_equation_blocks(
            [equation_block(with_crop=False)],
            resolver=_Resolver(),
            transcriber=self.transcriber(),
        )

        self.assertEqual(1, stats.skipped_blocks)
        self.assertEqual(0, stats.transcribed_blocks)

    def test_already_transcribed_equation_is_not_reread(self):
        existing = Representation(
            kind=RepresentationKind.VLM_LATEX,
            content="$$\nx = 1\n$$",
            media_type="application/x-latex",
        )
        resolver = _Resolver()

        _, stats = transcribe_equation_blocks(
            [equation_block(representations=(existing,))],
            resolver=resolver,
            transcriber=self.transcriber(),
        )

        self.assertEqual([], resolver.resolved)
        self.assertEqual(1, stats.skipped_blocks)

    def test_unusable_reply_leaves_the_block_unchanged(self):
        block = equation_block()

        blocks, stats = transcribe_equation_blocks(
            [block],
            resolver=_Resolver(),
            transcriber=self.transcriber(latex="The image shows an equation."),
        )

        self.assertEqual(block.representations, blocks[0].representations)
        self.assertEqual(1, stats.skipped_blocks)

    def test_failure_on_one_block_does_not_stop_the_rest(self):
        class Boom:
            def resolve(self, asset, include_bytes=False):
                raise OSError("no cas")

        blocks, stats = transcribe_equation_blocks(
            [equation_block(), equation_block()],
            resolver=Boom(),
            transcriber=self.transcriber(),
        )

        self.assertEqual(2, len(blocks))
        self.assertEqual(2, stats.failed_blocks)

    def test_non_equation_blocks_pass_through(self):
        block = equation_block()
        other = make_block(
            block_type=BlockType.PARAGRAPH,
            representations=(
                Representation(
                    kind=RepresentationKind.TEXT,
                    content="prose",
                    media_type="text/plain",
                ),
            ),
        )

        blocks, stats = transcribe_equation_blocks(
            [other, block],
            resolver=_Resolver(),
            transcriber=self.transcriber(),
        )

        self.assertIs(other, blocks[0])
        self.assertEqual(1, stats.equation_blocks)


class FindUnambiguousEquationSpanTests(unittest.TestCase):
    def test_no_dollar_spans_is_none(self):
        self.assertIsNone(_find_unambiguous_equation_span("plain text, no math"))

    def test_a_single_span_is_used_even_without_an_operator(self):
        match = _find_unambiguous_equation_span(r"see $\alpha$ above")
        self.assertEqual(r"\alpha", match.group(1))

    def test_two_spans_one_with_an_operator_picks_that_one(self):
        # The real reported case: a garbled equation next to an unrelated
        # units expression that also happens to be wrapped in "$...$".
        text = r"p = mv, in units of $\mathrm{kg}\,\mathrm{m}\,\mathrm{s}^{-1}$, namely $\rho = \pi \nu$."
        match = _find_unambiguous_equation_span(text)
        self.assertEqual(r"\rho = \pi \nu", match.group(1))

    def test_two_operator_bearing_spans_is_ambiguous(self):
        text = r"$a = b$ and also $c = d$"
        self.assertIsNone(_find_unambiguous_equation_span(text))

    def test_two_spans_neither_with_an_operator_is_ambiguous(self):
        text = r"$\alpha$ and $\beta$"
        self.assertIsNone(_find_unambiguous_equation_span(text))


class HasInlineFormulaCandidatesTests(unittest.TestCase):
    def test_true_for_an_eligible_block_containing_a_dollar_span(self):
        self.assertTrue(
            has_inline_formula_candidates([inline_block(text="p = $mv$")])
        )

    def test_false_with_no_dollar_sign_anywhere(self):
        self.assertFalse(
            has_inline_formula_candidates([inline_block(text="no math here")])
        )

    def test_false_for_an_equation_typed_block(self):
        # transcribe_equation_blocks already owns EQUATION blocks whole.
        block = inline_block(text="$p = mv$", block_type=BlockType.EQUATION)
        self.assertFalse(has_inline_formula_candidates([block]))


class TranscribeInlineEquationsTests(unittest.TestCase):
    def transcriber(self, latex="p = mv"):
        class VLM:
            def generate(self, prompt, images=None):
                return latex

        return VLMFormulaTranscriber(VLM())

    def test_the_real_reported_case_is_corrected_and_units_span_is_untouched(self):
        text = (
            "The product of mass m of a body and its velocity v is called "
            r"its linear momentum p. Mathematically $\rho = \pi \nu$ where "
            r"units are $\mathrm{kg}\,\mathrm{m}\,\mathrm{s}^{-1}$."
        )
        block = inline_block(text=text)

        blocks, stats = transcribe_inline_equations(
            [block], resolver=_Resolver(), transcriber=self.transcriber()
        )

        expected = text.replace(r"$\rho = \pi \nu$", "$p = mv$")
        corrected = blocks[0].representations[-1]
        self.assertEqual(expected, corrected.content)
        self.assertIn(r"$\mathrm{kg}\,\mathrm{m}\,\mathrm{s}^{-1}$", corrected.content)
        self.assertEqual(1, stats.corrected_blocks)
        self.assertEqual(0, stats.skipped_blocks)
        self.assertEqual(0, stats.failed_blocks)

    def test_corrected_representation_added_beside_the_original_with_higher_confidence(self):
        block = inline_block(text="p = $mv$", confidence=0.9325)

        blocks, _ = transcribe_inline_equations(
            [block], resolver=_Resolver(), transcriber=self.transcriber()
        )

        original, corrected = blocks[0].representations
        self.assertEqual("p = $mv$", original.content)
        self.assertEqual(RepresentationKind.TEXT, corrected.kind)
        self.assertGreater(corrected.confidence, original.confidence)
        self.assertEqual(
            "vlm.inline_formula_correction", corrected.metadata["producer"]
        )
        self.assertEqual(CROP_SHA, corrected.metadata["asset_sha256"])

    def test_equation_typed_blocks_are_left_to_transcribe_equation_blocks(self):
        block = inline_block(text="$p = mv$", block_type=BlockType.EQUATION)

        blocks, stats = transcribe_inline_equations(
            [block], resolver=_Resolver(), transcriber=self.transcriber()
        )

        self.assertEqual(block.representations, blocks[0].representations)
        self.assertEqual(0, stats.candidate_blocks)

    def test_no_crop_is_never_counted_as_a_candidate(self):
        block = inline_block(text="p = $mv$", with_crop=False)

        blocks, stats = transcribe_inline_equations(
            [block], resolver=_Resolver(), transcriber=self.transcriber()
        )

        self.assertEqual(block.representations, blocks[0].representations)
        self.assertEqual(0, stats.candidate_blocks)

    def test_ambiguous_multi_operator_spans_are_skipped_not_guessed(self):
        block = inline_block(text=r"$a = b$ and also $c = d$")

        blocks, stats = transcribe_inline_equations(
            [block], resolver=_Resolver(), transcriber=self.transcriber()
        )

        self.assertEqual(block.representations, blocks[0].representations)
        self.assertEqual(0, stats.candidate_blocks)

    def test_a_vlm_reply_matching_the_original_span_is_skipped_not_corrected(self):
        block = inline_block(text="p = $mv$")

        blocks, stats = transcribe_inline_equations(
            [block],
            resolver=_Resolver(),
            transcriber=self.transcriber(latex="mv"),
        )

        self.assertEqual(block.representations, blocks[0].representations)
        self.assertEqual(1, stats.candidate_blocks)
        self.assertEqual(1, stats.skipped_blocks)
        self.assertEqual(0, stats.corrected_blocks)

    def test_crop_resolution_failure_is_recorded_and_block_left_unchanged(self):
        class Boom:
            def resolve(self, asset, include_bytes=False):
                raise OSError("no cas")

        block = inline_block(text="p = $mv$")

        blocks, stats = transcribe_inline_equations(
            [block], resolver=Boom(), transcriber=self.transcriber()
        )

        self.assertEqual(block.representations, blocks[0].representations)
        self.assertEqual(1, stats.failed_blocks)

    def test_non_eligible_block_types_pass_through_untouched(self):
        # e.g. a table cell using "$" for currency, not LaTeX.
        block = inline_block(
            text="cost is $5", block_type=BlockType.TABLE
        )

        blocks, stats = transcribe_inline_equations(
            [block], resolver=_Resolver(), transcriber=self.transcriber()
        )

        self.assertEqual(block.representations, blocks[0].representations)
        self.assertEqual(0, stats.candidate_blocks)


if __name__ == "__main__":
    unittest.main()
