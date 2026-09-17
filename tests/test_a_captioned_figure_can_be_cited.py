"""What seals a node's evidence and what checks it must project it alike.

A tree node's identity is written by one component and verified by another.
The refiner seals a transformed node's ``derived:<digest>`` from the *block*
it transformed; the hydrator recomputes it from the *node* before an answer
may cite it. Nothing made the two agree, and they had drifted apart in three
places at once. Asking the Celestron EdgeHD paper "Compare Coma Free SCT and
Flat-Field EdgeHD" refused seven of its 225 nodes -- including the spot-diagram
figure that answers the question -- with "derived evidence digest differs from
its output or canonical lineage".

Each drift is a different pair of fields for the same node:

    caption + footnote   the tree writes one string with a space between the
                         two channels; the hydrator glued them with nothing,
                         so every figure and table carrying both was refused
    a merged figure      the refiner sealed ``block["text"]``, and a visual
                         block has no text channel -- so the seal covered the
                         empty string, protecting none of the figure's words
                         and being impossible to recompute from the node
    a promoted heading   the refiner sealed the block's type; the outline pass
                         afterwards promotes a section-opening text block to
                         "title", and the hydrator read the promoted type

``caption_footnote_text`` is now the single definition of the first, and the
hydrator takes the type from the same canonical projection the refiner was
handed for the third. The second was a real hole in the seal rather than a
disagreement, so the refiner was fixed to seal what the node carries.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from document_services.evidence.identity import (
    caption_footnote_text,
    channel_text,
    derived_evidence_id,
    derived_identity_payload,
    derived_output_identity,
)
from document_services.domain import (
    AssetKind,
    AssetRef,
    BlockType,
    Representation,
    RepresentationKind,
)
from document_services.evidence.hydration import EvidenceHydrationError

from retrieval_engine.parsing.pdf_refinement import _derived_output_content
from retrieval_engine.indexing.tree.nodes import create_node_by_type

from tests.test_evidence_hydration import (
    CANONICAL_CONTRACT_SCHEMA_VERSION,
    CANONICAL_EVIDENCE_SCHEMA_VERSION,
    CanonicalEvidenceHydrator,
    make_block,
    make_document,
    tree_record,
)

CAPTION = "Figure 7. Spot diagrams for the Coma-Free SCT and the EdgeHD."
FOOTNOTE = "Plotted at 0.0, 3.5, 7, 10.5 and 14 mm off-axis."


class HydrationFixture(unittest.TestCase):
    """One document, the blocks a test needs, and a hydrator over both."""

    blocks: tuple = ()

    def setUp(self) -> None:
        self.document = make_document()
        self.blocks = self.make_blocks()
        self.records = {
            index: tree_record(self.document, block, index)
            for index, block in enumerate(self.blocks, start=1)
        }
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.artifact = Path(self.temporary_directory.name) / "canonical_blocks.json"
        self.artifact.write_text(
            json.dumps(
                {
                    "artifact_schema_version": CANONICAL_EVIDENCE_SCHEMA_VERSION,
                    "contract_schema_version": CANONICAL_CONTRACT_SCHEMA_VERSION,
                    "document": self.document.to_dict(),
                    "parse_request_id": "request-fixture",
                    "block_count": len(self.blocks),
                    "canonical_blocks": [block.to_dict() for block in self.blocks],
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    def make_blocks(self) -> tuple:
        raise NotImplementedError

    def hydrator(self) -> CanonicalEvidenceHydrator:
        return CanonicalEvidenceHydrator.from_artifact(
            self.artifact,
            node_resolver=lambda ids: [deepcopy(self.records[item]) for item in ids],
            expected_document_version_id=self.document.document_version_id,
            expected_source_sha256=self.document.content_sha256,
            expected_source_uri=self.document.source_uri,
        )


class OneRuleForTwoChannels(unittest.TestCase):
    """Everything that joins a caption to a footnote joins it the same way."""

    def test_the_channels_are_separated(self):
        self.assertEqual(
            f"{CAPTION} {FOOTNOTE}",
            caption_footnote_text([CAPTION], [FOOTNOTE]),
        )

    def test_a_missing_channel_leaves_no_stray_separator(self):
        self.assertEqual(CAPTION, caption_footnote_text([CAPTION], []))
        self.assertEqual(FOOTNOTE, caption_footnote_text([], [FOOTNOTE]))
        self.assertEqual("", caption_footnote_text([], []))

    def test_entries_within_one_channel_are_joined_too(self):
        self.assertEqual("north south", channel_text(["north", "south"]))

    def test_a_channel_that_is_not_a_list_contributes_nothing(self):
        # The projection always emits lists; a string here would silently
        # become one character per entry.
        self.assertEqual("", channel_text(CAPTION))

    def test_the_tree_builder_writes_exactly_that(self):
        node = create_node_by_type(
            {
                "type": "image",
                "image_caption": [CAPTION],
                "image_footnote": [FOOTNOTE],
                "img_path": "/crop.png",
            },
            False,
        )

        self.assertEqual(f"{CAPTION} {FOOTNOTE}", node.meta_info.content)
        self.assertEqual(CAPTION, node.meta_info.caption)
        self.assertEqual(FOOTNOTE, node.meta_info.footnote)

    def test_a_table_node_is_written_the_same_way(self):
        node = create_node_by_type(
            {
                "type": "table",
                "table_caption": ["Table 2. Field curvature."],
                "table_footnote": ["Measured at 20 C."],
                "table_body": "<table></table>",
            },
            False,
        )

        self.assertEqual(
            "Table 2. Field curvature. Measured at 20 C.", node.meta_info.content
        )


class TheRefinerSealsWhatTheNodeCarries(unittest.TestCase):
    def test_a_figure_is_sealed_over_its_caption(self):
        output = _derived_output_content(
            {
                "type": "image",
                "image_caption": [CAPTION],
                "image_footnote": [FOOTNOTE],
            }
        )

        self.assertEqual(f"{CAPTION} {FOOTNOTE}", output["text"])

    def test_a_figure_is_no_longer_sealed_over_nothing(self):
        # ``text`` is the field a visual block does not have. Reading it sealed
        # every merged figure over "", so no change to the figure's own words
        # could ever break the digest.
        self.assertNotEqual(
            "",
            _derived_output_content(
                {"type": "image", "image_caption": [CAPTION]}
            )["text"],
        )

    def test_a_chart_and_a_diagram_are_visual_too(self):
        for content_type in ("chart", "diagram"):
            with self.subTest(content_type=content_type):
                output = _derived_output_content(
                    {"type": content_type, "image_caption": [CAPTION]}
                )
                self.assertEqual(CAPTION, output["text"])

    def test_prose_still_seals_its_own_text(self):
        output = _derived_output_content(
            {"type": "text", "text": "The corrector is a two-lens group."}
        )

        self.assertEqual("The corrector is a two-lens group.", output["text"])

    def test_a_table_still_seals_its_grid(self):
        output = _derived_output_content(
            {
                "type": "table",
                "table_body": "<table><tr><td>14</td></tr></table>",
                "table_caption": ["Table 2."],
                "table_footnote": ["Measured at 20 C."],
            }
        )

        self.assertEqual("<table><tr><td>14</td></tr></table>", output["table_body"])
        self.assertEqual("Table 2.", output["table_caption"])
        self.assertEqual("Measured at 20 C.", output["table_footnote"])


def spot_diagram_block(document, *, page_index: int = 2):
    """A chart with both a caption and a footnote, like the one in the paper."""

    return make_block(
        document,
        "spot-diagrams",
        page_index=page_index,
        block_type=BlockType.CHART,
        representations=(
            Representation(
                kind=RepresentationKind.CAPTION,
                content=CAPTION,
                media_type="text/plain",
            ),
            Representation(
                kind=RepresentationKind.FOOTNOTE,
                content=FOOTNOTE,
                media_type="text/plain",
            ),
        ),
        assets=(
            AssetRef(
                kind=AssetKind.REGION_CROP,
                uri="/evidence/spot-diagrams.png",
                media_type="image/png",
                sha256="d" * 64,
            ),
        ),
    )


def equation_block(document, *, page_index: int = 1):
    return make_block(
        document,
        "formula",
        page_index=page_index,
        block_type=BlockType.EQUATION,
        representations=(
            Representation(
                kind=RepresentationKind.LATEX,
                content=r"E = mc^2",
                media_type="application/x-latex",
            ),
        ),
    )


class ACaptionedFigureHydrates(HydrationFixture):
    """A figure carrying both channels reaches an answer."""

    def make_blocks(self) -> tuple:
        return (spot_diagram_block(self.document),)

    def test_a_figure_with_a_caption_and_a_footnote_is_accepted(self):
        item = self.hydrator().hydrate([deepcopy(self.records[1])])[0]

        self.assertEqual(f"{CAPTION} {FOOTNOTE}", item["content"])

    def test_both_channels_reach_the_answer(self):
        item = self.hydrator().hydrate([deepcopy(self.records[1])])[0]

        self.assertIn("Coma-Free SCT", item["content"])
        self.assertIn("off-axis", item["content"])

    def test_a_figure_whose_content_was_edited_is_still_refused(self):
        tampered = deepcopy(self.records[1])
        tampered["content"] = f"{CAPTION} Plotted on-axis only."
        self.records[1] = deepcopy(tampered)

        with self.assertRaisesRegex(EvidenceHydrationError, "content differs"):
            self.hydrator().hydrate([tampered])


class ATransformedNodeHydrates(HydrationFixture):
    """The two ways a derived node's identity used to become unreproducible."""

    def make_blocks(self) -> tuple:
        self.formula = equation_block(self.document)
        self.figure = spot_diagram_block(self.document)
        return (self.formula, self.figure)

    def _seal(self, node: dict, *, content_type: str) -> dict:
        refs = node["source_evidence_refs"]
        node["derived_evidence_id"] = derived_evidence_id(
            derived_identity_payload(
                output=derived_output_identity(
                    content_type=content_type, text=node["content"]
                ),
                source_revision_ids=node["source_canonical_revision_ids"],
                source_document_version_ids=[self.document.document_version_id],
                source_block_ids=node["source_canonical_block_ids"],
                source_observation_ids=node["source_observation_ids"],
                source_references=refs,
                transformations=node["transformations"],
            )
        )
        return node

    def _derived_from(self, block, *, node_type: str, content: str) -> dict:
        record = tree_record(self.document, block, 9)
        refs = deepcopy(record["source_evidence_refs"])
        for field in ("canonical_block_id", "canonical_revision_id", "observation_id"):
            record.pop(field, None)
        record.update(
            {
                "type": node_type,
                "content": content,
                "source_canonical_block_ids": [block.block_id],
                "source_canonical_revision_ids": [block.revision_id],
                "source_observation_ids": list(block.source_observation_ids),
                "source_evidence_refs": refs,
                "transformations": ["bookrag.text_merge.v1"],
            }
        )
        return record

    def test_a_heading_promoted_after_its_seal_is_accepted(self):
        # The refiner sealed "text"; the outline pass then decided this block
        # opens a section, and the node reached the hydrator as a title.
        node = self._derived_from(
            self.formula, node_type="title", content="Serosa -Connective tissues"
        )
        self.records[9] = self._seal(node, content_type="equation")

        item = self.hydrator().hydrate([deepcopy(self.records[9])])[0]

        self.assertEqual("title", item["type"])
        self.assertTrue(item["evidence_id"].startswith("derived:"))

    def test_a_promoted_heading_whose_text_was_edited_is_still_refused(self):
        # Accepting the promotion must not stop covering the output: the type
        # comes from the canonical block, the text still comes from the node.
        node = self._derived_from(
            self.formula, node_type="title", content="Serosa -Connective tissues"
        )
        self._seal(node, content_type="equation")
        node["content"] = "Serosa -Muscular tissues"
        self.records[9] = deepcopy(node)

        with self.assertRaisesRegex(EvidenceHydrationError, "derived evidence digest"):
            self.hydrator().hydrate([deepcopy(node)])

    def test_a_merged_figure_is_accepted(self):
        node = self._derived_from(
            self.figure, node_type="chart", content=f"{CAPTION} {FOOTNOTE}"
        )
        node["transformations"] = ["bookrag.figure_merge.v1"]
        self.records[9] = self._seal(node, content_type="chart")

        item = self.hydrator().hydrate([deepcopy(self.records[9])])[0]

        self.assertEqual(f"{CAPTION} {FOOTNOTE}", item["content"])

    def test_a_merged_figure_whose_caption_was_edited_is_refused(self):
        # This is what sealing "" gave away: a merged figure's words were
        # outside its own digest, so no edit to them could ever be detected.
        node = self._derived_from(
            self.figure, node_type="chart", content=f"{CAPTION} {FOOTNOTE}"
        )
        node["transformations"] = ["bookrag.figure_merge.v1"]
        self._seal(node, content_type="chart")
        node["content"] = "Figure 7. Spot diagrams for the EdgeHD alone."
        self.records[9] = deepcopy(node)

        with self.assertRaisesRegex(EvidenceHydrationError, "derived evidence digest"):
            self.hydrator().hydrate([deepcopy(node)])

    def test_a_seal_naming_another_block_type_is_refused(self):
        # The type is no longer whatever the node claims: a digest sealed over
        # a type the canonical block does not project to stays invalid.
        node = self._derived_from(
            self.formula, node_type="equation", content="E = mc^2 restated"
        )
        self._seal(node, content_type="text")
        self.records[9] = deepcopy(node)

        with self.assertRaisesRegex(EvidenceHydrationError, "derived evidence digest"):
            self.hydrator().hydrate([deepcopy(node)])


if __name__ == "__main__":
    unittest.main()
