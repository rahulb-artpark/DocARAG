"""The outline pass is a second opinion that can both reject and nominate.

Fixtures are the real readings of 10P_CH11: the parser labelled "↓ Correction"
and "atmospheric refraction." headings and the outline pass marked both invalid,
yet both headed sections -- the second pulling page 3's content into a page 2
section. The parser missed "Hypermetropia or Far-Sightedness", which is its own
block and reads like a heading, so its content sat under Myopia instead.
"""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

from Core.Index.Tree import DocumentTree, NodeType
from Core.pipelines.doc_tree_builder import construct_tree_index
from Core.pipelines.outline_extractor import (
    extract_pdf_outline_in_chunks,
    heading_shaped,
)


def _block(pdf_id, text, *, block_type="text", level=None, page=0):
    content = {
        "pdf_id": pdf_id,
        "type": block_type,
        "text": text,
        "page_idx": page,
        "canonical_block_id": f"block-{pdf_id}",
        "canonical_revision_id": f"revision-{pdf_id}",
        "source_sha256": "a" * 64,
        "geometry": {"bbox": [0, pdf_id, 10, pdf_id + 1]},
        "middle_json": {"bbox": [0, pdf_id * 10, 400, pdf_id * 10 + 12]},
    }
    if level is not None:
        content["text_level"] = level
    return content


def _new_tree():
    return DocumentTree(
        meta_dict={"file_name": "source.pdf"},
        cfg=SimpleNamespace(save_path="/tmp"),
    )


def _node_by_pdf_id(tree, pdf_id):
    return next(n for n in tree.get_nodes() if n.meta_info.pdf_id == pdf_id)


def _descendants(node):
    total = 0
    for child in node.children:
        total += 1 + _descendants(child)
    return total


class HeadingShape(unittest.TestCase):
    def test_a_short_capitalised_fragment_is_offered(self):
        self.assertTrue(heading_shaped("Hypermetropia or Far-Sightedness"))
        self.assertTrue(heading_shaped("Some Natwral Phenomenon Based on Refraction"))

    def test_a_sentence_is_not_offered(self):
        self.assertFalse(
            heading_shaped("Power of accommodation of eye decreases.")
        )
        self.assertFalse(heading_shaped("For normal eye: Far point -Infinity."))

    def test_ocr_debris_is_not_offered(self):
        self.assertFalse(heading_shaped("www.notesd"))
        self.assertFalse(heading_shaped("owrof th Sun at Sunvise 4 Sust"))

    def test_a_paragraph_is_not_offered(self):
        self.assertFalse(
            heading_shaped(
                "They are larger in size and are much closer to the earth so "
                "the light they emit does not flicker the way a star's does"
            )
        )

    def test_a_numbered_heading_is_offered(self):
        self.assertTrue(heading_shaped("2.1 Methodology"))
        self.assertTrue(heading_shaped("Fig.3 Ray diagram"))

    def test_a_sentence_carrying_a_label_is_not_offered(self):
        # Offered by a looser filter, promoted by the model, and it headed a
        # section holding the two blocks after it.
        self.assertFalse(
            heading_shaped("Power of accomodation of eye decreases. ↓ Causes")
        )

    def test_a_label_pointing_at_its_content_is_not_offered(self):
        # Offered by a looser filter, promoted to a top-level section, and it
        # then owned Atmospheric Refraction, Tyndall Effect and the rest of the
        # chapter.
        self.assertFalse(heading_shaped("Conditions for its formation →"))
        self.assertFalse(heading_shaped("↓ Causes"))
        self.assertFalse(heading_shaped("Correction ->"))

    def test_non_text_is_not_offered(self):
        self.assertFalse(heading_shaped(None))
        self.assertFalse(heading_shaped("   "))


class _FakeLLM:
    """Answers with a fixed verdict per pdf_id, and records what it was asked."""

    def __init__(self, verdicts):
        self.verdicts = verdicts
        self.seen = []
        self.config = SimpleNamespace(max_tokens=8000)

    def get_json_completion(self, prompt, _schema):
        self.prompt = prompt
        entries = []
        for pdf_id, (level, parent) in self.verdicts.items():
            self.seen.append(pdf_id)
            entries.append({"pdf_id": pdf_id, "level": level, "parent_id": parent})
        return SimpleNamespace(model_dump=lambda: {"outline": entries})


class TheOutlinePassCarriesItsRejections(unittest.TestCase):
    def test_an_invalid_verdict_on_a_parser_heading_is_returned(self):
        pdf_list = [
            _block(1, "Human Eye and the Colourful World", level=1),
            _block(2, "Presbyopia", level=1),
            _block(3, "↓ Correction", level=6),
        ]
        llm = _FakeLLM({1: (0, 0), 2: (1, 1), 3: (-1, -1)})
        outline, rejected = extract_pdf_outline_in_chunks(pdf_list, llm)
        self.assertEqual(rejected, frozenset({3}))
        self.assertNotIn(3, [entry["pdf_id"] for entry in outline])

    def test_a_declined_promotion_is_not_a_rejection(self):
        # Body text the pass refuses to promote was never a heading, so it must
        # not be reported as one the tree has to stop trusting.
        pdf_list = [
            _block(1, "Human Eye and the Colourful World", level=1),
            _block(2, "Conditions for its formation"),
        ]
        llm = _FakeLLM({1: (0, 0), 2: (-1, -1)})
        _, rejected = extract_pdf_outline_in_chunks(pdf_list, llm)
        self.assertEqual(rejected, frozenset())

    def test_a_missed_heading_is_offered_and_promoted(self):
        pdf_list = [
            _block(1, "Human Eye and the Colourful World", level=1),
            _block(2, "Myopia or Near-Sightedness", level=2),
            _block(3, "Cannot see distant objects distinctly."),
            _block(4, "Hypermetropia or Far-Sightedness"),
        ]
        llm = _FakeLLM({1: (0, 0), 2: (1, 1), 4: (1, 1)})
        outline, rejected = extract_pdf_outline_in_chunks(pdf_list, llm)
        self.assertIn(4, llm.seen, "the parser's miss was never offered")
        self.assertNotIn(3, llm.seen, "a sentence must not be offered")
        self.assertIn(4, [entry["pdf_id"] for entry in outline])
        self.assertEqual(rejected, frozenset())

    def test_a_failed_chunk_changes_nothing(self):
        pdf_list = [
            _block(1, "Human Eye and the Colourful World", level=1),
            _block(2, "↓ Correction", level=6),
            _block(3, "Hypermetropia or Far-Sightedness"),
        ]

        class Broken(_FakeLLM):
            def get_json_completion(self, prompt, _schema):
                raise RuntimeError("no response")

        outline, rejected = extract_pdf_outline_in_chunks(pdf_list, Broken({}))
        ids = [entry["pdf_id"] for entry in outline]
        self.assertEqual(rejected, frozenset(), "silence is not a verdict")
        self.assertIn(2, ids, "the parser's heading must survive a failed chunk")
        self.assertNotIn(
            3, ids, "a promotion must not ride on a response that never arrived"
        )


class ARejectedHeadingDoesNotOpenASection(unittest.TestCase):
    """The real page-2/page-3 shape of 10P_CH11, reduced to its skeleton."""

    def _blocks(self):
        return [
            _block(1, "Human Eye and the Colourful World", level=1),
            _block(2, "Presbyopia", level=1, page=1),
            _block(3, "Power of accommodation of eye decreases.", page=1),
            _block(4, "atmospheric refraction.", level=6, page=2),
            _block(5, "Sunlight gets refracted through the layers.", page=2),
        ]

    def _outline(self):
        return [
            {"pdf_id": 1, "text_level": 0, "parent_id": 0},
            {"pdf_id": 2, "text_level": 1, "parent_id": 1},
        ]

    def test_without_the_rejection_a_later_page_joins_an_earlier_section(self):
        tree = construct_tree_index(
            tree_index=_new_tree(),
            pdf_list=self._blocks(),
            title_outline=self._outline(),
        )
        fragment = _node_by_pdf_id(tree, 4)
        self.assertEqual(fragment.type, NodeType.TITLE)
        self.assertEqual(_descendants(_node_by_pdf_id(tree, 2)), 3)

    def test_the_rejection_keeps_it_body_text_under_its_own_section(self):
        tree = construct_tree_index(
            tree_index=_new_tree(),
            pdf_list=self._blocks(),
            title_outline=self._outline(),
            rejected_title_ids=frozenset({4}),
        )
        fragment = _node_by_pdf_id(tree, 4)
        self.assertEqual(fragment.type, NodeType.TEXT)
        self.assertFalse(fragment.outline_node)
        self.assertEqual(
            fragment.parent.meta_info.pdf_id,
            2,
            "a demoted heading stays inside the section it was found in",
        )
        # Page 3's body no longer hangs off a heading that was never a heading;
        # it is a sibling of the page-2 prose, not its nephew.
        self.assertEqual(
            _node_by_pdf_id(tree, 5).parent.meta_info.pdf_id, 2
        )

    def test_every_block_is_still_attached_exactly_once(self):
        tree = construct_tree_index(
            tree_index=_new_tree(),
            pdf_list=self._blocks(),
            title_outline=self._outline(),
            rejected_title_ids=frozenset({4}),
        )
        attached = [c.meta_info.pdf_id for n in tree.nodes for c in n.children]
        self.assertEqual(sorted(attached), [1, 2, 3, 4, 5])

    def test_a_rejection_cannot_be_overruled_by_a_hierarchy_entry(self):
        tree = construct_tree_index(
            tree_index=_new_tree(),
            pdf_list=self._blocks(),
            title_outline=self._outline() + [
                {"pdf_id": 4, "text_level": 2, "parent_id": 2}
            ],
            rejected_title_ids=frozenset({4}),
        )
        self.assertEqual(_node_by_pdf_id(tree, 4).type, NodeType.TEXT)

    def test_an_unknown_rejected_id_is_harmless(self):
        tree = construct_tree_index(
            tree_index=_new_tree(),
            pdf_list=self._blocks(),
            title_outline=self._outline(),
            rejected_title_ids=frozenset({99}),
        )
        self.assertEqual(_node_by_pdf_id(tree, 4).type, NodeType.TITLE)


if __name__ == "__main__":
    unittest.main()
