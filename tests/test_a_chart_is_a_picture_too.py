"""A chart is visual evidence because it is a chart, not because of its subtype.

``_visual_evidence_node`` decides whether an answer leases the crop behind a
node. Only image, table and equation qualified by *type*; chart and diagram
were left to the subtype fallback, which recognises the generic words
("chart", "diagram", "figure", "picture"...).

MinerU does not name a chart's subtype generically. It names the kind of chart
-- the Celestron EdgeHD paper's spot-diagram comparison on page 6 is
``semantic_subtype="bubble"`` -- so the fallback did not match, no asset ref
was selected, no crop was leased, ``img_path`` stayed unset, and the citation
reached the browser with ``hasImage=false``. The answer cited the figure twice
and never showed it, while ``/api/tree-image`` served that same crop happily
at 67 KB of JPEG.

Type is the authoritative signal here: the projection already separates chart
and diagram from image precisely because they are read differently, and every
other set in the query path -- ``_SECTIONED_TYPE_VALUES``,
``_SECTION_RERANK_NODE_TYPES``, hydration's ``_VISUAL_NODE_TYPES`` -- lists all
three. This one was the outlier.
"""

from __future__ import annotations

import unittest

from retrieval_engine.indexes.tree import NodeType
from retrieval_engine.rag.gbc.agent import GBCRAG

visual = GBCRAG._visual_evidence_node


class AChartIsVisualByType(unittest.TestCase):
    def test_a_chart_named_by_its_chart_kind(self):
        # The case from the paper: type says chart, subtype says "bubble".
        self.assertTrue(visual({"type": "chart", "semantic_subtype": "bubble"}))

    def test_a_diagram_named_by_its_diagram_kind(self):
        self.assertTrue(visual({"type": "diagram", "semantic_subtype": "sankey"}))

    def test_a_chart_with_no_subtype_at_all(self):
        self.assertTrue(visual({"type": "chart"}))

    def test_a_chart_whose_subtype_is_not_a_string(self):
        self.assertTrue(visual({"type": "chart", "semantic_subtype": None}))

    def test_the_node_type_enum_is_accepted_as_well_as_its_value(self):
        self.assertTrue(visual({"type": NodeType.CHART}))
        self.assertTrue(visual({"type": NodeType.DIAGRAM}))

    def test_every_kind_that_carries_a_crop_qualifies(self):
        for node_type in (
            NodeType.IMAGE,
            NodeType.CHART,
            NodeType.DIAGRAM,
            NodeType.TABLE,
            NodeType.EQUATION,
        ):
            with self.subTest(node_type=node_type):
                self.assertTrue(visual({"type": node_type.value}))


class TheSubtypeFallbackStillWorks(unittest.TestCase):
    """It exists for blocks typed generically, and still catches those."""

    def test_a_text_node_the_parser_called_a_figure(self):
        self.assertTrue(visual({"type": "text", "semantic_subtype": "figure"}))

    def test_a_subtype_written_with_a_space_or_a_dash(self):
        self.assertTrue(visual({"type": "text", "semantic_subtype": "flow chart"}))
        self.assertTrue(visual({"type": "text", "semantic_subtype": "Flow-Chart"}))


class ProseIsNotAPicture(unittest.TestCase):
    def test_a_paragraph_is_not_visual_evidence(self):
        self.assertFalse(visual({"type": "text", "semantic_subtype": "paragraph"}))

    def test_a_title_is_not_visual_evidence(self):
        self.assertFalse(visual({"type": "title"}))

    def test_a_paragraph_with_no_subtype(self):
        self.assertFalse(visual({"type": "text"}))


if __name__ == "__main__":
    unittest.main()
