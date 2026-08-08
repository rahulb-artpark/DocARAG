"""A decomposed question cites one table three times; it is still one table.

Asking "show me the image of Wolter Surface, its parameters and the equation
that defines it" decomposes into three sub-questions, and each gets its own
evidence scope. Node 8287 -- the parameter table on p577 -- therefore reaches
the answer as Q1-E5, Q2-E8 *and* Q3-E7, and node 8290 as Q1-E7, Q2-E10 and
Q3-E9. The inline-figure renderer skipped a citation whose *label* was already
drawn, so three labels for one node drew that node's crop three times.

The choice of which label anchors the figure matters as much as the count. The
lowest-numbered scope is not necessarily the one the prose names: in that
answer Q1-E5 never appears in the text and Q2-E8 does, so anchoring on the
first label parks the figure at the end of the answer instead of beside the
paragraph it illustrates.

``figuresToInline`` holds that decision and takes no DOM, so these run it for
real rather than asserting on the shape of the source.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
APP = (REPO / "Scripts" / "webui" / "app.js").read_text(encoding="utf-8")

#: The citations the Wolter answer actually produced, and the labels its prose
#: actually used.
WOLTER_CITATIONS = [
    {"label": "Q1-E1", "nodeId": 6742, "hasImage": False},
    {"label": "Q1-E3", "nodeId": 8285, "hasImage": True},
    {"label": "Q1-E5", "nodeId": 8287, "hasImage": True},
    {"label": "Q1-E7", "nodeId": 8290, "hasImage": True},
    {"label": "Q2-E8", "nodeId": 8287, "hasImage": True},
    {"label": "Q2-E10", "nodeId": 8290, "hasImage": True},
    {"label": "Q3-E7", "nodeId": 8287, "hasImage": True},
    {"label": "Q3-E9", "nodeId": 8290, "hasImage": True},
]
WOLTER_MENTIONED = ["Q1-E1", "Q1-E3", "Q1-E7", "Q2-E8", "Q2-E9", "Q3-E8"]


def _run(citations, mentioned):
    """Run the shipped figuresToInline over plain data."""

    script = """
const fs = require("fs");
const src = fs.readFileSync(process.argv[1], "utf8");
const match = src.match(/function figuresToInline[\\s\\S]*?\\n}\\n/);
if (!match) { console.error("figuresToInline not found"); process.exit(2); }
eval(match[0]);
const [citations, mentioned] = JSON.parse(process.argv[2]);
const seen = new Set(mentioned);
const chosen = figuresToInline(citations, (label) => seen.has(label));
process.stdout.write(JSON.stringify(chosen.map((c) => [c.label, c.nodeId])));
"""
    result = subprocess.run(
        [
            "node",
            "-e",
            script,
            str(REPO / "Scripts" / "webui" / "app.js"),
            json.dumps([citations, mentioned]),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        raise AssertionError(f"node failed: {result.stderr.strip()}")
    return [tuple(pair) for pair in json.loads(result.stdout)]


@unittest.skipUnless(shutil.which("node"), "node is required to run app.js")
class OneNodeIsOneFigure(unittest.TestCase):
    def test_the_table_cited_three_times_is_drawn_once(self):
        drawn = _run(WOLTER_CITATIONS, WOLTER_MENTIONED)

        self.assertEqual([8285, 8287, 8290], sorted(node for _, node in drawn))

    def test_seven_image_citations_become_three_figures(self):
        drawn = _run(WOLTER_CITATIONS, WOLTER_MENTIONED)

        self.assertEqual(3, len(drawn))

    def test_the_anchor_is_a_label_the_prose_uses(self):
        # Q1-E5 and Q3-E7 are the same table and neither is named; Q2-E8 is.
        drawn = dict((node, label) for label, node in _run(WOLTER_CITATIONS, WOLTER_MENTIONED))

        self.assertEqual("Q2-E8", drawn[8287])

    def test_an_unmentioned_node_still_gets_a_figure(self):
        # Cited as an image but never named: shown at the end rather than
        # dropped, so it must still be chosen here.
        drawn = _run(
            [{"label": "Q1-E5", "nodeId": 8287, "hasImage": True}], []
        )

        self.assertEqual([("Q1-E5", 8287)], drawn)

    def test_a_node_without_a_crop_is_never_drawn(self):
        drawn = _run([{"label": "Q1-E1", "nodeId": 6742, "hasImage": False}], ["Q1-E1"])

        self.assertEqual([], drawn)

    def test_a_citation_with_no_node_is_never_drawn(self):
        drawn = _run([{"label": "E1", "nodeId": None, "hasImage": True}], ["E1"])

        self.assertEqual([], drawn)

    def test_an_undecomposed_answer_is_unaffected(self):
        # The single-scope case has one label per node and must behave exactly
        # as it did before.
        citations = [
            {"label": "E1", "nodeId": 8285, "hasImage": True},
            {"label": "E2", "nodeId": 8287, "hasImage": True},
        ]

        self.assertEqual([("E1", 8285), ("E2", 8287)], _run(citations, ["E1", "E2"]))

    def test_no_citations_draws_nothing(self):
        self.assertEqual([], _run([], []))


class TheDedupeKeyIsTheNode(unittest.TestCase):
    """Source-level guards, so the regression is caught even without node."""

    def test_the_figure_element_is_keyed_by_node_id(self):
        self.assertIn('data-figure="${escapeHtml(String(citation.nodeId))}"', APP)

    def test_the_already_drawn_check_uses_the_node_id(self):
        self.assertIn(
            'querySelector(`[data-figure="${CSS.escape(String(citation.nodeId))}"]`)', APP
        )

    def test_the_renderer_goes_through_the_decision_function(self):
        self.assertIn("figuresToInline(citations, (label) =>", APP)


if __name__ == "__main__":
    unittest.main()
