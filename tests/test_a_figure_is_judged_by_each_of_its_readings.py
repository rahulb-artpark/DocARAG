"""A node's summary and its content are scored separately, best score wins.

A figure that shows something without naming it carries two unrelated texts:
the heading and summary saying what it is, and a description of what it looks
like. Joined into one reranker document the longer one decides the score.

Node 8297 of test.pdf is the case. Its summary names the subject in 320
characters; its vision-model content spends 610 arguing the opposite -- "a 3D
rendered icon or graphic, not a chart, table, or diagram ... typical of digital
icons or interface elements". Asked "show me the image of Zernike Surface and
the equation that defines it", the joined document scored 0.9844 and the figure
ranked **31st of 512** candidates for 14 slots, so the answer said the document
held no image of a Zernike surface. Its vectors had it at rank 1 on both typed
legs.

Scored as two documents it ranks **5th at 1.0000** on its naming view, and the
pixel description keeps a document of its own, so "what does that figure look
like" still reaches it. Measured: 666 documents for 512 candidates, because only
a node that both shows and names gets a second one.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from retrieval_engine.indexes.tree import NodeType
from retrieval_engine.rag.gbc.retrieval import Retriever

HEADING = "2.3.2.2.94. Zernike Surface"
SUMMARY = "A 3D rendering of the Zernike surface, whose polynomials describe aberrations."
CONTENT = (
    "This is a 3D rendered icon or graphic, not a chart, table, or diagram. "
    "A teardrop form in grey gradient, typical of digital interface elements."
)


def _node(node_type, *, summary=SUMMARY, content=CONTENT, heading=HEADING):
    parent = (
        SimpleNamespace(type=NodeType.TITLE, meta_info=SimpleNamespace(content=heading))
        if heading
        else None
    )
    return SimpleNamespace(
        type=node_type,
        summary=summary,
        parent=parent,
        index_id=8297,
        meta_info=SimpleNamespace(
            content=content, caption=None, footnote=None,
            table_body=None, semantic_substructures=None,
        ),
    )


class EachReadingIsItsOwnDocument(unittest.TestCase):
    def test_a_figure_yields_a_naming_view_and_a_body_view(self):
        views = dict(Retriever._node_rerank_views(_node(NodeType.IMAGE)))

        self.assertEqual({"named", "body"}, set(views))

    def test_the_naming_view_is_not_diluted_by_the_pixel_description(self):
        views = dict(Retriever._node_rerank_views(_node(NodeType.IMAGE)))

        self.assertIn(SUMMARY, views["named"])
        self.assertIn(HEADING, views["named"])
        self.assertNotIn("interface elements", views["named"])

    def test_the_body_view_keeps_the_description_and_its_heading(self):
        # A question about what the picture looks like has to still reach it,
        # and the heading is the only thing placing it in the document.
        views = dict(Retriever._node_rerank_views(_node(NodeType.IMAGE)))

        self.assertIn("interface elements", views["body"])
        self.assertIn(HEADING, views["body"])
        self.assertNotIn(SUMMARY, views["body"])

    def test_every_kind_that_shows_without_naming_splits(self):
        for node_type in (
            NodeType.IMAGE, NodeType.CHART, NodeType.DIAGRAM,
            NodeType.TABLE, NodeType.EQUATION, NodeType.CODE,
        ):
            with self.subTest(node_type=node_type):
                views = dict(Retriever._node_rerank_views(_node(node_type)))
                self.assertEqual({"named", "body"}, set(views))

    def test_a_paragraph_stays_one_document(self):
        # Prose is its own subject; there is nothing to separate, and a second
        # document would double the reranker's work for no signal.
        views = Retriever._node_rerank_views(_node(NodeType.TEXT))

        self.assertEqual(1, len(views))
        self.assertEqual("whole", views[0][0])

    def test_a_figure_with_no_summary_is_still_judged_by_its_heading(self):
        # A bare heading is a strong document on its own: measured on the Wolter
        # figure, heading alone scored 0.9961 where its pixel description scored
        # 0.0000. So an unsummarised figure still gets a naming view.
        views = dict(Retriever._node_rerank_views(_node(NodeType.IMAGE, summary="")))

        self.assertEqual(HEADING, views["named"])
        self.assertIn("interface elements", views["body"])

    def test_a_figure_with_neither_heading_nor_summary_stays_one_document(self):
        views = Retriever._node_rerank_views(
            _node(NodeType.IMAGE, summary="", heading="")
        )

        self.assertEqual(1, len(views))
        self.assertIn("interface elements", views[0][1])

    def test_a_figure_with_no_description_stays_one_document(self):
        views = Retriever._node_rerank_views(_node(NodeType.IMAGE, content=""))

        self.assertEqual(1, len(views))
        self.assertIn(SUMMARY, views[0][1])

    def test_nothing_is_lost_against_the_single_document_projection(self):
        # Every word the one-document projection carried must still be scored
        # somewhere, or splitting has traded dilution for a blind spot.
        node = _node(NodeType.IMAGE)
        whole = Retriever._node_rerank_text(node)
        joined = " ".join(text for _kind, text in Retriever._node_rerank_views(node))

        for line in whole.splitlines():
            self.assertIn(line.strip(), joined)


class TheBestReadingDecidesTheScore(unittest.TestCase):
    """Max, not mean: averaging reintroduces the dilution this splits apart."""

    class _Reranker:
        max_length = 4096

        def __init__(self, by_text):
            self.by_text = by_text
            self.seen = []

        def rerank(self, *, query, documents, instruction):
            self.seen = list(documents)
            return [
                next(
                    (score for token, score in self.by_text.items() if token in doc),
                    0.0,
                )
                for doc in documents
            ]

        def clean_cache(self):
            pass

    def _retriever(self, reranker):
        retriever = object.__new__(Retriever)
        retriever.reranker = reranker
        retriever.timings = SimpleNamespace(time=lambda _name: _Quiet())
        return retriever

    def test_a_node_scores_as_well_as_its_best_reading(self):
        reranker = self._Reranker({"Zernike surface, whose": 1.0, "interface": 0.1})
        retriever = self._retriever(reranker)
        node = _node(NodeType.IMAGE)

        with _patched_candidates(retriever, [node]):
            ranked = retriever.text_reranker([node], "Zernike Surface image")

        self.assertEqual([(8297, 1.0)], ranked)

    def test_the_weak_reading_does_not_drag_it_down(self):
        # Under one joined document this node scored the dilution; under the
        # mean it would score 0.55. It scores its naming view.
        reranker = self._Reranker({"Zernike surface, whose": 1.0, "interface": 0.1})
        retriever = self._retriever(reranker)
        node = _node(NodeType.IMAGE)

        with _patched_candidates(retriever, [node]):
            ranked = retriever.text_reranker([node], "Zernike Surface image")

        self.assertNotAlmostEqual(0.55, ranked[0][1], places=2)
        self.assertEqual(2, len(reranker.seen), "both readings were scored")

    def test_a_question_about_the_picture_still_reaches_the_body(self):
        reranker = self._Reranker({"interface": 0.9, "Zernike surface, whose": 0.2})
        retriever = self._retriever(reranker)
        node = _node(NodeType.IMAGE)

        with _patched_candidates(retriever, [node]):
            ranked = retriever.text_reranker([node], "what does the icon look like")

        self.assertAlmostEqual(0.9, ranked[0][1])


class _Quiet:
    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


class _patched_candidates:
    """Feed text_reranker a fixed candidate list without a vector store."""

    def __init__(self, retriever, candidates):
        self.retriever = retriever
        self.candidates = candidates

    def __enter__(self):
        self.original = Retriever._hybrid_candidates
        Retriever._hybrid_candidates = lambda *_a, **_k: self.candidates
        return self

    def __exit__(self, *_exc):
        Retriever._hybrid_candidates = self.original
        return False


if __name__ == "__main__":
    unittest.main()
