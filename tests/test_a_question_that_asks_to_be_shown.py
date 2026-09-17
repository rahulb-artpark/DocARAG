"""A question that asks to be shown a figure should get one.

Named vectors made the modality expressible; nothing acted on it. "show me the
image of Wolter Surface, its parameters and the equation that defines it"
returned twenty text references and no picture, and each of the four legs
ranked the figure on how well it matched -- never on the fact that a figure was
what the question asked for.

So a requested kind of content gets its own leg per vector, restricted to that
kind. Two properties matter and are tested here.

**It is added, never substituted.** The same question also asked for the
parameters and the equation, and the answer needed the paragraph and the
parameter table beside the figure. A filter would have deleted them, so the
unrestricted legs stay exactly as they were.

**Refusing a request is the hard half.** In a technical document the trigger
words are the subject matter: this corpus uses "image" 3913 times, at least 1622
of them in "image surface", "image space", "image height" -- optics, not a
request to be shown a picture. That is why the request comes from the planner,
which reads intent, rather than from spotting a word, and why the prompt carries
the negative cases explicitly.
"""

from __future__ import annotations

import unittest

from qdrant_client import models as real_models

from retrieval_engine.indexes.tree import NodeType
from retrieval_engine.indexes.vector_store import (
    TREE_NODE_PAYLOAD_SCHEMA,
    TREE_SCOPED_VECTOR_SUFFIX,
    VectorSchema,
    VectorStore,
    tree_vector_base_name,
)
from retrieval_engine.rag.gbc.planner import (
    REQUESTED_NODE_TYPE_FAMILIES,
    PlanResult,
    SubQuestion,
    TaskPlanner,
    requested_node_type_filter,
)
from retrieval_engine.rag.gbc.retrieval import Retriever
from tests.qdrant_fakes import FakeModels, FakeQdrantClient


class _Embedder:
    MM_EMBEDDER = True

    def embed_texts(self, texts):
        return [[0.1, 0.2] for _ in texts]

    def close(self):
        pass


def _store(models=FakeModels):
    client = FakeQdrantClient()
    store = VectorStore(
        _Embedder(),
        collection_name="c",
        client=client,
        models=models,
        vector_schema=VectorSchema.TREE_NODE,
    )
    client.create_collection(
        collection_name="c",
        vectors_config={
            name: models.VectorParams(size=2, distance=models.Distance.COSINE)
            for name in ("text", "context", "image")
        },
        sparse_vectors_config={
            "lexical": models.SparseVectorParams(
                index=models.SparseIndexParams(), modifier=models.Modifier.IDF
            )
        },
    )
    for index, node_type in enumerate(("text", "image", "chart", "equation", "table")):
        client.upsert(
            collection_name="c",
            points=[
                models.PointStruct(
                    id=f"record_node_{index}",
                    vector={
                        "text": [0.1, 0.2],
                        "context": [0.1, 0.2],
                        "image": [0.1, 0.2],
                        "lexical": models.SparseVector(indices=[7], values=[1.0]),
                    },
                    payload={
                        "_docarag_schema": TREE_NODE_PAYLOAD_SCHEMA,
                        "_docarag_record_id": f"record_node_{index}",
                        "_docarag_content": f"content {index}",
                        "_docarag_contents": {
                            "text": f"content {index}",
                            "context": f"section of {index}",
                            "image": f"/crops/{index}.jpg",
                        },
                        "node_id": index,
                        "node_type": node_type,
                    },
                )
            ],
            wait=True,
        )
    return store, client


class TheFamiliesAreWhatAReaderMeans(unittest.TestCase):
    def test_asking_for_an_image_covers_charts_and_diagrams(self):
        # Whether a picture was labelled image, chart or diagram is a fact
        # about the extractor, not about the question.
        self.assertEqual(
            ("image", "chart", "diagram"), requested_node_type_filter(["image"])
        )

    def test_asking_for_code_covers_algorithms(self):
        self.assertEqual(("code", "algorithm"), requested_node_type_filter(["code"]))

    def test_an_equation_is_its_own_kind(self):
        self.assertEqual(("equation",), requested_node_type_filter(["equation"]))

    def test_two_families_merge_without_repeating_a_type(self):
        types = requested_node_type_filter(["image", "equation", "image"])

        self.assertEqual(("image", "chart", "diagram", "equation"), types)

    def test_asking_for_a_table_scopes_nothing(self):
        # A table, a list and a paragraph are the same content in three
        # formats, so that family is the default population. Scoping to it
        # could not lift anything, and a second leg over the same nodes would
        # only double their weight in a rank-based fusion.
        self.assertEqual((), requested_node_type_filter(["table"]))

    def test_nothing_requested_scopes_nothing(self):
        for value in (None, (), [], "image", 7):
            with self.subTest(value=value):
                self.assertEqual((), requested_node_type_filter(value))

    def test_every_family_maps_only_to_real_node_types(self):
        known = {kind.value for kind in NodeType}
        for family, types in REQUESTED_NODE_TYPE_FAMILIES.items():
            with self.subTest(family=family):
                self.assertEqual(set(), set(types) - known)


class TheScopedLegsAreAdded(unittest.TestCase):
    def test_no_request_leaves_exactly_the_four_legs(self):
        store, client = _store()

        rows = store.search_tree_vectors("q", top_k=5, lexical_terms={7: 1.0})

        self.assertEqual({"text", "context", "image", "lexical"}, set(rows))
        self.assertEqual(4, len(client.batch_query_calls[0]["requests"]))

    def test_a_request_adds_one_scoped_leg_per_vector(self):
        store, client = _store()

        rows = store.search_tree_vectors(
            "q", top_k=5, lexical_terms={7: 1.0}, scoped_node_types=("image", "chart")
        )

        self.assertEqual(
            {
                "text",
                "context",
                "image",
                "lexical",
                f"text{TREE_SCOPED_VECTOR_SUFFIX}",
                f"context{TREE_SCOPED_VECTOR_SUFFIX}",
                f"image{TREE_SCOPED_VECTOR_SUFFIX}",
                f"lexical{TREE_SCOPED_VECTOR_SUFFIX}",
            },
            set(rows),
        )
        self.assertEqual(8, len(client.batch_query_calls[0]["requests"]))

    def test_one_round_trip_either_way(self):
        store, client = _store()

        store.search_tree_vectors(
            "q", top_k=5, lexical_terms={7: 1.0}, scoped_node_types=("image",)
        )

        self.assertEqual(1, len(client.batch_query_calls))

    def test_the_scoped_leg_returns_only_the_requested_kinds(self):
        store, _ = _store()

        rows = store.search_tree_vectors(
            "q", top_k=10, lexical_terms={7: 1.0}, scoped_node_types=("image", "chart")
        )

        scoped = rows[f"text{TREE_SCOPED_VECTOR_SUFFIX}"]
        self.assertTrue(scoped)
        self.assertEqual(
            {"image", "chart"}, {row["metadata"]["node_type"] for row in scoped}
        )

    def test_the_unrestricted_legs_are_untouched_by_the_request(self):
        # The question also asked for the parameters and the equation; a filter
        # would have deleted the paragraph that answers them.
        store, _ = _store()

        plain = store.search_tree_vectors("q", top_k=10, lexical_terms={7: 1.0})
        scoped = store.search_tree_vectors(
            "q", top_k=10, lexical_terms={7: 1.0}, scoped_node_types=("image",)
        )

        for name in ("text", "context", "image", "lexical"):
            with self.subTest(name=name):
                self.assertEqual(
                    [row["id"] for row in plain[name]],
                    [row["id"] for row in scoped[name]],
                )

    def test_a_scoped_hit_keeps_the_row_id_of_the_vector_it_matched(self):
        # A scoped and an unscoped hit on one vector are the same evidence.
        # Different row ids would stop them collapsing into one candidate.
        store, _ = _store()

        rows = store.search_tree_vectors(
            "q", top_k=10, lexical_terms={7: 1.0}, scoped_node_types=("image",)
        )
        scoped = {row["id"] for row in rows[f"image{TREE_SCOPED_VECTOR_SUFFIX}"]}

        self.assertTrue(scoped)
        self.assertTrue(scoped.issubset({row["id"] for row in rows["image"]}))

    def test_the_scoped_leg_still_carries_the_query_scope(self):
        store, client = _store()

        store.search_tree_vectors(
            "q", top_k=5, lexical_terms={7: 1.0}, scoped_node_types=("image",)
        )
        requests = client.batch_query_calls[0]["requests"]
        scoped = [r for r in requests if r.filter is not None]

        self.assertEqual(4, len(scoped))
        for request in scoped:
            keys = [getattr(c, "key", None) for c in request.filter.must]
            self.assertIn("node_type", keys)

    def test_the_request_validates_against_the_real_client(self):
        store, client = _store(models=real_models)

        store.search_tree_vectors(
            "q", top_k=5, lexical_terms={7: 1.0}, scoped_node_types=("image", "chart")
        )

        requests = client.batch_query_calls[0]["requests"]
        self.assertEqual(8, len(requests))
        self.assertEqual(
            ["context", "context", "image", "image", "lexical", "lexical", "text", "text"],
            sorted(r.using for r in requests),
        )


class TheTraceSaysWhichLegAnswered(unittest.TestCase):
    def test_a_scoped_leg_is_its_own_leg_in_the_trace(self):
        # A reader has to be able to tell whether a node arrived because it
        # matched or because its kind was asked for.
        plain = Retriever._leg_identity("image")
        scoped = Retriever._leg_identity(f"image{TREE_SCOPED_VECTOR_SUFFIX}")

        self.assertEqual(("dense-tree-image", "dense", "qdrant-tree-dense"), plain)
        self.assertEqual(
            ("typed-dense-tree-image", "dense", "qdrant-tree-dense-typed"), scoped
        )

    def test_the_scoped_sparse_leg_is_distinguished_too(self):
        self.assertEqual(
            ("typed-lexical-tree", "sparse", "qdrant-tree-sparse-typed"),
            Retriever._leg_identity(f"lexical{TREE_SCOPED_VECTOR_SUFFIX}"),
        )

    def test_no_two_legs_share_an_id(self):
        names = ("text", "context", "image", "lexical")
        ids = [Retriever._leg_identity(n)[0] for n in names]
        ids += [
            Retriever._leg_identity(f"{n}{TREE_SCOPED_VECTOR_SUFFIX}")[0] for n in names
        ]

        self.assertEqual(len(ids), len(set(ids)))

    def test_the_base_name_survives_the_marker(self):
        for name in ("text", "context", "image", "lexical"):
            with self.subTest(name=name):
                self.assertEqual(
                    name, tree_vector_base_name(f"{name}{TREE_SCOPED_VECTOR_SUFFIX}")
                )
                self.assertEqual(name, tree_vector_base_name(name))


class ThePlanIsWhereTheRequestComesFrom(unittest.TestCase):
    def test_a_simple_plan_may_ask_to_be_shown_something(self):
        plan = PlanResult(
            query_type="simple",
            original_query="show me the image of the Wolter Surface",
            requested_node_types=["image"],
        )

        self.assertEqual(["image"], plan.requested_node_types)

    def test_a_complex_plan_asks_per_sub_question(self):
        # "show me the image and the equation" is one query and two modalities.
        plan = PlanResult(
            query_type="complex",
            original_query="show me the image of X and the equation for it",
            sub_questions=[
                SubQuestion(
                    question="show me the image of X",
                    type="retrieval",
                    requested_node_types=["image"],
                ),
                SubQuestion(
                    question="what equation defines X",
                    type="retrieval",
                    requested_node_types=["equation"],
                ),
                SubQuestion(question="combine them", type="synthesis"),
            ],
        )

        self.assertEqual(
            [["image"], ["equation"], None],
            [sub.requested_node_types for sub in plan.sub_questions],
        )

    def test_a_complex_plan_may_not_also_ask_at_the_top_level(self):
        with self.assertRaises(ValueError):
            PlanResult(
                query_type="complex",
                original_query="q",
                sub_questions=[SubQuestion(question="a", type="retrieval")],
                requested_node_types=["image"],
            )

    def test_a_global_plan_says_it_through_filters_instead(self):
        with self.assertRaises(ValueError):
            PlanResult(
                query_type="global",
                original_query="how many images are in the document",
                filters=[{"filter_type": "image"}],
                operation="COUNT",
                requested_node_types=["image"],
            )

    def test_a_synthesis_step_cannot_request_a_modality(self):
        # It does no lookup, so the request would scope a retrieval that never
        # happens.
        with self.assertRaises(ValueError):
            SubQuestion(
                question="combine them",
                type="synthesis",
                requested_node_types=["image"],
            )

    def test_an_empty_list_is_rejected_rather_than_treated_as_no_request(self):
        with self.assertRaises(ValueError):
            SubQuestion(question="a", type="retrieval", requested_node_types=[])

    def test_a_repeated_type_is_rejected(self):
        with self.assertRaises(ValueError):
            SubQuestion(
                question="a", type="retrieval", requested_node_types=["image", "image"]
            )

    def test_an_unknown_family_is_rejected(self):
        with self.assertRaises(ValueError):
            SubQuestion(question="a", type="retrieval", requested_node_types=["title"])


class ThePromptTeachesTheRefusal(unittest.TestCase):
    """The negative cases are the load-bearing half and must not be dropped.

    Without them the model keys on the word, and in this corpus the word is
    ordinary optics vocabulary 41% of the time.
    """

    def setUp(self):
        self.planner = TaskPlanner(llm=object())
        self.rules = self.planner._requested_node_types_rules()

    def test_the_optics_negatives_are_spelled_out(self):
        for phrase in ("paraxial image plane", "image height"):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, self.rules)

    def test_it_says_the_word_is_not_the_signal(self):
        self.assertIn("NOT the signal", self.rules)

    def test_it_says_to_omit_rather_than_send_an_empty_list(self):
        self.assertIn("Omit the field entirely", self.rules)

    def test_both_prompts_carry_the_rules(self):
        for prompt in (
            self.planner._get_classification_prompt("q"),
            self.planner._get_complex_prompt("q"),
        ):
            with self.subTest(prompt=prompt[:40]):
                self.assertIn("REQUESTED NODE TYPES", prompt)
                self.assertIn("paraxial image plane", prompt)

    def test_only_the_four_families_are_offered(self):
        for family in REQUESTED_NODE_TYPE_FAMILIES:
            with self.subTest(family=family):
                self.assertIn(f'"{family}"', self.rules)


if __name__ == "__main__":
    unittest.main()
