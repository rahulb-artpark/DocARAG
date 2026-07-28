from __future__ import annotations

from types import SimpleNamespace
import unittest

from Core.prompts.gbc_prompt import QuestionEntity
from Core.rag.gbc_rag import GBCRAG


class _EntityStore:
    def __init__(self, results=None) -> None:
        self.results = [] if results is None else results
        self.queries = []

    def search_entities(self, *, name_text, top_k, **kwargs):
        self.queries.append((name_text, top_k))
        return self.results


class _FailingLLM:
    def get_json_completion(self, *_args, **_kwargs):
        raise AssertionError("the LLM must not be called for an empty query")


class QueryEntitySafetyTests(unittest.TestCase):
    def _rag(self, store):
        rag = GBCRAG.__new__(GBCRAG)
        rag.gbc_index = SimpleNamespace(
            entity_vdb=store,
            GraphIndex=SimpleNamespace(),
        )
        rag.llm = _FailingLLM()
        rag.threshold_e = 0.5
        return rag

    def test_blank_query_returns_no_mapping_without_model_or_store_calls(self):
        store = _EntityStore()
        rag = self._rag(store)

        self.assertEqual({}, rag._get_query_entity("  \n\t"))
        self.assertEqual([], store.queries)

    def test_entity_mapping_handles_empty_vector_results(self):
        store = _EntityStore(results=[])
        rag = self._rag(store)
        entity = QuestionEntity(entity_name="missing", entity_type="CONCEPT")

        self.assertEqual({}, rag._entity_map([entity], force_one=True))
        self.assertEqual(1, len(store.queries))

    def test_force_one_never_overrides_the_distance_threshold(self):
        store = _EntityStore(
            results=[
                {
                    "distance": 0.9,
                    "metadata": {
                        "entity_name": "unrelated",
                        "entity_type": "CONCEPT",
                    },
                }
            ]
        )
        rag = self._rag(store)
        rag.gbc_index.GraphIndex = SimpleNamespace(
            get_node_name_from_str=lambda name, kind: f"{name} ({kind})"
        )
        entity = QuestionEntity(entity_name="missing", entity_type="CONCEPT")

        self.assertEqual({}, rag._entity_map([entity], force_one=True))
        self.assertEqual(1, len(store.queries))


if __name__ == "__main__":
    unittest.main()
