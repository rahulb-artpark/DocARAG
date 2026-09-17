from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import sqlite3
import re
import tempfile
import unittest
from unittest.mock import Mock, patch

import networkx as nx
from retrieval_engine.indexes.tree import NodeType
from retrieval_engine.prompts.gbc import SecEXPSelection
from retrieval_engine.rag.gbc.planner import PlanResult, SubQuestion
from retrieval_engine.rag.gbc.retrieval import Retriever
from retrieval_engine.rag.gbc.agent import GBCRAG
from retrieval_engine.rag.gbc.context import GBCRAGContext, SubStep
from retrieval_engine.rag.stored_indexes import StoredGraphIndex, entity_node_name
from document_services.persistence.corpus_store import CorpusStore
from document_services.persistence.tree_store import GraphEdge
from tests.test_gbc_planner_bounds import _SequencedLLM, _section, SectionSelectionBoundTests


class ScopedSelectionTests(unittest.TestCase):
    def select(self, responses):
        llm = _SequencedLLM(*responses)
        rag = SectionSelectionBoundTests._rag(
            [_section(11, 'Primary'), _section(12, 'Primary too'), _section(42, 'Additional')], llm
        )
        rag.link_section = lambda _nodes, **kwargs: {11: ['a'], 12: ['b']}
        step = SubStep(sub_query='question', sub_number=1)
        rag.llm_section_selection('question', [], step)
        return step, llm

    def test_primary_ids_are_redundant_not_an_error(self):
        step, llm = self.select([SecEXPSelection(supplementary_ids=[11, 12, 42], explanation='relevant')])
        self.assertEqual([11, 12, 42], step.retrieval_sec_ids)
        self.assertEqual([42], step.supplementary_ids)
        schema = llm.calls[0][1].model_json_schema()
        self.assertEqual([11, 12, 42], schema['properties']['supplementary_ids']['items']['enum'])

    def test_unknown_id_gets_one_corrective_retry(self):
        step, llm = self.select([
            SecEXPSelection(supplementary_ids=[999], explanation='bad'),
            SecEXPSelection(supplementary_ids=[42], explanation='corrected'),
        ])
        self.assertEqual([11, 12, 42], step.retrieval_sec_ids)
        self.assertEqual(2, len(llm.calls))
        self.assertIn('CORRECTION', llm.calls[1][0])
        self.assertIn('[42]', llm.calls[1][0])

    def test_document_tree_mismatch_is_rejected_before_model(self):
        rag = SectionSelectionBoundTests._rag([_section(11, 'Allowed')], _SequencedLLM())
        tree = rag.gbc_index.TreeIndex
        rag.gbc_index = {'a': SimpleNamespace(TreeIndex=tree)}
        rag.corpus_bundle = SimpleNamespace(documents=rag.gbc_index)
        step = SubStep(sub_query='q', sub_number=1)
        with self.assertRaisesRegex(ValueError, 'does not belong'):
            rag.llm_section_selection('q', [], step, tree_index=object(), document_version_id='a')
        with self.assertRaisesRegex(ValueError, 'outside'):
            rag.llm_section_selection('q', [], step, tree_index=tree, document_version_id='outside')
        self.assertEqual([], rag.llm.calls)


class GroupGraphTests(unittest.TestCase):
    def rag(self, ranked=None):
        rag = GBCRAG.__new__(GBCRAG)
        rag.cfg = SimpleNamespace(cross_doc_edge_discount_factor=0.7)
        rag.variant = 'standard'
        rag.evidence_hydrator = None
        rag.evidence_hydrators = {}
        rag._corpus_store = None
        rag.gbc_index = {}
        for doc in ('a', 'b'):
            node = SimpleNamespace(document_version_id=doc, index_id=11)
            tree = SimpleNamespace(
                get_subtree_nodes=Mock(return_value=[node]),
                get_nodes_data=Mock(return_value=[{'document_version_id': doc, 'index_id': 11, 'type': NodeType.TEXT, 'content': doc}]),
            )
            graph = nx.Graph()
            graph.add_node('shared', source_ids={(doc, 11)})
            rag.gbc_index[doc] = SimpleNamespace(
                TreeIndex=tree, GraphIndex=SimpleNamespace(get_kg_subgraph=Mock(return_value=graph))
            )
        rag.corpus_bundle = SimpleNamespace(documents=rag.gbc_index, document_titles={})
        rag.retriever = SimpleNamespace(
            timings=SimpleNamespace(time=lambda _: nullcontext()),
            skyline_filter=Mock(return_value=(ranked if ranked is not None else [('a', 11), ('b', 11)], ['shared'])),
        )
        rag._with_section_titles = lambda ids, **kwargs: ids
        return rag

    def step(self):
        return SubStep(sub_query='q', sub_number=1, documents_in_scope=['a', 'b'], retrieval_sec_ids_by_doc={'a': [11], 'b': [11]})

    def test_equal_node_numbers_keep_document_identity_and_graph_sources(self):
        rag = self.rag()
        step = self.step()
        rag._get_GBC_info_corpus(step)
        graph = rag.retriever.skyline_filter.call_args.args[2]
        self.assertEqual({('a', 11), ('b', 11)}, graph.nodes['shared']['source_ids'])
        self.assertEqual(['a', 'b'], [row['document_version_id'] for row in step.retrieval_nodes])
        with tempfile.TemporaryDirectory() as directory:
            ids = rag._save_retrieval_res(GBCRAGContext(query='q', iterations=[step]), Path(directory))
        self.assertEqual([{'document_version_id': 'a', 'index_id': 11}, {'document_version_id': 'b', 'index_id': 11}], ids)

    def test_ranker_cannot_return_outside_document_or_unselected_node(self):
        for key in [('outside', 11), ('a', 999)]:
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'outside'):
                self.rag([key])._get_GBC_info_corpus(self.step())

    def test_mislabelled_tree_row_cannot_reach_answer(self):
        rag = self.rag()
        rag.gbc_index['a'].TreeIndex.get_nodes_data.return_value[0]['document_version_id'] = 'b'
        with self.assertRaisesRegex(ValueError, 'different document'):
            rag._get_GBC_info_corpus(self.step())

    def test_hydrator_cannot_introduce_outside_document(self):
        rag = self.rag()
        rag.evidence_hydrators = {'a': Mock()}
        rag.evidence_hydrators['a'].hydrate.return_value = [{'document_version_id': 'outside'}]
        with self.assertRaisesRegex(ValueError, 'different document'):
            rag._process_retrieved_nodes([{'document_version_id': 'a'}], self.step())

    def test_missing_or_outside_evidence_document_is_rejected(self):
        rag = self.rag()
        for row in [{}, {'document_version_id': 'outside'}]:
            with self.subTest(row=row), self.assertRaisesRegex(ValueError, 'outside'):
                rag._process_retrieved_nodes([row], self.step())

    def test_relation_lookup_receives_the_selected_document_allowlist(self):
        rag = self.rag()
        rag._corpus_store = Mock()
        rag._corpus_store.entity_key_for_canonical.return_value = 1
        rag._corpus_store.inferred_relations_for_entities.return_value = []
        rag.gbc_index['a'].GraphIndex.add_cross_document_edges = Mock()
        rag._get_GBC_info_corpus(self.step())
        rag._corpus_store.inferred_relations_for_entities.assert_called_once_with(
            {1}, document_version_ids=['a', 'b']
        )

    def test_outside_graph_source_is_rejected_before_ranking(self):
        rag = self.rag()
        graph = rag.gbc_index['a'].GraphIndex.get_kg_subgraph.return_value
        graph.nodes['shared']['source_ids'].add(('outside', 11))
        with self.assertRaisesRegex(ValueError, 'graph source node is outside'):
            rag._get_GBC_info_corpus(self.step())
        rag.retriever.skyline_filter.assert_not_called()

    def test_entity_hit_outside_group_is_rejected_before_prompting(self):
        rag = self.rag()
        rag.entity_vdb = Mock()
        rag.entity_vdb.search_entities.return_value = [{'metadata': {'document_version_id': 'outside'}}]
        with self.assertRaisesRegex(ValueError, 'entity search returned evidence outside'):
            rag._search_entities('query', 4, 'candidate search')

    def test_vector_hits_are_joined_by_document_and_node_together(self):
        retriever = Retriever.__new__(Retriever)
        retriever.candidate_pool_size = 10
        a, b = object(), object()
        rows = [{'metadata': {'document_version_id': doc, 'node_id': 11}} for doc in ['outside', 'a', 'b']]
        results = retriever._bounded_subtree_rows(rows, {('a', 11): a, ('b', 11): b})
        self.assertEqual([a, b], [node for node, _ in results])

    def test_only_scoped_subanswers_reach_final_synthesis(self):
        rag = self.rag()
        rag.retrieval_callback = None
        def retrieve(query, step):
            step.documents_in_scope = ['a', 'b']
            step.retrieval_sec_ids_by_doc = {'a': [11], 'b': [11]}
            rag._get_GBC_info_corpus(step)
        rag._retrieve = retrieve
        def answer(**kwargs):
            self.assertEqual({'a', 'b'}, {row['document_version_id'] for row in kwargs['retrieved_nodes']})
            return 'Findings from a and b', [], []
        rag.answer = SimpleNamespace(
            answer_simple_question=Mock(side_effect=answer),
            answer_complex_question_with_validation=Mock(return_value=('Combined answer', {})),
        )
        context = GBCRAGContext(query='Compare')
        rag.process_analysis(context, PlanResult(query_type='complex', original_query='Compare', sub_questions=[
            SubQuestion(question='Find facts', type='retrieval'),
            SubQuestion(question='Combine facts', type='synthesis'),
        ]))
        synthesis = rag.answer.answer_complex_question_with_validation.call_args.kwargs
        self.assertEqual(['Findings from a and b'], [result['answer'] for result in synthesis['sub_question_results']])
        self.assertEqual('Combined answer', context.final_answer)
        rag.retriever.skyline_filter.return_value = ([('outside', 11)], [])
        rag.answer.answer_simple_question.reset_mock()
        with self.assertRaisesRegex(ValueError, 'outside'):
            rag.process_analysis(GBCRAGContext(query='q'), PlanResult(query_type='simple', original_query='q'))
        rag.answer.answer_simple_question.assert_not_called()

    def test_cross_edges_cannot_add_unselected_entities(self):
        graph = nx.Graph()
        graph.add_nodes_from([entity_node_name('A', ''), entity_node_name('B', '')])
        StoredGraphIndex(None).add_cross_document_edges(graph, [
            GraphEdge('A', 'B', 'allowed'), GraphEdge('A', 'OUTSIDE', 'forbidden'),
        ])
        self.assertEqual(2, len(graph))
        self.assertEqual(1, graph.number_of_edges())


class CrossDocumentProvenanceTests(unittest.TestCase):
    def test_shared_endpoints_do_not_admit_outside_or_unprovenanced_relations(self):
        # Execute the actual relational query against an isolated SQL fixture;
        # translate only psycopg placeholders/ANY arrays to SQLite syntax.
        db = sqlite3.connect(':memory:')
        self.addCleanup(db.close)
        db.row_factory = sqlite3.Row
        db.execute("ATTACH DATABASE ':memory:' AS docarag")
        db.executescript('''
            CREATE TABLE docarag.cross_doc_statement (tenant_id, corpus_id, stmt_key, predicate, weight, description_blob_key);
            CREATE TABLE docarag.cross_doc_statement_role (tenant_id, corpus_id, stmt_key, role, entity_key);
            CREATE TABLE docarag.entity (entity_key, canonical, entity_kind);
            CREATE TABLE docarag.text_blob (blob_key, body);
            CREATE TABLE docarag.cross_doc_statement_property (tenant_id, corpus_id, stmt_key, name, text_val);
            INSERT INTO docarag.entity VALUES (1, 'A', ''), (2, 'B', '');
        ''')
        for key, pair in enumerate([('a', 'b'), ('a', 'outside'), ('outside', 'b'), (None, None)], 1):
            db.execute('INSERT INTO docarag.cross_doc_statement VALUES (?, ?, ?, ?, ?, ?)', ('tenant', 'corpus', key, f'relation-{key}', 1, None))
            for role, entity, doc in [('subject', 1, pair[0]), ('object', 2, pair[1])]:
                db.execute('INSERT INTO docarag.cross_doc_statement_role VALUES (?, ?, ?, ?, ?)', ('tenant', 'corpus', key, role, entity))
                if doc:
                    db.execute('INSERT INTO docarag.cross_doc_statement_property VALUES (?, ?, ?, ?, ?)', ('tenant', 'corpus', key, f'source_document_version_id_{role}', doc))

        def execute(sql, params):
            args = iter(params)
            bound = []
            def replace(match):
                value = next(args)
                if match.group(0).startswith('= ANY'):
                    bound.extend(value)
                    return 'IN (' + ','.join('?' for _ in value) + ')'
                bound.append(value)
                return '?'
            return db.execute(re.sub(r'= ANY\(%s\)|%s', replace, sql), bound)

        store = CorpusStore(lambda: None, 'tenant', 'corpus')
        connection = SimpleNamespace(execute=execute)
        with patch.object(CorpusStore, '_session', return_value=nullcontext(connection)):
            self.assertEqual(['relation-1'], [edge.predicate for edge in store.inferred_relations_for_entities([1, 2], document_version_ids=['a', 'b'])])
            self.assertEqual([], store.inferred_relations_for_entities([1, 2], document_version_ids=[]))
            self.assertEqual(4, len(store.inferred_relations_for_entities([1, 2])))
