from __future__ import annotations

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
import tempfile
import unittest
from unittest.mock import Mock, patch

import pandas as pd

from retrieval_engine.runtime.inference import create_rag_session, run_rag
from retrieval_engine.providers.token_tracker import TokenTracker
from retrieval_engine.rag.session import RAGSession


class InferenceLifecycleTests(unittest.TestCase):
    def test_query_failure_still_closes_rag_resources(self):
        rag = Mock()
        rag.generation.side_effect = RuntimeError("query failed")

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "query failed"):
                run_rag(
                    rag,
                    Path(directory),
                    data_df=pd.DataFrame([{"question": "What is supported?"}]),
                )

        rag.close.assert_called_once_with()

    def test_legacy_default_takes_ownership_on_success(self):
        rag = Mock()
        rag.generation.return_value = ("answer", ["node"])

        with tempfile.TemporaryDirectory() as directory:
            run_rag(
                rag,
                Path(directory),
                data_df=pd.DataFrame([{"question": "What is supported?"}]),
            )

        rag.close.assert_called_once_with()

    def test_explicit_borrow_does_not_close_the_agent(self):
        rag = Mock()
        rag.generation.return_value = ("answer", ["node"])

        with tempfile.TemporaryDirectory() as directory:
            run_rag(
                rag,
                Path(directory),
                data_df=pd.DataFrame([{"question": "What is supported?"}]),
                close_agent_on_exit=False,
            )

        rag.close.assert_not_called()

    def test_injected_run_scope_captures_provider_usage(self):
        tracker = TokenTracker.get_instance()
        tracker.reset()
        scope = tracker.scope(name="request")
        rag = Mock()

        def generate(_query, _output_dir):
            TokenTracker.get_instance().add_usage(8, 3)
            return "answer", ["node"]

        rag.generation.side_effect = generate
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            run_rag(
                rag,
                output,
                data_df=pd.DataFrame([{"question": "What is supported?"}]),
                token_scope=scope,
            )
            saved = (output / "token_cost.json").read_text(encoding="utf-8")

        self.assertEqual(
            {"prompt_tokens": 8, "completion_tokens": 3, "total_tokens": 11},
            scope.get_usage(),
        )
        self.assertIn('"total_tokens": 11', saved)
        self.assertEqual(0, tracker.get_usage()["total_tokens"])

    def test_run_reuses_active_request_scope_and_keeps_prior_totals(self):
        tracker = TokenTracker.get_instance()
        tracker.reset()
        scope = tracker.scope(name="outer-request")
        rag = Mock()

        def generate(_query, _output_dir):
            TokenTracker.get_instance().add_usage(5, 1)
            return "answer", ["node"]

        rag.generation.side_effect = generate
        with tempfile.TemporaryDirectory() as directory, scope:
            tracker.add_usage(2, 1)
            tracker.record_stage("setup")
            run_rag(
                rag,
                Path(directory),
                data_df=pd.DataFrame([{"question": "What is supported?"}]),
            )

        self.assertEqual(9, scope.get_usage()["total_tokens"])
        self.assertEqual(6, scope.stage_history["rag_cost"]["total_tokens"])
        self.assertEqual(0, tracker.get_usage()["total_tokens"])

    def test_concurrent_runs_do_not_mix_token_cost_files_or_stage_deltas(self):
        tracker = TokenTracker.get_instance()
        tracker.reset()
        barrier = Barrier(2)

        def execute(directory: Path, prompt_tokens: int):
            rag = Mock()
            scope = tracker.scope(name=f"request-{prompt_tokens}")

            def generate(_query, _output_dir):
                barrier.wait()
                TokenTracker.get_instance().add_usage(prompt_tokens, 1)
                return "answer", ["node"]

            rag.generation.side_effect = generate
            run_rag(
                rag,
                directory,
                data_df=pd.DataFrame([{"question": "q"}]),
                token_scope=scope,
            )
            return scope, rag

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            left_dir = root / "left"
            right_dir = root / "right"
            left_dir.mkdir()
            right_dir.mkdir()
            with ThreadPoolExecutor(max_workers=2) as executor:
                left_future = executor.submit(execute, left_dir, 5)
                right_future = executor.submit(execute, right_dir, 50)
                left_scope, left_rag = left_future.result()
                right_scope, right_rag = right_future.result()

            left_cost = (left_dir / "token_cost.json").read_text(encoding="utf-8")
            right_cost = (right_dir / "token_cost.json").read_text(encoding="utf-8")

        self.assertEqual(6, left_scope.stage_history["rag_cost"]["total_tokens"])
        self.assertEqual(51, right_scope.stage_history["rag_cost"]["total_tokens"])
        self.assertIn('"total_tokens": 6', left_cost)
        self.assertIn('"total_tokens": 51', right_cost)
        left_rag.close.assert_called_once_with()
        right_rag.close.assert_called_once_with()
        self.assertEqual(0, tracker.get_usage()["total_tokens"])

    def test_session_closes_once_and_preserves_an_active_failure(self):
        rag = Mock()
        rag.close.side_effect = RuntimeError("cleanup failed")
        session = RAGSession(rag)

        with self.assertRaisesRegex(ValueError, "query failed"):
            with session:
                raise ValueError("query failed")

        session.close()
        rag.close.assert_called_once_with()
        self.assertTrue(session.closed)

    def test_session_surfaces_close_failure_without_an_active_failure(self):
        rag = Mock()
        rag.close.side_effect = RuntimeError("cleanup failed")
        session = RAGSession(rag)

        with self.assertRaisesRegex(RuntimeError, "cleanup failed"):
            with session:
                pass

        session.close()
        rag.close.assert_called_once_with()

    def test_config_factory_session_owns_agent_after_construction(self):
        cfg = Mock()
        cfg.rag.strategy_config = object()
        cfg.llm = object()
        cfg.vlm = object()
        rag = Mock()

        with (
            patch("retrieval_engine.runtime.inference.prepare_rag_dependencies", return_value={}),
            patch("retrieval_engine.runtime.inference.create_rag_agent", return_value=rag),
        ):
            session = create_rag_session(cfg)

        with self.assertRaisesRegex(RuntimeError, "later setup failed"):
            with session:
                raise RuntimeError("later setup failed")

        rag.close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
