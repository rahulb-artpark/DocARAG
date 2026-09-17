from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import logging
from pathlib import Path
import stat
import tempfile
import threading
import unittest

from document_services.operations.run_logging import RunLogScope, current_logging_run_id


class RunLogScopeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.logger = logging.getLogger("docarag_tests.run_logging")
        self.original_propagate = self.logger.propagate
        self.logger.propagate = False

    def tearDown(self) -> None:
        self.logger.propagate = self.original_propagate

    def test_scope_writes_run_id_and_does_not_touch_root_handlers(self) -> None:
        root_handlers = tuple(logging.getLogger().handlers)
        original_level = self.logger.level

        with tempfile.TemporaryDirectory() as directory:
            with RunLogScope(
                directory,
                logger_names=(self.logger.name,),
            ) as info:
                self.assertEqual(info.run_id, current_logging_run_id())
                self.logger.info("scoped message")

            content = info.log_file.read_text(encoding="utf-8")
            log_mode = stat.S_IMODE(info.log_file.stat().st_mode)
            directory_mode = stat.S_IMODE(info.log_file.parent.stat().st_mode)

        self.assertIn(info.run_id, content)
        self.assertIn("scoped message", content)
        self.assertEqual(0o600, log_mode)
        self.assertEqual(0o700, directory_mode)
        self.assertIsNone(current_logging_run_id())
        self.assertEqual(original_level, self.logger.level)
        self.assertEqual(root_handlers, tuple(logging.getLogger().handlers))

    def test_nested_scopes_restore_outer_context_without_cross_writing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            outer_dir = Path(directory) / "outer"
            inner_dir = Path(directory) / "inner"
            with RunLogScope(
                outer_dir,
                logger_names=(self.logger.name,),
            ) as outer:
                self.logger.info("outer before")
                with RunLogScope(
                    inner_dir,
                    logger_names=(self.logger.name,),
                ) as inner:
                    self.logger.info("inner only")
                self.assertEqual(outer.run_id, current_logging_run_id())
                self.logger.info("outer after")

            outer_content = outer.log_file.read_text(encoding="utf-8")
            inner_content = inner.log_file.read_text(encoding="utf-8")

        self.assertIn("outer before", outer_content)
        self.assertIn("outer after", outer_content)
        self.assertNotIn("inner only", outer_content)
        self.assertIn("inner only", inner_content)
        self.assertNotIn("outer before", inner_content)

    def test_concurrent_threads_are_isolated(self) -> None:
        barrier = threading.Barrier(2)

        def worker(base: Path, marker: str) -> Path:
            with RunLogScope(
                base,
                logger_names=(self.logger.name,),
            ) as info:
                barrier.wait(timeout=5)
                self.logger.info("message %s", marker)
                barrier.wait(timeout=5)
                return info.log_file

        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            with ThreadPoolExecutor(max_workers=2) as executor:
                first_future = executor.submit(worker, base / "first", "first")
                second_future = executor.submit(worker, base / "second", "second")
                first = first_future.result(timeout=10).read_text(encoding="utf-8")
                second = second_future.result(timeout=10).read_text(encoding="utf-8")

        self.assertIn("message first", first)
        self.assertNotIn("message second", first)
        self.assertIn("message second", second)
        self.assertNotIn("message first", second)

    def test_exception_closes_handler_and_restores_context(self) -> None:
        original_handlers = tuple(self.logger.handlers)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "boom"):
                with RunLogScope(
                    directory,
                    logger_names=(self.logger.name,),
                ):
                    raise RuntimeError("boom")

        self.assertIsNone(current_logging_run_id())
        self.assertEqual(original_handlers, tuple(self.logger.handlers))

    def test_scope_is_single_use(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            scope = RunLogScope(
                directory,
                logger_names=(self.logger.name,),
                run_id="query-123",
            )
            with scope:
                self.assertEqual("query-123", current_logging_run_id())
            with self.assertRaisesRegex(RuntimeError, "single-use"):
                with scope:
                    pass

    def test_invalid_explicit_run_id_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "run_id must match"):
                RunLogScope(
                    directory,
                    logger_names=(self.logger.name,),
                    run_id="query\nforged",
                )

    def test_overlapping_logger_names_write_each_record_once(self) -> None:
        parent = logging.getLogger("docarag_tests.overlap")
        child = logging.getLogger("docarag_tests.overlap.child")
        original_parent_propagate = parent.propagate
        original_child_propagate = child.propagate
        parent.propagate = False
        child.propagate = True
        try:
            with tempfile.TemporaryDirectory() as directory:
                with RunLogScope(
                    directory,
                    logger_names=(parent.name, child.name, parent.name),
                ) as info:
                    child.info("one logical event")
                content = info.log_file.read_text(encoding="utf-8")
        finally:
            parent.propagate = original_parent_propagate
            child.propagate = original_child_propagate

        self.assertEqual(1, content.count("one logical event"))


if __name__ == "__main__":
    unittest.main()
