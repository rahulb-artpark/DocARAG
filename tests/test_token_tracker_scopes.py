from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
import unittest

from Core.provider.TokenTracker import TokenTracker, TokenUsageScope
from Core.utils.bounded_executor import iter_bounded_ordered


class TokenTrackerScopeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tracker = TokenTracker.get_instance()
        # No scope is active in unittest's calling context, so this clears only
        # the documented compatibility ledger.
        self.tracker.reset()

    def tearDown(self) -> None:
        self.tracker.reset()

    def test_legacy_facade_remains_available_without_an_explicit_scope(self):
        self.tracker.add_usage(3, 2)

        self.assertEqual(
            {
                "prompt_tokens": 3,
                "completion_tokens": 2,
                "total_tokens": 5,
            },
            self.tracker.get_usage(),
        )
        self.assertEqual(
            {
                "prompt_tokens": 3,
                "completion_tokens": 2,
                "total_tokens": 5,
            },
            self.tracker.record_stage("legacy"),
        )

    def test_nested_scopes_restore_the_parent_without_cross_charging(self):
        self.tracker.add_usage(1, 1)
        outer = self.tracker.scope(name="outer")
        inner = self.tracker.scope(name="inner")

        with outer:
            self.tracker.add_usage(10, 2)
            with inner:
                self.tracker.add_usage(100, 20)
            self.tracker.add_usage(3, 4)

        self.assertEqual(19, outer.get_usage()["total_tokens"])
        self.assertEqual(120, inner.get_usage()["total_tokens"])
        self.assertEqual(2, self.tracker.get_usage()["total_tokens"])

    def test_injected_scope_uses_the_existing_provider_facade(self):
        scope = TokenUsageScope(name="injected")
        with self.tracker.use_scope(scope):
            self.tracker.add_usage(7, 5)

        self.assertEqual(12, scope.get_usage()["total_tokens"])
        self.assertEqual(0, self.tracker.get_usage()["total_tokens"])

    def test_concurrent_scopes_have_independent_totals_and_stage_deltas(self):
        barrier = Barrier(2)

        def account(prompt: int, completion: int):
            scope = self.tracker.scope(name=f"run-{prompt}")
            with scope:
                barrier.wait()
                for _ in range(40):
                    self.tracker.add_usage(prompt, completion)
                first = self.tracker.record_stage("first")
                barrier.wait()
                for _ in range(15):
                    self.tracker.add_usage(prompt + 1, completion + 1)
                second = self.tracker.record_stage("second")
            return scope, first, second

        with ThreadPoolExecutor(max_workers=2) as executor:
            left_future = executor.submit(account, 2, 1)
            right_future = executor.submit(account, 7, 3)
            left = left_future.result()
            right = right_future.result()

        left_scope, left_first, left_second = left
        right_scope, right_first, right_second = right
        self.assertEqual(120, left_first["total_tokens"])
        self.assertEqual(75, left_second["total_tokens"])
        self.assertEqual(195, left_scope.get_usage()["total_tokens"])
        self.assertEqual(400, right_first["total_tokens"])
        self.assertEqual(180, right_second["total_tokens"])
        self.assertEqual(580, right_scope.get_usage()["total_tokens"])
        self.assertEqual(0, self.tracker.get_usage()["total_tokens"])

    def test_bounded_worker_threads_inherit_the_run_scope(self):
        scope = self.tracker.scope(name="bounded-workers")

        def worker(value: int) -> int:
            self.tracker.add_usage(value, 1)
            return value * 2

        with scope:
            outcomes = list(
                iter_bounded_ordered(
                    range(1, 31),
                    worker,
                    max_workers=4,
                    max_in_flight=7,
                )
            )

        self.assertTrue(all(outcome.ok for outcome in outcomes))
        self.assertEqual(list(range(2, 61, 2)), [item.value for item in outcomes])
        self.assertEqual(sum(range(1, 31)), scope.get_usage()["prompt_tokens"])
        self.assertEqual(30, scope.get_usage()["completion_tokens"])
        # A missing ContextVar propagation would charge these to the fallback.
        self.assertEqual(0, self.tracker.get_usage()["total_tokens"])

    def test_stage_history_is_a_detached_snapshot(self):
        scope = self.tracker.scope()
        scope.add_usage(4, 1)
        scope.record_stage("parse")

        snapshot = scope.stage_history
        snapshot["parse"]["total_tokens"] = 999

        self.assertEqual(5, scope.stage_history["parse"]["total_tokens"])

    def test_invalid_token_counts_fail_before_mutating_the_scope(self):
        scope = self.tracker.scope()

        with self.assertRaises(TypeError):
            scope.add_usage(1.5, 2)
        with self.assertRaises(ValueError):
            scope.add_usage(-1, 2)

        self.assertEqual(0, scope.get_usage()["total_tokens"])


if __name__ == "__main__":
    unittest.main()
