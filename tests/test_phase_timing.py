"""Durations, not call counts, are what says where a stage's time goes.

The graph stage reranked on 85.4% of resolutions, and that number was read as a
cost share it never measured. These tests pin the accumulator's contract: time
is attributed even when a call fails, concurrent work is summed rather than
wall-clocked, and the summary orders phases by cost.
"""

from __future__ import annotations

import unittest
from concurrent.futures import ThreadPoolExecutor

from retrieval_engine.utils.timing import PhaseTimings, PhaseTotal


class ConstructionTests(unittest.TestCase):
    def test_a_stage_needs_a_name(self):
        for bad in ("", "   ", None, 5):
            with self.assertRaises(ValueError):
                PhaseTimings(bad)

    def test_the_stage_name_is_stripped(self):
        self.assertEqual("graph", PhaseTimings("  graph  ").stage)


class AccumulationTests(unittest.TestCase):
    def test_repeated_calls_sum_into_one_phase(self):
        timings = PhaseTimings("graph")

        timings.add("rerank", 1.5)
        timings.add("rerank", 2.5)

        self.assertEqual(PhaseTotal(calls=2, seconds=4.0), timings.totals()["rerank"])

    def test_the_mean_is_reported_per_call(self):
        timings = PhaseTimings("graph")

        timings.add("rerank", 3.0, calls=4)

        self.assertAlmostEqual(0.75, timings.totals()["rerank"].mean_seconds)

    def test_a_phase_that_never_ran_has_no_mean(self):
        self.assertEqual(0.0, PhaseTotal(calls=0, seconds=0.0).mean_seconds)

    def test_a_phase_needs_a_name_and_a_non_negative_duration(self):
        timings = PhaseTimings("graph")

        with self.assertRaises(ValueError):
            timings.add("  ", 1.0)
        with self.assertRaises(ValueError):
            timings.add("rerank", -0.1)

    def test_totals_are_ordered_by_cost(self):
        timings = PhaseTimings("graph")

        timings.add("vector_search", 1.0)
        timings.add("rerank", 9.0)
        timings.add("llm_merge", 4.0)

        self.assertEqual(
            ["rerank", "llm_merge", "vector_search"], list(timings.totals())
        )


class ContextManagerTests(unittest.TestCase):
    def test_the_timed_block_is_attributed_to_its_phase(self):
        timings = PhaseTimings("graph")

        with timings.time("rerank"):
            pass

        self.assertEqual(1, timings.totals()["rerank"].calls)

    def test_a_failing_call_still_records_the_time_it_spent(self):
        # A stage that fails slowly must not be measured as fast.
        timings = PhaseTimings("graph")

        with self.assertRaises(RuntimeError):
            with timings.time("llm_merge"):
                raise RuntimeError("budget exhausted")

        self.assertEqual(1, timings.totals()["llm_merge"].calls)


class ConcurrencyTests(unittest.TestCase):
    def test_concurrent_calls_are_all_counted(self):
        timings = PhaseTimings("tree")

        def work(_: int) -> None:
            with timings.time("summary"):
                pass

        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(work, range(400)))

        self.assertEqual(400, timings.totals()["summary"].calls)


class SummaryTests(unittest.TestCase):
    def test_as_dict_reports_calls_and_seconds_per_phase(self):
        timings = PhaseTimings("graph")

        timings.add("rerank", 4.0, calls=2)

        self.assertEqual(
            {"calls": 2, "seconds": 4.0, "mean_seconds": 2.0},
            timings.as_dict()["rerank"],
        )

    def test_an_uninstrumented_stage_logs_without_raising(self):
        PhaseTimings("graph").log_summary()

    def test_a_summary_reports_every_phase_that_ran(self):
        timings = PhaseTimings("graph")
        timings.add("rerank", 4.0)
        timings.add("vector_search", 1.0)

        with self.assertLogs("retrieval_engine.utils.timing", level="INFO") as captured:
            timings.log_summary()

        line = captured.output[0]
        self.assertIn("graph timings", line)
        self.assertIn("rerank", line)
        self.assertIn("vector_search", line)


if __name__ == "__main__":
    unittest.main()
