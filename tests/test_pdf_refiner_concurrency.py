"""Judgment batches run concurrently but must land in batch order.

The three mergers spent ~6h15m of a 13h25m tree build issuing one model call
at a time against a provider provisioned for four. Batches are independent and
merging happens in a later pass, so only the calls overlap -- but the judgment
list is consumed positionally, so out-of-order completion would attach one
batch's verdicts to another batch's blocks.
"""

from __future__ import annotations

import threading
import time
import unittest

from retrieval_engine.parsing.pdf_refinement import _judge_batches
from retrieval_engine.prompts.refinement import StitchingJudgment, StitchingJudgmentsResponse


class _ScriptedLLM:
    """Answers each batch with its own verdicts, slowest batch first."""

    def __init__(self, *, max_workers=4, delays=None, fail_on=()):
        self.max_workers = max_workers
        self.max_in_flight = max_workers
        self.delays = delays or {}
        self.fail_on = set(fail_on)
        self.concurrent = 0
        self.peak_concurrent = 0
        self._lock = threading.Lock()

    def get_json_completion(self, *, prompt, schema):
        with self._lock:
            self.concurrent += 1
            self.peak_concurrent = max(self.peak_concurrent, self.concurrent)
        try:
            marker = prompt.strip().splitlines()[-1]
            time.sleep(self.delays.get(marker, 0))
            if marker in self.fail_on:
                raise RuntimeError(f"model refused {marker}")
            return StitchingJudgmentsResponse(
                judgments=[StitchingJudgment(stitched_pdf_ids=[int(marker)], explanation="ok")]
            )
        finally:
            with self._lock:
                self.concurrent -= 1


def _judge(llm, markers):
    return _judge_batches(
        [(1, marker) for marker in markers],
        llm,
        "template\n{json_text}",
        StitchingJudgmentsResponse,
        log_label="text",
        format_key="json_text",
    )


class JudgeBatchesOrderingTests(unittest.TestCase):
    def test_verdicts_arrive_in_batch_order_not_completion_order(self):
        # Batch "1" is slowest, so completion order is the reverse of input.
        llm = _ScriptedLLM(delays={"1": 0.20, "2": 0.10, "3": 0.0})
        judgments, unresolved = _judge(llm, ["1", "2", "3"])

        self.assertEqual([], unresolved)
        self.assertEqual(
            [[1], [2], [3]], [j.stitched_pdf_ids for j in judgments]
        )

    def test_batches_actually_overlap(self):
        llm = _ScriptedLLM(max_workers=4, delays={m: 0.10 for m in "1234"})
        _judge(llm, ["1", "2", "3", "4"])
        self.assertGreater(llm.peak_concurrent, 1)

    def test_a_single_worker_still_works(self):
        llm = _ScriptedLLM(max_workers=1)
        judgments, unresolved = _judge(llm, ["1", "2", "3"])
        self.assertEqual([], unresolved)
        self.assertEqual([[1], [2], [3]], [j.stitched_pdf_ids for j in judgments])
        self.assertEqual(1, llm.peak_concurrent)

    def test_a_missing_worker_count_falls_back_to_serial(self):
        class _Minimal(_ScriptedLLM):
            def __init__(self):
                super().__init__()
                del self.max_workers
                del self.max_in_flight

        judgments, unresolved = _judge(_Minimal(), ["1", "2"])
        self.assertEqual([], unresolved)
        self.assertEqual([[1], [2]], [j.stitched_pdf_ids for j in judgments])


class JudgeBatchesFailureTests(unittest.TestCase):
    def test_a_failed_batch_is_reported_without_losing_the_others(self):
        llm = _ScriptedLLM(fail_on={"2"})
        judgments, unresolved = _judge(llm, ["1", "2", "3"])

        self.assertEqual(["2"], unresolved)
        self.assertEqual([[1], [3]], [j.stitched_pdf_ids for j in judgments])

    def test_a_length_mismatch_is_treated_as_a_failure(self):
        class _TooMany(_ScriptedLLM):
            def get_json_completion(self, *, prompt, schema):
                return StitchingJudgmentsResponse(
                    judgments=[
                        StitchingJudgment(stitched_pdf_ids=[1], explanation="ok"),
                        StitchingJudgment(stitched_pdf_ids=[2], explanation="ok"),
                    ]
                )

        judgments, unresolved = _judge(_TooMany(), ["1"])
        self.assertEqual(["1"], unresolved)
        self.assertEqual([], judgments)

    def test_every_batch_failing_reports_all_of_them(self):
        llm = _ScriptedLLM(fail_on={"1", "2"})
        judgments, unresolved = _judge(llm, ["1", "2"])
        self.assertEqual(["1", "2"], unresolved)
        self.assertEqual([], judgments)


if __name__ == "__main__":
    unittest.main()
