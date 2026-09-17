from __future__ import annotations

import tempfile
import threading
import time
import unittest
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

from retrieval_engine.indexes.tree import DocumentTree, NodeType, TreeNode
from retrieval_engine.configuration.language_model import LLMConfig
from retrieval_engine.configuration.vision_model import VLMConfig
from retrieval_engine.indexing.tree.summary import (
    TreeSummaryGenerationError,
    generate_tree_node_summary,
)
from retrieval_engine.providers.language_model import (
    BatchCompletionError,
    CompletionAttemptsExhausted,
    CompletionRequest,
    LLM,
)
from retrieval_engine.providers.vision_model import (
    BatchGenerationError,
    GenerationAttemptsExhausted,
    GenerationRequest,
    VLM,
)
from retrieval_engine.utils.bounded_executor import iter_bounded_ordered


def make_llm(controller, **overrides) -> LLM:
    values = {
        "backend": "openai",
        "max_workers": 2,
        "max_in_flight": 2,
        "max_retries": 2,
        "retry_delay_seconds": 0,
    }
    values.update(overrides)
    config = LLMConfig(**values)
    llm = object.__new__(LLM)
    llm.config = config
    llm.max_workers = config.max_workers
    llm.max_in_flight = config.max_in_flight
    llm._admission = threading.BoundedSemaphore(config.max_workers)
    llm.llm = controller
    return llm


def make_vlm(controller, **overrides) -> VLM:
    values = {
        "backend": "ollama",
        "max_workers": 2,
        "max_in_flight": 2,
        "max_retries": 2,
        "retry_delay_seconds": 0,
    }
    values.update(overrides)
    config = VLMConfig(**values)
    vlm = object.__new__(VLM)
    vlm.config = config
    vlm.max_workers = config.max_workers
    vlm.max_in_flight = config.max_in_flight
    concurrency = 1 if config.backend == "qwen" else config.max_workers
    vlm._admission = threading.BoundedSemaphore(concurrency)
    vlm.vlm = controller
    return vlm


class BoundedExecutorTests(unittest.TestCase):
    def test_bounds_consumed_work_and_preserves_input_order(self):
        release = threading.Event()
        workers_started = threading.Event()
        window_filled = threading.Event()
        lock = threading.Lock()
        consumed = 0
        active = 0
        max_active = 0

        def items():
            nonlocal consumed
            for value in range(12):
                with lock:
                    consumed += 1
                    if consumed == 3:
                        window_filled.set()
                yield value

        def worker(value):
            nonlocal active, max_active
            with lock:
                active += 1
                max_active = max(max_active, active)
                if active == 2:
                    workers_started.set()
            self.assertTrue(release.wait(2), "test worker was not released")
            # Deliberately finish out of order; the iterator contract is ordered.
            if value % 2 == 0:
                time.sleep(0.005)
            with lock:
                active -= 1
            return value * 10

        outcomes = []
        errors = []

        def consume():
            try:
                outcomes.extend(
                    iter_bounded_ordered(
                        items(),
                        worker,
                        max_workers=2,
                        max_in_flight=3,
                    )
                )
            except Exception as exc:  # pragma: no cover - assertion aid
                errors.append(exc)

        thread = threading.Thread(target=consume)
        thread.start()
        self.assertTrue(workers_started.wait(2))
        # Worker threads can start immediately after the first two submissions;
        # do not race the producer while it is filling the third in-flight slot.
        self.assertTrue(window_filled.wait(2))
        with lock:
            self.assertEqual(consumed, 3)
            self.assertEqual(max_active, 2)
        release.set()
        thread.join(3)

        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual([result.index for result in outcomes], list(range(12)))
        self.assertEqual(
            [result.unwrap() for result in outcomes], [i * 10 for i in range(12)]
        )

    def test_cancellation_stops_consuming_the_source(self):
        release = threading.Event()
        started = threading.Event()
        cancel = threading.Event()
        consumed = 0
        lock = threading.Lock()

        def items():
            nonlocal consumed
            for value in range(100):
                with lock:
                    consumed += 1
                yield value

        def worker(value):
            started.set()
            self.assertTrue(release.wait(2), "test worker was not released")
            return value

        outcomes = []
        thread = threading.Thread(
            target=lambda: outcomes.extend(
                iter_bounded_ordered(
                    items(),
                    worker,
                    max_workers=2,
                    max_in_flight=4,
                    cancel_event=cancel,
                )
            )
        )
        thread.start()
        self.assertTrue(started.wait(2))
        cancel.set()
        release.set()
        thread.join(3)

        self.assertFalse(thread.is_alive())
        with lock:
            self.assertLessEqual(consumed, 4)
        self.assertLessEqual(len(outcomes), 1)

    def test_exceptions_are_typed_outcomes_not_text(self):
        def worker(value):
            if value == 1:
                raise LookupError("model unavailable")
            return f"value-{value}"

        outcomes = list(
            iter_bounded_ordered(
                range(3), worker, max_workers=2, max_in_flight=2
            )
        )

        self.assertTrue(outcomes[0].ok)
        self.assertFalse(outcomes[1].ok)
        self.assertIsInstance(outcomes[1].error, LookupError)
        self.assertIsNone(outcomes[1].value)
        self.assertTrue(outcomes[2].ok)


class FakeLLMController:
    def __init__(self):
        self.attempts = defaultdict(int)
        self.prompts = []
        self.lock = threading.Lock()

    def get_completion(self, prompt, json_response=False):
        with self.lock:
            self.attempts[prompt] += 1
            attempt = self.attempts[prompt]
            self.prompts.append(prompt)
        if prompt == "retry" and attempt == 1:
            raise ConnectionError("temporary")
        if prompt == "bad":
            raise ConnectionError("permanent")
        if prompt == "empty":
            return "  "
        if prompt == "slow":
            time.sleep(0.02)
        return f"answer-{prompt}"


class LLMExecutionTests(unittest.TestCase):
    def test_direct_calls_share_provider_wide_admission_limit(self):
        release = threading.Event()
        at_limit = threading.Event()
        lock = threading.Lock()

        class Controller:
            def __init__(self):
                self.active = 0
                self.maximum_active = 0

            def get_completion(self, prompt, json_response=False):
                with lock:
                    self.active += 1
                    self.maximum_active = max(self.maximum_active, self.active)
                    if self.active == 2:
                        at_limit.set()
                self_assert = release.wait(2)
                with lock:
                    self.active -= 1
                if not self_assert:
                    raise TimeoutError("test call was not released")
                return f"done:{prompt}"

        controller = Controller()
        llm = make_llm(
            controller,
            max_workers=2,
            max_in_flight=4,
            max_retries=1,
        )
        results = []
        threads = [
            threading.Thread(
                target=lambda value=index: results.append(
                    llm.get_completion(str(value))
                )
            )
            for index in range(6)
        ]
        for thread in threads:
            thread.start()
        self.assertTrue(at_limit.wait(2))
        time.sleep(0.01)
        self.assertEqual(controller.maximum_active, 2)
        release.set()
        for thread in threads:
            thread.join(2)

        self.assertTrue(all(not thread.is_alive() for thread in threads))
        self.assertEqual(len(results), 6)

    def test_stream_reports_ordered_success_retry_and_failure(self):
        controller = FakeLLMController()
        llm = make_llm(controller)
        requests = [
            CompletionRequest("first", "slow"),
            CompletionRequest("second", "retry"),
            CompletionRequest("third", "bad"),
            CompletionRequest("fourth", "empty"),
        ]

        outcomes = list(llm.iter_completion_results(requests))

        self.assertEqual(
            [outcome.key for outcome in outcomes],
            ["first", "second", "third", "fourth"],
        )
        self.assertEqual(outcomes[0].unwrap(), "answer-slow")
        self.assertEqual(outcomes[1].unwrap(), "answer-retry")
        self.assertFalse(outcomes[2].ok)
        self.assertIsInstance(outcomes[2].error, CompletionAttemptsExhausted)
        self.assertFalse(outcomes[3].ok)
        self.assertIsInstance(outcomes[3].error, CompletionAttemptsExhausted)
        self.assertEqual(controller.attempts["retry"], 2)
        self.assertEqual(controller.attempts["bad"], 2)
        self.assertEqual(controller.attempts["empty"], 2)

    def test_legacy_batch_raises_structured_error_instead_of_fake_response(self):
        llm = make_llm(FakeLLMController(), max_retries=1)

        with self.assertRaises(BatchCompletionError) as caught:
            llm.batch_get_completion((prompt for prompt in ["ok", "bad"]))

        self.assertEqual(len(caught.exception.failures), 1)
        failure = caught.exception.failures[0]
        self.assertEqual(failure.key, 1)
        self.assertIsNone(failure.value)
        self.assertNotIn("Error:", str(caught.exception))

    def test_config_rejects_unbounded_or_invalid_limits(self):
        with self.assertRaisesRegex(ValueError, "max_workers"):
            LLMConfig(max_workers=0)
        with self.assertRaisesRegex(ValueError, "max_in_flight"):
            LLMConfig(max_workers=2, max_in_flight=0)
        with self.assertRaisesRegex(ValueError, "max_retries"):
            LLMConfig(max_retries=0)


class FakeVLMController:
    def __init__(self):
        self.attempts = defaultdict(int)
        self.images = []

    def generate(self, prompt, images=None):
        self.attempts[prompt] += 1
        self.images.append(images)
        if prompt == "bad":
            raise OSError("vision server unavailable")
        if prompt == "retry" and self.attempts[prompt] == 1:
            raise OSError("temporary")
        if prompt == "slow":
            time.sleep(0.02)
        return f"vision-{prompt}"

    def generate_json(self, prompt, images=None, schema=None):
        return {"value": prompt}


class VLMExecutionTests(unittest.TestCase):
    def test_qwen_direct_text_and_json_calls_are_single_flight(self):
        release = threading.Event()
        entered = threading.Event()
        lock = threading.Lock()

        class Controller:
            def __init__(self):
                self.active = 0
                self.maximum_active = 0

            def _call(self, value):
                with lock:
                    self.active += 1
                    self.maximum_active = max(self.maximum_active, self.active)
                    entered.set()
                if not release.wait(2):
                    raise TimeoutError("test call was not released")
                with lock:
                    self.active -= 1
                return value

            def generate(self, prompt, images):
                return self._call(f"text:{prompt}")

            def generate_json(self, prompt, images, schema):
                return self._call({"value": prompt})

        controller = Controller()
        vlm = make_vlm(
            controller,
            backend="qwen",
            max_workers=4,
            max_in_flight=4,
            max_retries=1,
        )
        errors = []

        def text_call():
            try:
                vlm.generate("one")
            except Exception as error:  # pragma: no cover - assertion aid
                errors.append(error)

        def json_call():
            try:
                vlm.generate_json("two", schema=object())
            except Exception as error:  # pragma: no cover - assertion aid
                errors.append(error)

        threads = [threading.Thread(target=text_call), threading.Thread(target=json_call)]
        for thread in threads:
            thread.start()
        self.assertTrue(entered.wait(2))
        time.sleep(0.01)
        self.assertEqual(controller.maximum_active, 1)
        release.set()
        for thread in threads:
            thread.join(2)

        self.assertEqual(errors, [])
        self.assertTrue(all(not thread.is_alive() for thread in threads))

    def test_stream_is_ordered_and_has_typed_failures(self):
        controller = FakeVLMController()
        vlm = make_vlm(controller)
        outcomes = list(
            vlm.iter_generation_results(
                [
                    GenerationRequest("a", "slow", ["a.png"]),
                    GenerationRequest("b", "retry", ["b.png"]),
                    GenerationRequest("c", "bad", ["c.png"]),
                ]
            )
        )

        self.assertEqual([outcome.key for outcome in outcomes], ["a", "b", "c"])
        self.assertEqual(outcomes[0].unwrap(), "vision-slow")
        self.assertEqual(outcomes[1].unwrap(), "vision-retry")
        self.assertIsInstance(outcomes[2].error, GenerationAttemptsExhausted)
        self.assertEqual(controller.attempts["retry"], 2)

    def test_legacy_batch_normalizes_image_paths_and_never_returns_error_text(self):
        controller = FakeVLMController()
        vlm = make_vlm(controller, max_retries=1)

        self.assertEqual(
            vlm.batch_generate(["one", "two"], ["1.png", ["2.png"]]),
            ["vision-one", "vision-two"],
        )
        self.assertEqual(controller.images[:2], [["1.png"], ["2.png"]])

        with self.assertRaises(BatchGenerationError) as caught:
            vlm.batch_generate(["bad"], ["bad.png"])
        self.assertIsInstance(
            caught.exception.failures[0].error, GenerationAttemptsExhausted
        )
        self.assertIsNone(caught.exception.failures[0].value)

    def test_batch_rejects_mismatched_query_and_image_counts(self):
        vlm = make_vlm(FakeVLMController())
        with self.assertRaisesRegex(ValueError, "equal lengths"):
            vlm.batch_generate(["one", "two"], ["1.png"])


class TreeSummaryStreamingTests(unittest.TestCase):
    @staticmethod
    def add_node(tree, parent, node_type, content, **meta):
        node = TreeNode({"content": content, **meta})
        node.type = node_type
        tree.add_node(node)
        parent.add_child(node)
        return node

    def build_tree(self, save_path, image_path=None):
        tree = DocumentTree(
            {"file_name": "sample.pdf"},
            cfg=SimpleNamespace(save_path=str(save_path)),
        )
        section = self.add_node(tree, tree.root_node, NodeType.TITLE, "Overview")
        alpha = self.add_node(tree, section, NodeType.TEXT, "alpha facts")
        beta = self.add_node(tree, section, NodeType.TEXT, "beta facts")
        image = None
        if image_path is not None:
            image = self.add_node(
                tree,
                section,
                NodeType.IMAGE,
                "architecture diagram",
                img_path=str(image_path),
            )
        return tree, section, alpha, beta, image

    def test_tree_streams_llm_and_vlm_then_summarizes_parent_bottom_up(self):
        class TreeLLMController:
            def __init__(self):
                self.section_prompt = None

            def get_completion(self, prompt, json_response=False):
                if "alpha facts" in prompt:
                    return "alpha-summary"
                if "beta facts" in prompt:
                    return "beta-summary"
                if "**Section Text:**" in prompt and "Overview" in prompt:
                    self.section_prompt = prompt
                    return "section-summary"
                raise AssertionError(f"unexpected prompt: {prompt}")

        class TreeVLMController(FakeVLMController):
            def generate(self, prompt, images=None):
                self.images.append(images)
                return "image-summary"

        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "diagram.png"
            image_path.write_bytes(b"not decoded by the fake provider")
            tree, section, alpha, beta, image = self.build_tree(tmp, image_path)
            llm_controller = TreeLLMController()
            vlm_controller = TreeVLMController()

            result = generate_tree_node_summary(
                tree,
                make_llm(llm_controller),
                use_VLM=True,
                vlm=make_vlm(vlm_controller),
            )

        self.assertIs(result, tree)
        self.assertEqual(alpha.summary, "alpha-summary")
        self.assertEqual(beta.summary, "beta-summary")
        self.assertEqual(image.summary, "image-summary")
        self.assertEqual(section.summary, "section-summary")
        self.assertEqual(vlm_controller.images, [[str(image_path)]])
        self.assertIn("alpha-summary", llm_controller.section_prompt)
        self.assertIn("beta-summary", llm_controller.section_prompt)
        self.assertIn("image-summary", llm_controller.section_prompt)

    def test_tree_restart_reuses_committed_text_visual_and_section_summaries(self):
        class FirstLLMController:
            def get_completion(self, prompt, json_response=False):
                if "alpha facts" in prompt:
                    return "alpha-summary"
                if "beta facts" in prompt:
                    return "beta-summary"
                if "**Section Text:**" in prompt:
                    return "section-summary"
                raise AssertionError(f"unexpected prompt: {prompt}")

        class FirstVLMController(FakeVLMController):
            def generate(self, prompt, images=None):
                return "image-summary"

        class MustNotRunLLM:
            def get_completion(self, prompt, json_response=False):
                raise AssertionError("checkpointed LLM request was repeated")

        class MustNotRunVLM(FakeVLMController):
            def generate(self, prompt, images=None):
                raise AssertionError("checkpointed VLM request was repeated")

        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "diagram.png"
            image_path.write_bytes(b"stable image bytes")
            tree, section, alpha, beta, image = self.build_tree(tmp, image_path)
            generate_tree_node_summary(
                tree,
                make_llm(FirstLLMController()),
                use_VLM=True,
                vlm=make_vlm(FirstVLMController()),
            )
            for node in (section, alpha, beta, image):
                node.summary = ""

            generate_tree_node_summary(
                tree,
                make_llm(MustNotRunLLM()),
                use_VLM=True,
                vlm=make_vlm(MustNotRunVLM()),
            )

            self.assertTrue((Path(tmp) / "summary-checkpoints.sqlite3").is_file())
            self.assertEqual(alpha.summary, "alpha-summary")
            self.assertEqual(beta.summary, "beta-summary")
            self.assertEqual(image.summary, "image-summary")
            self.assertEqual(section.summary, "section-summary")

    def test_model_failure_aborts_stage_without_persisting_error_as_summary(self):
        class FailingTreeController:
            def get_completion(self, prompt, json_response=False):
                if "alpha facts" in prompt:
                    raise RuntimeError("secret backend failure")
                return "valid-summary"

        with tempfile.TemporaryDirectory() as tmp:
            tree, _, alpha, _, _ = self.build_tree(tmp)
            alpha.summary = "previous-valid-summary"

            with self.assertRaises(TreeSummaryGenerationError) as caught:
                generate_tree_node_summary(
                    tree,
                    make_llm(FailingTreeController(), max_retries=1),
                )

        self.assertEqual(caught.exception.node_id, alpha.index_id)
        self.assertEqual(caught.exception.provider, "LLM")
        self.assertEqual(alpha.summary, "previous-valid-summary")
        self.assertNotIn("Error:", alpha.summary)

    def test_missing_vlm_artifact_is_a_stage_failure_not_a_silent_blank(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "missing.png"
            tree, _, _, _, image = self.build_tree(tmp, missing)

            with self.assertRaises(TreeSummaryGenerationError) as caught:
                generate_tree_node_summary(
                    tree,
                    make_llm(FakeLLMController()),
                    use_VLM=True,
                    vlm=make_vlm(FakeVLMController()),
                )

        self.assertEqual(caught.exception.node_id, image.index_id)
        self.assertEqual(caught.exception.provider, "VLM")
        self.assertEqual(image.summary, "")


if __name__ == "__main__":
    unittest.main()
