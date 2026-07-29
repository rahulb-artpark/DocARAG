from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from time import monotonic
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

from Core.provider.embedding import (
    MultimodalEmbeddingProvider,
    MMRerankerProvider,
    _MultimodalSharedRuntime,
)


class GmeResourceOwnershipTests(unittest.TestCase):
    @staticmethod
    def _fake_loader(cuda):
        def load(runtime: _MultimodalSharedRuntime) -> None:
            runtime.torch = SimpleNamespace(cuda=cuda)
            runtime.torch_functional = object()
            runtime.model = object()

        return load

    @staticmethod
    def _wait_for_waiters(runtime: _MultimodalSharedRuntime, count: int) -> None:
        deadline = monotonic() + 2
        with runtime._condition:
            while runtime._waiting_calls != count:
                remaining = deadline - monotonic()
                if remaining <= 0:
                    raise AssertionError(
                        f"expected {count} GME waiters, found "
                        f"{runtime._waiting_calls}"
                    )
                runtime._condition.wait(timeout=remaining)

    @staticmethod
    def _wait_for_closing(runtime: _MultimodalSharedRuntime) -> None:
        deadline = monotonic() + 2
        with runtime._condition:
            while not runtime._closing:
                remaining = deadline - monotonic()
                if remaining <= 0:
                    raise AssertionError("GME runtime did not begin closing")
                runtime._condition.wait(timeout=remaining)

    def test_two_load_dtypes_never_share_one_set_of_weights(self):
        cuda = Mock()
        cuda.is_available.return_value = False

        with patch.object(
            _MultimodalSharedRuntime, "_load_model", new=self._fake_loader(cuda)
        ):
            half = MultimodalEmbeddingProvider(model_name="test/dtype-gme", device="cuda:0")
            full = MultimodalEmbeddingProvider(
                model_name="test/dtype-gme", device="cuda:0", torch_dtype="float32"
            )
            try:
                self.assertEqual("bfloat16", half.torch_dtype)
                self.assertIsNot(half._runtime, full._runtime)
            finally:
                half.close()
                full.close()

    def test_compatible_handles_share_weights_without_sharing_close_state(self):
        cuda = Mock()
        cuda.is_available.return_value = True
        key = ("test/shared-gme", "cuda:0", "bfloat16")

        with patch.object(
            _MultimodalSharedRuntime,
            "_load_model",
            new=self._fake_loader(cuda),
        ):
            first = MultimodalEmbeddingProvider(model_name=key[0], device=key[1])
            second = MultimodalEmbeddingProvider(model_name=key[0], device=key[1])

        self.assertIsNot(first, second)
        self.assertIs(first._runtime, second._runtime)
        self.assertEqual(2, first._runtime.owner_count)

        first.close()
        first.close()

        self.assertIsNotNone(second._runtime)
        self.assertEqual(1, second._runtime.owner_count)
        self.assertIn(key, MultimodalEmbeddingProvider._runtime_pool)
        cuda.empty_cache.assert_not_called()

        second.close()
        second.close()

        cuda.empty_cache.assert_called_once_with()
        self.assertNotIn(key, MultimodalEmbeddingProvider._runtime_pool)

    def test_different_model_or_device_never_silently_reuses_runtime(self):
        cuda = Mock()
        cuda.is_available.return_value = False
        loaded = []

        def load(runtime: _MultimodalSharedRuntime) -> None:
            loaded.append((runtime.model_name, runtime.device_strategy))
            self._fake_loader(cuda)(runtime)

        with patch.object(_MultimodalSharedRuntime, "_load_model", new=load):
            first = MultimodalEmbeddingProvider(model_name="test/model-a", device="cuda:0")
            second = MultimodalEmbeddingProvider(model_name="test/model-b", device="cuda:0")
            third = MultimodalEmbeddingProvider(model_name="test/model-a", device="cuda:1")

        try:
            self.assertEqual(
                [
                    ("test/model-a", "cuda:0"),
                    ("test/model-b", "cuda:0"),
                    ("test/model-a", "cuda:1"),
                ],
                loaded,
            )
            self.assertIsNot(first._runtime, second._runtime)
            self.assertIsNot(first._runtime, third._runtime)
        finally:
            first.close()
            second.close()
            third.close()

    def test_final_close_waits_for_an_inflight_runtime_call(self):
        cuda = Mock()
        cuda.is_available.return_value = False
        entered = Event()
        release_call = Event()
        close_wait_started = Event()
        key = ("test/inflight-gme", "cuda:0", "bfloat16")

        with patch.object(
            _MultimodalSharedRuntime,
            "_load_model",
            new=self._fake_loader(cuda),
        ):
            provider = MultimodalEmbeddingProvider(model_name=key[0], device=key[1])
        runtime = provider._runtime
        original_wait = runtime.wait_until_idle

        def observed_wait() -> None:
            close_wait_started.set()
            original_wait()

        runtime.wait_until_idle = observed_wait

        def hold_runtime() -> None:
            with provider._use_runtime():
                entered.set()
                self.assertTrue(release_call.wait(timeout=2))

        with ThreadPoolExecutor(max_workers=2) as executor:
            use_future = executor.submit(hold_runtime)
            self.assertTrue(entered.wait(timeout=2))
            close_future = executor.submit(provider.close)
            self.assertTrue(close_wait_started.wait(timeout=2))
            self.assertFalse(close_future.done())
            self.assertIn(key, MultimodalEmbeddingProvider._runtime_pool)
            release_call.set()
            use_future.result(timeout=2)
            close_future.result(timeout=2)

        self.assertNotIn(key, MultimodalEmbeddingProvider._runtime_pool)
        self.assertIsNone(runtime.model)

    def test_shared_runtime_serializes_calls_from_concurrent_handles(self):
        cuda = Mock()
        cuda.is_available.return_value = False
        first_entered = Event()
        release_first = Event()
        second_entered = Event()
        key = ("test/serialized-gme", "cuda:0", "bfloat16")

        with patch.object(
            _MultimodalSharedRuntime,
            "_load_model",
            new=self._fake_loader(cuda),
        ):
            first = MultimodalEmbeddingProvider(model_name=key[0], device=key[1])
            second = MultimodalEmbeddingProvider(model_name=key[0], device=key[1])
        runtime = first._runtime

        def hold_first_slot() -> None:
            with first._use_runtime():
                first_entered.set()
                self.assertTrue(release_first.wait(timeout=2))

        def enter_second_slot() -> None:
            with second._use_runtime():
                second_entered.set()

        with ThreadPoolExecutor(max_workers=2) as executor:
            first_future = executor.submit(hold_first_slot)
            self.assertTrue(first_entered.wait(timeout=2))
            second_future = executor.submit(enter_second_slot)
            try:
                self._wait_for_waiters(runtime, 1)
                self.assertEqual(1, runtime._active_calls)
                self.assertFalse(second_entered.is_set())
            finally:
                release_first.set()
            first_future.result(timeout=2)
            second_future.result(timeout=2)

        self.assertTrue(second_entered.is_set())
        self.assertEqual(0, runtime._active_calls)
        first.close()
        second.close()

    def test_final_close_cancels_a_waiter_without_deadlock(self):
        cuda = Mock()
        cuda.is_available.return_value = False
        active_entered = Event()
        release_active = Event()
        key = ("test/cancelled-waiter-gme", "cuda:0", "bfloat16")

        with patch.object(
            _MultimodalSharedRuntime,
            "_load_model",
            new=self._fake_loader(cuda),
        ):
            provider = MultimodalEmbeddingProvider(model_name=key[0], device=key[1])
        runtime = provider._runtime

        def hold_slot() -> None:
            with provider._use_runtime():
                active_entered.set()
                self.assertTrue(release_active.wait(timeout=2))

        def wait_for_slot() -> None:
            with provider._use_runtime():
                raise AssertionError("a closed provider waiter must not be admitted")

        with ThreadPoolExecutor(max_workers=3) as executor:
            active_future = executor.submit(hold_slot)
            self.assertTrue(active_entered.wait(timeout=2))
            waiter_future = executor.submit(wait_for_slot)
            self._wait_for_waiters(runtime, 1)
            close_future = executor.submit(provider.close)
            try:
                with self.assertRaisesRegex(
                    RuntimeError,
                    "provider closed while waiting",
                ):
                    waiter_future.result(timeout=2)
                self._wait_for_closing(runtime)
                self.assertFalse(close_future.done())
            finally:
                release_active.set()
            active_future.result(timeout=2)
            close_future.result(timeout=2)

        self.assertEqual(0, runtime._active_calls)
        self.assertEqual(0, runtime._waiting_calls)
        self.assertNotIn(key, MultimodalEmbeddingProvider._runtime_pool)

    def test_failed_runtime_load_is_cleaned_and_never_published(self):
        failed_cuda = Mock()
        failed_cuda.is_available.return_value = True
        key = ("test/failing-gme", "cuda:0", "bfloat16")

        def fail_after_partial_allocation(runtime: _MultimodalSharedRuntime) -> None:
            runtime.torch = SimpleNamespace(cuda=failed_cuda)
            runtime.model = object()
            raise RuntimeError("model load failed")

        with (
            patch.object(
                _MultimodalSharedRuntime,
                "_load_model",
                new=fail_after_partial_allocation,
            ),
            self.assertRaisesRegex(RuntimeError, "model load failed"),
        ):
            MultimodalEmbeddingProvider(model_name=key[0], device=key[1])

        failed_cuda.empty_cache.assert_called_once_with()
        self.assertNotIn(key, MultimodalEmbeddingProvider._runtime_pool)

        healthy_cuda = Mock()
        healthy_cuda.is_available.return_value = False
        with patch.object(
            _MultimodalSharedRuntime,
            "_load_model",
            new=self._fake_loader(healthy_cuda),
        ):
            recovered = MultimodalEmbeddingProvider(model_name=key[0], device=key[1])
        recovered.close()
        self.assertNotIn(key, MultimodalEmbeddingProvider._runtime_pool)

    def test_mmr_injected_provider_has_explicit_borrowed_or_owned_lifecycle(self):
        borrowed = Mock()
        owned = Mock()
        with tempfile.TemporaryDirectory() as directory:
            borrowed_reranker = MMRerankerProvider(
                directory,
                mm_provider=borrowed,
            )
            owned_reranker = MMRerankerProvider(
                directory,
                mm_provider=owned,
                owns_mm_provider=True,
            )

        borrowed_reranker.close()
        borrowed_reranker.close()
        owned_reranker.close()
        owned_reranker.close()

        borrowed.close.assert_not_called()
        owned.close.assert_called_once_with()

    def test_closing_one_mmr_owner_keeps_the_other_runtime_live(self):
        cuda = Mock()
        cuda.is_available.return_value = True
        key = ("test/mmr-shared-gme", "cuda:0", "bfloat16")
        with tempfile.TemporaryDirectory() as directory, patch.object(
            _MultimodalSharedRuntime,
            "_load_model",
            new=self._fake_loader(cuda),
        ):
            first = MMRerankerProvider(
                directory,
                model_name=key[0],
                device=key[1],
            )
            second = MMRerankerProvider(
                directory,
                model_name=key[0],
                device=key[1],
            )

        first.close()
        first.close()

        with second.mm_provider._use_runtime() as live_runtime:
            self.assertIsNotNone(live_runtime.model)
            self.assertEqual(1, live_runtime.owner_count)
        cuda.empty_cache.assert_not_called()

        second.close()
        second.close()
        cuda.empty_cache.assert_called_once_with()
        self.assertNotIn(key, MultimodalEmbeddingProvider._runtime_pool)

    def test_mmr_constructor_failure_releases_only_transferred_ownership(self):
        owned = Mock()
        borrowed = Mock()
        with patch.object(
            MMRerankerProvider,
            "_load_embeddings",
            side_effect=RuntimeError("metadata is corrupt"),
        ):
            with self.assertRaisesRegex(RuntimeError, "metadata is corrupt"):
                MMRerankerProvider(
                    str(Path("unused")),
                    mm_provider=owned,
                    owns_mm_provider=True,
                )
            with self.assertRaisesRegex(RuntimeError, "metadata is corrupt"):
                MMRerankerProvider(
                    str(Path("unused")),
                    mm_provider=borrowed,
                )

        owned.close.assert_called_once_with()
        borrowed.close.assert_not_called()


if __name__ == "__main__":
    unittest.main()
