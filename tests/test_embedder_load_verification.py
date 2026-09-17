"""A multimodal embedder that loaded without learning anything must not start.

Qwen3-VL-Embedding through sentence-transformers resolves to a bare
Qwen3VLModel, warns that every weight was "newly initialized", and then returns
correctly shaped vectors carrying no signal. Indexing succeeds, Qdrant fills,
and retrieval is noise.
"""

import unittest

import torch

from retrieval_engine.providers.embeddings import _MIN_SEMANTIC_MARGIN, _MultimodalSharedRuntime


class _Randomized:
    """Stands in for a checkpoint whose weights never loaded."""

    def process(self, items):
        generator = torch.Generator().manual_seed(0)
        vectors = torch.randn(len(items), 16, generator=generator)
        return vectors / vectors.norm(dim=-1, keepdim=True)


class _Working:
    """Stands in for real weights: the answer scores above the distractor."""

    def process(self, items):
        rows = [
            [1.0, 0.0, 0.0],   # query
            [0.95, 0.31, 0.0],  # its answer
            [0.0, 0.0, 1.0],   # unrelated
        ][: len(items)]
        vectors = torch.tensor(rows)
        return vectors / vectors.norm(dim=-1, keepdim=True)


def _runtime(model):
    runtime = _MultimodalSharedRuntime.__new__(_MultimodalSharedRuntime)
    runtime.model_name = "fixture/model"
    runtime.model = model
    return runtime


class EmbedderLoadVerification(unittest.TestCase):
    def test_randomized_weights_are_rejected(self):
        with self.assertRaises(RuntimeError) as caught:
            _runtime(_Randomized())._verify_loaded()
        self.assertIn("does not discriminate", str(caught.exception))

    def test_working_weights_are_accepted(self):
        _runtime(_Working())._verify_loaded()

    def test_margin_threshold_is_between_the_two_regimes(self):
        """0.44 is what real weights score; 0.10 is what randomized ones score."""

        self.assertGreater(_MIN_SEMANTIC_MARGIN, 0.10)
        self.assertLess(_MIN_SEMANTIC_MARGIN, 0.44)


if __name__ == "__main__":
    unittest.main()
