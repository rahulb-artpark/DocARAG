from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import textwrap
from types import SimpleNamespace
import unittest
from unittest.mock import Mock


class RemoteOnlyImportBoundaryTests(unittest.TestCase):
    def test_remote_providers_do_not_import_local_ml_stack(self):
        script = textwrap.dedent(
            r"""
            import builtins
            import json
            import sys
            from types import SimpleNamespace

            forbidden = {"torch", "modelscope", "transformers", "flash_attn"}
            forbidden_attempts = []
            original_import = builtins.__import__

            class ForbiddenModuleFinder:
                def find_spec(self, fullname, path=None, target=None):
                    del path, target
                    if fullname.split(".", 1)[0] in forbidden:
                        forbidden_attempts.append(fullname)
                        raise AssertionError(f"remote-only path imported {fullname}")
                    return None

            def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
                if name.split(".", 1)[0] in forbidden:
                    forbidden_attempts.append(name)
                    raise AssertionError(f"remote-only path imported {name}")
                return original_import(name, globals, locals, fromlist, level)

            builtins.__import__ = guarded_import
            sys.meta_path.insert(0, ForbiddenModuleFinder())

            from Core.configs.embedding_config import EmbeddingConfig
            import Core.configs.dataset_config
            import Core.configs.gbc_limits
            import Core.configs.graph_config
            from Core.configs.llm_config import LLMConfig
            import Core.configs.mineru_config
            import Core.configs.rag_config
            from Core.configs.rerank_config import RerankerConfig
            import Core.configs.system_config
            import Core.configs.tree_config
            import Core.configs.vdb_config
            from Core.configs.vlm_config import VLMConfig
            import Core.utils.resource_loader
            import Core.provider.TokenTracker
            from Core.provider.embedding import TextEmbeddingProvider
            import Core.provider.extract_pdf_info
            from Core.provider.llm import OllamaController, OpenAIController
            import Core.provider.mineru_reconciliation
            from Core.provider.rerank import TextRerankerProvider
            import Core.provider.vdb
            from Core.provider.vlm import GPTVLMController, OllamaVLMController

            class OpenAIClient:
                def __init__(self, answer):
                    self.answer = answer
                    self.close_count = 0
                    self.embeddings = SimpleNamespace(create=self.create_embeddings)
                    self.chat = SimpleNamespace(
                        completions=SimpleNamespace(create=self.create_completion)
                    )

                def create_embeddings(self, *, input, **_kwargs):
                    vectors = [[3.0, 4.0], [0.0, 5.0]]
                    return SimpleNamespace(
                        data=[
                            SimpleNamespace(embedding=vector)
                            for vector in vectors[: len(input)]
                        ]
                    )

                def create_completion(self, **_kwargs):
                    return SimpleNamespace(
                        usage=None,
                        choices=[
                            SimpleNamespace(
                                message=SimpleNamespace(content=self.answer)
                            )
                        ],
                    )

                def close(self):
                    self.close_count += 1

            class OllamaClient:
                def __init__(self, answer):
                    self.answer = answer
                    self.close_count = 0

                def embed(self, *, model, input):
                    del model
                    vectors = [[8.0, 6.0], [0.0, 2.0]]
                    return {"embeddings": vectors[: len(input)]}

                def chat(self, **_kwargs):
                    return {
                        "message": {"content": self.answer},
                        "prompt_eval_count": 2,
                        "eval_count": 1,
                    }

                def close(self):
                    self.close_count += 1

            class RerankResponse:
                def raise_for_status(self):
                    return None

                def json(self):
                    return {
                        "results": [
                            {"index": 1, "relevance_score": 0.9},
                            {"index": 0, "relevance_score": 0.2},
                        ]
                    }

            class RerankSession:
                def __init__(self):
                    self.close_count = 0

                def post(self, *_args, **_kwargs):
                    return RerankResponse()

                def close(self):
                    self.close_count += 1

            openai_embedding_client = OpenAIClient("unused")
            ollama_embedding_client = OllamaClient("unused")
            openai_llm_client = OpenAIClient("openai-llm")
            ollama_llm_client = OllamaClient("ollama-llm")
            gpt_vlm_client = OpenAIClient("gpt-vlm")
            ollama_vlm_client = OllamaClient("ollama-vlm")
            rerank_session = RerankSession()

            openai_embedding = TextEmbeddingProvider(
                model_name="remote-embedding",
                backend="openai",
                client=openai_embedding_client,
                owns_client=True,
            )
            ollama_embedding = TextEmbeddingProvider(
                model_name="remote-embedding",
                backend="ollama",
                client=ollama_embedding_client,
                owns_client=True,
            )
            reranker = TextRerankerProvider(
                model_name="remote-reranker",
                backend="vllm",
                api_base="http://reranker.test/v1",
                session=rerank_session,
                owns_session=True,
            )
            openai_llm = OpenAIController(
                LLMConfig(backend="openai", api_key="test-key"),
                client=openai_llm_client,
                owns_client=True,
            )
            ollama_llm = OllamaController(
                LLMConfig(backend="ollama", api_key=None),
                client=ollama_llm_client,
                owns_client=True,
            )
            gpt_vlm = GPTVLMController(
                VLMConfig(backend="gpt", api_key="test-key"),
                client=gpt_vlm_client,
                owns_client=True,
            )
            ollama_vlm = OllamaVLMController(
                VLMConfig(backend="ollama", api_key=None),
                client=ollama_vlm_client,
                owns_client=True,
            )

            result = {
                "openai_embedding": openai_embedding.embed_texts(["a", "b"]).tolist(),
                "ollama_embedding": ollama_embedding.embed_texts(["a", "b"]).tolist(),
                "reranker": reranker.rerank("query", ["a", "b"], batch_size=2),
                "openai_llm": openai_llm.get_completion("question"),
                "ollama_llm": ollama_llm.get_completion("question"),
                "gpt_vlm": gpt_vlm.generate("describe"),
                "ollama_vlm": ollama_vlm.generate("describe"),
            }

            providers = [
                openai_embedding,
                ollama_embedding,
                reranker,
                openai_llm,
                ollama_llm,
                gpt_vlm,
                ollama_vlm,
            ]
            for provider in providers:
                provider.close()
                provider.close()

            result["close_counts"] = [
                openai_embedding_client.close_count,
                ollama_embedding_client.close_count,
                rerank_session.close_count,
                openai_llm_client.close_count,
                ollama_llm_client.close_count,
                gpt_vlm_client.close_count,
                ollama_vlm_client.close_count,
            ]
            result["forbidden_attempts"] = forbidden_attempts
            result["forbidden_loaded"] = sorted(
                name
                for name in sys.modules
                if name.split(".", 1)[0] in forbidden
            )
            print(json.dumps(result))
            """
        )
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
        )
        result = json.loads(completed.stdout)

        self.assertEqual(
            result["openai_embedding"], [[0.6000000238418579, 0.800000011920929], [0.0, 1.0]]
        )
        self.assertEqual(
            result["ollama_embedding"], [[0.800000011920929, 0.6000000238418579], [0.0, 1.0]]
        )
        self.assertEqual(result["reranker"], [0.2, 0.9])
        self.assertEqual(result["openai_llm"], "openai-llm")
        self.assertEqual(result["ollama_llm"], "ollama-llm")
        self.assertEqual(result["gpt_vlm"], "gpt-vlm")
        self.assertEqual(result["ollama_vlm"], "ollama-vlm")
        self.assertEqual(result["close_counts"], [1, 1, 1, 1, 1, 1, 1])
        self.assertEqual(result["forbidden_attempts"], [])
        self.assertEqual(result["forbidden_loaded"], [])

    def test_text_embedding_cuda_cleanup_uses_owned_runtime_once(self):
        from Core.provider.embedding import TextEmbeddingProvider

        cuda = Mock()
        cuda.is_available.return_value = True
        provider = object.__new__(TextEmbeddingProvider)
        provider._closed = False
        provider.backend = "local"
        provider.device = "cuda:0"
        provider.model_name = "local-model"
        provider.model = object()
        provider.tokenizer = object()
        provider._torch = SimpleNamespace(cuda=cuda)
        provider._torch_functional = object()

        provider.close()
        provider.close()

        cuda.empty_cache.assert_called_once_with()
        self.assertIsNone(provider._torch)
        self.assertIsNone(provider._torch_functional)

    def test_gme_cuda_cleanup_and_singleton_release_are_idempotent(self):
        from Core.provider.embedding import MultimodalEmbeddingProvider

        cuda = Mock()
        cuda.is_available.return_value = True
        provider = object.__new__(MultimodalEmbeddingProvider)
        provider.model_name = "gme-model"
        provider.model = object()
        provider._initialized = True
        provider._torch = SimpleNamespace(cuda=cuda)
        provider._torch_functional = object()
        previous_instance = MultimodalEmbeddingProvider._instance
        MultimodalEmbeddingProvider._instance = provider
        try:
            provider.close()
            provider.close()

            cuda.empty_cache.assert_called_once_with()
            self.assertIsNone(provider._torch)
            self.assertIsNone(provider._torch_functional)
            self.assertIsNone(MultimodalEmbeddingProvider._instance)
        finally:
            MultimodalEmbeddingProvider._instance = previous_instance


if __name__ == "__main__":
    unittest.main()
