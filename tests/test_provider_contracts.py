from __future__ import annotations

import sys
import tempfile
import types
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import numpy as np
from pydantic import BaseModel

from retrieval_engine.configuration.embeddings import EmbeddingConfig
from retrieval_engine.configuration.knowledge_graph import GraphConfig
from retrieval_engine.configuration.language_model import LLMConfig
from retrieval_engine.configuration.reranking import RerankerConfig
from retrieval_engine.configuration.vision_model import VLMConfig
from retrieval_engine.providers.embeddings import TextEmbeddingProvider
from retrieval_engine.providers.language_model import OllamaController, OpenAIController
from retrieval_engine.providers.reranking import TextRerankerProvider
from retrieval_engine.providers.vision_model import (
    GPTVLMController,
    OllamaVLMController,
    QwenVLController,
)


class Payload(BaseModel):
    value: str


def _chat_response(content: str):
    return {
        "message": {"content": content},
        "prompt_eval_count": 3,
        "eval_count": 2,
    }


class ProviderConfigValidationTests(unittest.TestCase):
    def test_generation_configs_reject_values_backends_cannot_honor(self):
        with self.assertRaisesRegex(ValueError, "temperature"):
            LLMConfig(temperature=float("nan"))
        with self.assertRaisesRegex(ValueError, "max_tokens"):
            VLMConfig(max_tokens=0)
        with self.assertRaisesRegex(ValueError, "not supported"):
            LLMConfig(backend="ollama", frequency_penalty=0.5)
        with self.assertRaisesRegex(ValueError, "keep_alive"):
            VLMConfig(backend="ollama", keep_alive="forever")
        with self.assertRaisesRegex(ValueError, "keep_alive"):
            LLMConfig(backend="ollama", keep_alive=-0.5)
        with self.assertRaisesRegex(ValueError, "only configurable for Ollama"):
            LLMConfig(backend="openai", unload_on_close=True)
        with self.assertRaisesRegex(ValueError, "not supported"):
            VLMConfig(
                backend="qwen",
                api_base="http://unused.test",
            )

    def test_embedding_and_reranker_configs_validate_provider_boundaries(self):
        with self.assertRaisesRegex(ValueError, "local embedding"):
            EmbeddingConfig(backend="local", api_key="unused")
        with self.assertRaisesRegex(ValueError, "api_base is required"):
            RerankerConfig(backend="vllm")
        with self.assertRaisesRegex(ValueError, "request_timeout_seconds"):
            RerankerConfig(
                backend="vllm",
                api_base="http://reranker.test/v1",
                request_timeout_seconds=0,
            )
        with self.assertRaisesRegex(ValueError, "device is only supported"):
            RerankerConfig(
                backend="vllm",
                api_base="http://reranker.test/v1",
                device="cuda:0",
            )
        with self.assertRaisesRegex(ValueError, "device is only supported"):
            EmbeddingConfig(backend="ollama", device="cuda:0")

        cpu_config = RerankerConfig(backend="local", device="cpu")
        self.assertFalse(cpu_config.use_flash_attention)
        self.assertIsNone(cpu_config.flash_attention_fallback)
        cuda_config = RerankerConfig(backend="local", device="cuda:0")
        self.assertTrue(cuda_config.use_flash_attention)
        self.assertTrue(cuda_config.flash_attention_fallback)


class LLMProviderContractTests(unittest.TestCase):
    def test_ollama_client_receives_configured_base_and_auth(self):
        with patch("ollama.Client") as client_factory:
            controller = OllamaController(
                LLMConfig(
                    backend="ollama",
                    api_base="http://ollama.test",
                    api_key="configured-key",
                )
            )

        client_factory.assert_called_once_with(
            host="http://ollama.test",
            headers={"Authorization": "Bearer configured-key"},
            timeout=300.0,
        )
        controller.close()
        client_factory.return_value.close.assert_called_once_with()

    def test_ollama_generation_forwards_limits_and_retention_to_both_paths(self):
        client = Mock()
        client.chat.side_effect = [
            _chat_response("plain"),
            _chat_response('{"value":"structured"}'),
        ]
        config = LLMConfig(
            backend="ollama",
            model_name="text-model",
            api_base="http://ollama.test",
            api_key=None,
            temperature=0.25,
            max_tokens=77,
            keep_alive="7m",
        )
        controller = OllamaController(config, client=client)

        self.assertEqual(controller.get_completion("hello"), "plain")
        result = controller.get_json_completion("hello", Payload, think_mode=True)

        self.assertEqual(result.value, "structured")
        plain_args = client.chat.call_args_list[0].kwargs
        json_args = client.chat.call_args_list[1].kwargs
        for outgoing in (plain_args, json_args):
            self.assertEqual(outgoing["model"], "text-model")
            self.assertEqual(outgoing["keep_alive"], "7m")
            self.assertEqual(
                outgoing["options"], {"temperature": 0.25, "num_predict": 77}
            )
        self.assertFalse(plain_args["think"])
        self.assertTrue(json_args["think"])

    def test_ollama_close_does_not_evict_shared_model_unless_opted_in(self):
        shared_client = Mock()
        shared = OllamaController(
            LLMConfig(backend="ollama", api_key=None), client=shared_client
        )

        shared.close()
        shared.close()

        shared_client.generate.assert_not_called()
        shared_client.close.assert_not_called()

        owned_client = Mock()
        owned = OllamaController(
            LLMConfig(
                backend="ollama",
                model_name="owned-model",
                api_key=None,
                unload_on_close=True,
            ),
            client=owned_client,
            owns_client=True,
        )

        owned.close()
        owned.close()

        owned_client.generate.assert_called_once_with(
            model="owned-model", prompt="", keep_alive=0
        )
        owned_client.close.assert_called_once_with()

    def test_openai_structured_generation_forwards_all_generation_settings(self):
        client = Mock()
        parsed = Payload(value="ok")
        client.beta.chat.completions.parse.return_value = SimpleNamespace(
            usage=None,
            choices=[SimpleNamespace(message=SimpleNamespace(parsed=parsed))],
        )
        config = LLMConfig(
            backend="openai",
            model_name="compatible-model",
            api_key="configured-key",
            api_base="http://openai-compatible.test/v1",
            temperature=0.4,
            max_tokens=200,
            frequency_penalty=0.3,
            presence_penalty=-0.2,
        )
        controller = OpenAIController(config, client=client)

        self.assertIs(controller.get_json_completion("question", Payload), parsed)

        outgoing = client.beta.chat.completions.parse.call_args.kwargs
        self.assertEqual(outgoing["temperature"], 0.4)
        self.assertGreater(outgoing["max_tokens"], 0)
        self.assertLessEqual(outgoing["max_tokens"], 200)
        self.assertEqual(outgoing["frequency_penalty"], 0.3)
        self.assertEqual(outgoing["presence_penalty"], -0.2)
        self.assertEqual(outgoing["timeout"], 300.0)


class VLMProviderContractTests(unittest.TestCase):
    def test_gpt_client_receives_configured_base_and_key(self):
        with patch("openai.OpenAI") as client_factory:
            controller = GPTVLMController(
                VLMConfig(
                    backend="gpt",
                    model_name="vision-model",
                    api_key="configured-key",
                    api_base="http://vision.test/v1",
                )
            )

        client_factory.assert_called_once_with(
            api_key="configured-key",
            base_url="http://vision.test/v1",
            timeout=300.0,
            max_retries=0,
        )
        controller.close()
        client_factory.return_value.close.assert_called_once_with()

    def test_gpt_vlm_forwards_temperature_and_limit_in_plain_and_json_calls(self):
        client = Mock()
        client.chat.completions.create.side_effect = [
            SimpleNamespace(
                usage=None,
                choices=[SimpleNamespace(message=SimpleNamespace(content="plain"))],
            ),
            SimpleNamespace(
                usage=None,
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content='{"value":"json"}')
                    )
                ],
            ),
        ]
        controller = GPTVLMController(
            VLMConfig(
                backend="gpt",
                model_name="vision-model",
                api_key="key",
                api_base="http://vision.test/v1",
                temperature=0.35,
                max_tokens=321,
            ),
            client=client,
        )

        self.assertEqual(controller.generate("describe"), "plain")
        self.assertEqual(
            controller.generate_json("describe", schema=Payload).value, "json"
        )

        for outgoing_call in client.chat.completions.create.call_args_list:
            self.assertEqual(outgoing_call.kwargs["temperature"], 0.35)
            self.assertEqual(outgoing_call.kwargs["max_tokens"], 321)
            self.assertEqual(outgoing_call.kwargs["timeout"], 300.0)

    def test_ollama_vlm_forwards_limits_and_has_opt_in_eviction(self):
        client = Mock()
        client.chat.side_effect = [
            _chat_response("plain"),
            _chat_response('{"value":"json"}'),
        ]
        config = VLMConfig(
            backend="ollama",
            model_name="vision-model",
            api_key=None,
            api_base="http://ollama.test",
            temperature=0.15,
            max_tokens=222,
            keep_alive="3m",
            unload_on_close=True,
        )
        controller = OllamaVLMController(
            config, client=client, owns_client=True
        )

        self.assertEqual(controller.generate("describe"), "plain")
        self.assertEqual(
            controller.generate_json("describe", schema=Payload).value, "json"
        )
        for outgoing_call in client.chat.call_args_list:
            self.assertEqual(outgoing_call.kwargs["keep_alive"], "3m")
            self.assertEqual(
                outgoing_call.kwargs["options"],
                {"temperature": 0.15, "num_predict": 222},
            )

        controller.close()
        controller.close()
        client.generate.assert_called_once_with(
            model="vision-model", prompt="", keep_alive=0
        )
        client.close.assert_called_once_with()

    def test_ollama_vlm_shared_client_is_not_closed_or_evicted_by_default(self):
        client = Mock()
        controller = OllamaVLMController(
            VLMConfig(backend="ollama", api_key=None),
            client=client,
        )

        controller.close()
        controller.close()

        client.generate.assert_not_called()
        client.close.assert_not_called()

    def test_qwen_generation_uses_configured_temperature_and_limit(self):
        class Inputs(dict):
            input_ids = [[1, 2]]
            moved_to = None

            def to(self, device):
                self.moved_to = device
                return self

        processor = Mock()
        processor.apply_chat_template.return_value = "rendered"
        inputs = Inputs()
        processor.return_value = inputs
        processor.batch_decode.return_value = ["answer"]
        model = Mock()
        model.device = "cpu"
        model.generate.return_value = [[1, 2, 3]]
        controller = object.__new__(QwenVLController)
        controller.processor = processor
        controller.model = model
        controller.temperature = 0.45
        controller.max_tokens = 88

        vision_utils = types.ModuleType("qwen_vl_utils")
        vision_utils.process_vision_info = lambda _messages: ([], [])
        with patch.dict(sys.modules, {"qwen_vl_utils": vision_utils}):
            self.assertEqual(controller.generate("describe"), "answer")

        outgoing = model.generate.call_args.kwargs
        self.assertEqual(inputs.moved_to, "cpu")
        self.assertEqual(outgoing["max_new_tokens"], 88)
        self.assertTrue(outgoing["do_sample"])
        self.assertEqual(outgoing["temperature"], 0.45)


class EmbeddingProviderContractTests(unittest.TestCase):
    def test_openai_client_receives_configured_key_and_base(self):
        with patch("openai.OpenAI") as client_factory:
            provider = TextEmbeddingProvider(
                model_name="embedding-model",
                backend="openai",
                api_key="configured-key",
                api_base="http://embedding.test/v1",
            )

        client_factory.assert_called_once_with(
            api_key="configured-key", base_url="http://embedding.test/v1"
        )
        provider.close()
        client_factory.return_value.close.assert_called_once_with()

    def test_ollama_embeddings_use_configured_owned_client_and_batch_api(self):
        client = Mock()
        client.embed.return_value = {"embeddings": [[3.0, 4.0], [0.0, 2.0]]}
        with patch("ollama.Client", return_value=client) as client_factory:
            provider = TextEmbeddingProvider(
                model_name="embedding-model",
                backend="ollama",
                api_key="configured-key",
                api_base="http://ollama-embedding.test",
            )
        result = provider.embed_texts(["one", "two"])

        client_factory.assert_called_once_with(
            host="http://ollama-embedding.test",
            headers={"Authorization": "Bearer configured-key"},
        )
        client.embed.assert_called_once_with(
            model="embedding-model", input=["one", "two"]
        )
        np.testing.assert_allclose(result, [[0.6, 0.8], [0.0, 1.0]])
        provider.close()
        client.close.assert_called_once_with()

    def test_injected_embedding_client_is_not_closed_by_default(self):
        client = Mock()
        provider = TextEmbeddingProvider(
            model_name="embedding-model", backend="ollama", client=client
        )

        provider.close()
        provider.close()

        client.close.assert_not_called()


class RerankerProviderContractTests(unittest.TestCase):
    def test_remote_reranker_uses_timeout_and_restores_document_order(self):
        session = Mock()
        response = Mock()
        response.json.return_value = {
            "results": [
                {"index": 1, "relevance_score": 0.9},
                {"index": 0, "relevance_score": 0.2},
            ]
        }
        session.post.return_value = response
        provider = TextRerankerProvider(
            model_name="reranker",
            backend="vllm",
            api_base="http://reranker.test/v1",
            request_timeout_seconds=4.5,
            session=session,
        )

        scores = provider.rerank("query", ["first", "second"], batch_size=2)

        self.assertEqual(scores, [0.2, 0.9])
        self.assertEqual(session.post.call_args.kwargs["timeout"], 4.5)
        self.assertEqual(
            session.post.call_args.args[0], "http://reranker.test/v1/rerank"
        )
        provider.close()
        session.close.assert_not_called()

    def test_owned_remote_session_is_closed_once(self):
        session = Mock()
        provider = TextRerankerProvider(
            backend="vllm",
            api_base="http://reranker.test/v1",
            session=session,
            owns_session=True,
        )

        provider.close()
        provider.close()

        session.close.assert_called_once_with()

    def test_flash_attention_failure_or_fallback_happens_before_model_load(self):
        # Keep Torch outside the ``patch.dict(sys.modules, ...)`` snapshots below.
        # Otherwise the first context removes Torch after importing it, and the
        # extension cannot be initialized a second time in the same process.
        import torch  # noqa: F401

        tokenizer = Mock()
        tokenizer.convert_tokens_to_ids.side_effect = [1, 2]
        tokenizer.encode.return_value = [3]
        tokenizer_factory = Mock()
        tokenizer_factory.from_pretrained.return_value = tokenizer
        model = Mock()
        model.to.return_value = model
        model.eval.return_value = model
        model_factory = Mock()
        model_factory.from_pretrained.return_value = model
        modelscope = types.ModuleType("modelscope")
        modelscope.AutoTokenizer = tokenizer_factory
        modelscope.AutoModelForCausalLM = model_factory

        with (
            patch.dict(sys.modules, {"modelscope": modelscope}),
            patch(
                "retrieval_engine.providers.reranking.importlib.import_module",
                side_effect=ImportError("not installed"),
            ),
        ):
            fallback = TextRerankerProvider(
                device="cuda:0",
                use_flash_attention=True,
                flash_attention_fallback=True,
            )

        model_args = model_factory.from_pretrained.call_args.kwargs
        self.assertNotIn("attn_implementation", model_args)
        fallback.device = "cpu"
        fallback.close()

        tokenizer_factory.reset_mock()
        model_factory.reset_mock()
        with (
            patch.dict(sys.modules, {"modelscope": modelscope}),
            patch(
                "retrieval_engine.providers.reranking.importlib.import_module",
                side_effect=ImportError("not installed"),
            ),
            self.assertRaisesRegex(RuntimeError, "requested but is unavailable"),
        ):
            TextRerankerProvider(
                device="cuda:0",
                use_flash_attention=True,
                flash_attention_fallback=False,
            )

        tokenizer_factory.from_pretrained.assert_not_called()
        model_factory.from_pretrained.assert_not_called()

    def test_graph_config_settings_reach_embedding_and_reranker_providers(self):
        from retrieval_engine.indexing.graph.refiner import KGRefiner

        graph_config = GraphConfig(
            embedding_config=EmbeddingConfig(
                backend="openai",
                model_name="embedding-model",
                api_key="configured-key",
                api_base="http://embedding.test/v1",
            ),
            reranker_config=RerankerConfig(
                backend="vllm",
                model_name="reranker-model",
                api_base="http://reranker.test/v1",
                request_timeout_seconds=6.25,
            ),
        )
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("retrieval_engine.indexing.graph.refiner.TextEmbeddingProvider") as embedding,
            patch("retrieval_engine.indexing.graph.refiner.TextRerankerProvider") as reranker,
            patch("retrieval_engine.indexing.graph.refiner.VectorStore"),
        ):
            refiner = KGRefiner(
                llm=object(),
                graph_config=graph_config,
                graph_index=object(),
                save_path=directory,
                vdb_config=SimpleNamespace(
                    qdrant_url="http://qdrant.test:6333",
                    api_key_env="QDRANT_API_KEY",
                    timeout_seconds=2.0,
                ),
                scope_filter={
                    "tenant_id": "tenant-a",
                    "corpus_id": "corpus-a",
                    "document_version_id": "revision-a",
                    "source_sha256": "a" * 64,
                    "index_kind": "entity",
                },
            )

        self.assertEqual(embedding.call_args.kwargs["api_key"], "configured-key")
        self.assertEqual(
            reranker.call_args.kwargs["request_timeout_seconds"], 6.25
        )
        self.assertIsNone(reranker.call_args.kwargs["use_flash_attention"])
        self.assertIsNone(reranker.call_args.kwargs["flash_attention_fallback"])
        refiner.close()


if __name__ == "__main__":
    unittest.main()
