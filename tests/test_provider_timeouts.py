from __future__ import annotations

import math
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from pydantic import BaseModel

from Core.configs.llm_config import LLMConfig
from Core.configs.vlm_config import VLMConfig
from Core.provider.llm import (
    CompletionAttemptsExhausted,
    LLM,
    OllamaController,
    OpenAIController,
)
from Core.provider.vlm import GPTVLMController, OllamaVLMController


class _Payload(BaseModel):
    value: str


def _openai_response(content: str):
    return SimpleNamespace(
        usage=None,
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
    )


class ProviderTimeoutConfigTests(unittest.TestCase):
    def test_retry_attempts_have_an_inclusive_hard_ceiling(self):
        self.assertEqual(10, LLMConfig(max_retries=10).max_retries)
        self.assertEqual(
            10,
            VLMConfig(backend="gpt", max_retries=10).max_retries,
        )
        for config_factory in (
            lambda: LLMConfig(max_retries=11),
            lambda: VLMConfig(backend="ollama", max_retries=11),
        ):
            with self.subTest(config_factory=config_factory):
                with self.assertRaisesRegex(ValueError, "between 1 and 10"):
                    config_factory()

    def test_remote_defaults_are_finite_and_local_qwen_uses_none_sentinel(self):
        self.assertEqual(300.0, LLMConfig().request_timeout_seconds)
        self.assertEqual(
            300.0,
            VLMConfig(backend="gpt").request_timeout_seconds,
        )
        self.assertEqual(
            300.0,
            VLMConfig(backend="ollama").request_timeout_seconds,
        )
        self.assertIsNone(VLMConfig(backend="qwen").request_timeout_seconds)

    def test_remote_timeouts_must_be_finite_and_above_zero(self):
        for invalid in (True, 0, -1, math.inf, -math.inf, math.nan, "30"):
            with self.subTest(config="llm", invalid=invalid):
                with self.assertRaisesRegex(ValueError, "request_timeout_seconds"):
                    LLMConfig(request_timeout_seconds=invalid)
            with self.subTest(config="gpt-vlm", invalid=invalid):
                with self.assertRaisesRegex(ValueError, "request_timeout_seconds"):
                    VLMConfig(
                        backend="gpt",
                        request_timeout_seconds=invalid,
                    )

    def test_local_qwen_rejects_an_explicit_remote_network_timeout(self):
        with self.assertRaisesRegex(ValueError, "not in-process qwen"):
            VLMConfig(backend="qwen", request_timeout_seconds=10)


class LLMTimeoutPropagationTests(unittest.TestCase):
    def test_owned_openai_client_gets_timeout_and_no_hidden_sdk_retries(self):
        config = LLMConfig(
            backend="openai",
            api_key="configured-key",
            api_base="http://openai-compatible.test/v1",
            request_timeout_seconds=12.5,
        )
        with patch("openai.OpenAI") as client_factory:
            controller = OpenAIController(config)

        client_factory.assert_called_once_with(
            api_key="configured-key",
            base_url="http://openai-compatible.test/v1",
            timeout=12.5,
            max_retries=0,
        )
        controller.close()

    def test_injected_openai_plain_and_json_calls_get_per_call_timeout(self):
        client = Mock()
        client.chat.completions.create.return_value = _openai_response("plain")
        parsed = _Payload(value="structured")
        client.beta.chat.completions.parse.return_value = SimpleNamespace(
            usage=None,
            choices=[SimpleNamespace(message=SimpleNamespace(parsed=parsed))],
        )
        controller = OpenAIController(
            LLMConfig(request_timeout_seconds=17.25),
            client=client,
        )

        self.assertEqual("plain", controller.get_completion("question"))
        self.assertIs(
            parsed,
            controller.get_json_completion("question", _Payload),
        )

        self.assertEqual(
            17.25,
            client.chat.completions.create.call_args.kwargs["timeout"],
        )
        self.assertEqual(
            17.25,
            client.beta.chat.completions.parse.call_args.kwargs["timeout"],
        )

    def test_owned_ollama_client_gets_transport_timeout(self):
        config = LLMConfig(
            backend="ollama",
            api_base="http://ollama.test",
            api_key="configured-key",
            request_timeout_seconds=21.5,
        )
        with patch("ollama.Client") as client_factory:
            controller = OllamaController(config)

        client_factory.assert_called_once_with(
            host="http://ollama.test",
            headers={"Authorization": "Bearer configured-key"},
            timeout=21.5,
        )
        controller.close()

    def test_outer_retry_count_is_exact_and_exception_secret_is_not_logged(self):
        secret = "never-log-this-api-key"
        llm = LLM.__new__(LLM)
        llm.config = LLMConfig(
            api_key=secret,
            max_retries=2,
            retry_delay_seconds=0,
            request_timeout_seconds=0.01,
        )
        llm._admission = None
        llm.llm = Mock()
        llm.llm.get_completion.side_effect = TimeoutError(secret)

        with self.assertLogs("Core.provider.llm", level="WARNING") as captured:
            with self.assertRaises(CompletionAttemptsExhausted):
                llm.get_completion("question")

        self.assertEqual(2, llm.llm.get_completion.call_count)
        self.assertNotIn(secret, "\n".join(captured.output))


class VLMTimeoutPropagationTests(unittest.TestCase):
    def test_owned_gpt_client_gets_timeout_and_no_hidden_sdk_retries(self):
        config = VLMConfig(
            backend="gpt",
            api_key="configured-key",
            api_base="http://vision.test/v1",
            request_timeout_seconds=31.0,
        )
        with patch("openai.OpenAI") as client_factory:
            controller = GPTVLMController(config)

        client_factory.assert_called_once_with(
            api_key="configured-key",
            base_url="http://vision.test/v1",
            timeout=31.0,
            max_retries=0,
        )
        controller.close()

    def test_injected_gpt_plain_and_json_calls_get_per_call_timeout(self):
        client = Mock()
        client.chat.completions.create.side_effect = [
            _openai_response("plain"),
            _openai_response('{"value":"structured"}'),
        ]
        controller = GPTVLMController(
            VLMConfig(
                backend="gpt",
                api_key="configured-key",
                request_timeout_seconds=41.75,
            ),
            client=client,
        )

        self.assertEqual("plain", controller.generate("describe"))
        self.assertEqual(
            "structured",
            controller.generate_json("describe", schema=_Payload).value,
        )
        self.assertEqual(2, client.chat.completions.create.call_count)
        for outgoing in client.chat.completions.create.call_args_list:
            self.assertEqual(41.75, outgoing.kwargs["timeout"])

    def test_owned_ollama_vlm_client_gets_transport_timeout(self):
        config = VLMConfig(
            backend="ollama",
            api_key=None,
            api_base="http://ollama.test",
            request_timeout_seconds=51.5,
        )
        with patch("ollama.Client") as client_factory:
            controller = OllamaVLMController(config)

        client_factory.assert_called_once_with(
            host="http://ollama.test",
            timeout=51.5,
        )
        controller.close()


if __name__ == "__main__":
    unittest.main()
