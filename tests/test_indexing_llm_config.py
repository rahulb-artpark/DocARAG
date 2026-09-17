from __future__ import annotations

import unittest
from unittest.mock import Mock

from pydantic import BaseModel

from retrieval_engine.configuration.language_model import LLMConfig, indexing_llm_config
from retrieval_engine.providers.language_model import OllamaController


class Payload(BaseModel):
    value: str


def _ollama_config(**overrides) -> LLMConfig:
    base = {
        "backend": "ollama",
        "model_name": "qwen3.6:35b",
        "api_base": "http://localhost:11434",
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
    }
    base.update(overrides)
    return LLMConfig(**base)


class IndexingLLMConfigTests(unittest.TestCase):
    def test_unset_thinking_is_disabled(self):
        self.assertIs(False, indexing_llm_config(_ollama_config()).think)

    def test_explicitly_enabled_thinking_is_honored(self):
        config = _ollama_config(think=True)
        self.assertIs(True, indexing_llm_config(config).think)

    def test_explicitly_disabled_thinking_is_left_alone(self):
        config = _ollama_config(think=False)
        self.assertIs(config, indexing_llm_config(config))

    def test_the_caller_s_config_is_not_mutated(self):
        config = _ollama_config()
        indexing_llm_config(config)
        self.assertIsNone(config.think)

    def test_every_other_setting_survives(self):
        config = _ollama_config(max_tokens=4321, num_ctx=16384, max_workers=4)
        derived = indexing_llm_config(config)
        self.assertEqual(4321, derived.max_tokens)
        self.assertEqual(16384, derived.num_ctx)
        self.assertEqual(4, derived.max_workers)
        self.assertEqual("qwen3.6:35b", derived.model_name)


class IndexingThinkingReachesOllamaTests(unittest.TestCase):
    """The derived config must actually put ``think=False`` on the wire."""

    def _controller(self, config: LLMConfig) -> tuple[OllamaController, Mock]:
        client = Mock()
        client.chat.return_value = {
            "message": {"content": '{"value": "ok"}'},
            "prompt_eval_count": 3,
            "eval_count": 2,
        }
        return OllamaController(llm_config=config, client=client), client

    def test_structured_completion_sends_think_false(self):
        controller, client = self._controller(indexing_llm_config(_ollama_config()))
        controller.get_json_completion("prompt", Payload)
        self.assertIs(False, client.chat.call_args.kwargs["think"])

    def test_unconfigured_structured_completion_sends_no_think_flag(self):
        # The regression this guards: an unset think leaves the flag off the
        # request entirely, so the model reasons until num_predict runs out.
        controller, client = self._controller(_ollama_config())
        controller.get_json_completion("prompt", Payload)
        self.assertNotIn("think", client.chat.call_args.kwargs)

    def test_indexing_config_overrides_a_per_call_thinking_request(self):
        controller, client = self._controller(indexing_llm_config(_ollama_config()))
        controller.get_json_completion("prompt", Payload, think_mode=True)
        self.assertIs(False, client.chat.call_args.kwargs["think"])


if __name__ == "__main__":
    unittest.main()
