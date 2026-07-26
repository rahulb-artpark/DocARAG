"""A slow model call must say why it was slow.

Four hypotheses for the stalls in a graph run were each disproved by indirect
evidence. Ollama reports how long it spent itself, so the remainder of the wall
time is queue wait, and load_duration separates a model reload from generation.
"""

from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from pydantic import BaseModel

from Core.configs.llm_config import LLMConfig
from Core.provider.llm import OllamaController


class Payload(BaseModel):
    value: str


def _config() -> LLMConfig:
    return LLMConfig(
        backend="ollama",
        model_name="qwen3.6:35b",
        api_base="http://localhost:11434",
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )


def _response(**durations) -> dict:
    payload = {
        "message": {"content": '{"value": "ok"}'},
        "prompt_eval_count": 1200,
        "eval_count": 40,
    }
    payload.update(durations)
    return payload


class SlowCallLoggingTests(unittest.TestCase):
    def _run(self, wall_seconds: float, response: dict):
        client = Mock()
        client.chat.return_value = response
        controller = OllamaController(llm_config=_config(), client=client)
        ticks = iter([0.0, wall_seconds])
        with patch("Core.provider.llm.time.monotonic", lambda: next(ticks)):
            with self.assertLogs("Core.provider.llm", level="WARNING") as logs:
                controller.get_json_completion("prompt", Payload)
        return "\n".join(logs.output)

    def _run_quiet(self, wall_seconds: float, response: dict) -> None:
        client = Mock()
        client.chat.return_value = response
        controller = OllamaController(llm_config=_config(), client=client)
        ticks = iter([0.0, wall_seconds])
        with patch("Core.provider.llm.time.monotonic", lambda: next(ticks)):
            controller.get_json_completion("prompt", Payload)

    def test_a_model_reload_is_attributed_to_load_duration(self):
        output = self._run(
            100.0,
            _response(
                total_duration=99_000_000_000,
                load_duration=95_000_000_000,
                prompt_eval_duration=1_000_000_000,
                eval_duration=3_000_000_000,
            ),
        )
        self.assertIn("load=95.0s", output)
        self.assertIn("unaccounted=1.0s", output)

    def test_a_queue_wait_shows_up_as_unaccounted_time(self):
        # The server says it only worked for 2s, so the other 58s was waiting.
        output = self._run(
            60.0,
            _response(
                total_duration=2_000_000_000,
                load_duration=0,
                prompt_eval_duration=500_000_000,
                eval_duration=1_500_000_000,
            ),
        )
        self.assertIn("unaccounted=58.0s", output)
        self.assertIn("load=0.0s", output)

    def test_slow_generation_is_not_blamed_on_waiting(self):
        output = self._run(
            40.0,
            _response(
                total_duration=39_500_000_000,
                load_duration=0,
                prompt_eval_duration=500_000_000,
                eval_duration=39_000_000_000,
            ),
        )
        self.assertIn("eval=39.0s", output)
        self.assertIn("unaccounted=0.5s", output)

    def test_token_counts_are_reported(self):
        output = self._run(30.0, _response(total_duration=30_000_000_000))
        self.assertIn("prompt_tokens=1200", output)
        self.assertIn("eval_tokens=40", output)

    def test_a_fast_call_logs_nothing(self):
        with patch.object(OllamaController, "_log_slow_call") as spy:
            self._run_quiet(0.4, _response(total_duration=400_000_000))
        spy.assert_called_once()
        self.assertLess(spy.call_args.args[1], 5.0)

    def test_a_response_without_timings_does_not_raise(self):
        output = self._run(30.0, _response())
        self.assertIn("wall=30.0s", output)


if __name__ == "__main__":
    unittest.main()
