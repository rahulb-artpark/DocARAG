"""A retried VLM failure must say what the provider actually refused.

Six figures were dropped from a knowledge graph by a provider 400. The run
recorded "VLM JSON generation attempt 1/3 failed (RuntimeError)" three times per
figure and nothing else: the type of a wrapper exception whose own message names
only the model. The server's text -- "illegal base64 data at input byte 3" --
and the image the request carried were both discarded, so the log could not say
which payload had been refused.
"""

from __future__ import annotations

import unittest

from Core.provider.vlm import _root_cause


class TheRootCauseIsTheOneThatSaysWhatBroke(unittest.TestCase):
    def test_it_reaches_past_a_wrapper(self):
        inner = ValueError("illegal base64 data at input byte 3")
        try:
            try:
                raise inner
            except ValueError as error:
                raise RuntimeError("Could not get a JSON response") from error
        except RuntimeError as outer:
            self.assertEqual("illegal base64 data at input byte 3", _root_cause(outer))

    def test_it_reaches_past_several_wrappers(self):
        error: BaseException = ValueError("the real fault")
        for depth in range(3):
            try:
                raise RuntimeError(f"wrapper {depth}") from error
            except RuntimeError as wrapped:
                error = wrapped
        self.assertEqual("the real fault", _root_cause(error))

    def test_an_unwrapped_error_is_its_own_cause(self):
        self.assertEqual("plain", _root_cause(ValueError("plain")))

    def test_a_message_less_error_still_names_its_type(self):
        self.assertEqual("KeyboardInterrupt", _root_cause(KeyboardInterrupt()))

    def test_a_cycle_cannot_hang_the_walk(self):
        first = RuntimeError("first")
        second = RuntimeError("second")
        first.__cause__ = second
        second.__cause__ = first
        self.assertIn(_root_cause(first), {"first", "second"})


if __name__ == "__main__":
    unittest.main()
