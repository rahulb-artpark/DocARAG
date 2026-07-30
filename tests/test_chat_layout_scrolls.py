"""The chat pane must give its overflow to the message list, not to the page.

Guarding one specific regression rather than testing CSS in general. Adding the
conversation sidebar wrapped .chatShell in a grid (.chatLayout). A grid sizes an
unconstrained row to its tallest item, and a grid item defaults to
min-height: auto, so .chatShell grew to fit every message: .messages never
overflowed and so showed no scrollbar, and the composer holding the Ask button
was pushed below the viewport, which reads as the chat freezing after the first
long answer. Before the wrapper, .chatShell was a direct flex child of #chatView
and took its definite height from align-items: stretch.

There is no browser or jsdom in this environment to assert on real layout, so
these check the declarations the fix depends on. That is weak as a CSS test and
deliberate as a regression guard: each of the three, removed on its own, brings
the frozen chat back and nothing else here would notice.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
STYLES = REPO / "Scripts" / "webui" / "styles.css"


def rule_body(selector: str) -> str:
    """The declarations of the first rule whose selector list starts with this."""

    css = STYLES.read_text(encoding="utf-8")
    match = re.search(
        r"(?:^|\})\s*" + re.escape(selector) + r"\s*(?:,[^{]*)?\{(?P<body>[^}]*)\}",
        css,
        re.MULTILINE,
    )
    assert match, f"no rule found for {selector}"
    return match.group("body")


def declares(selector: str, prop: str, value: str) -> bool:
    body = rule_body(selector)
    pattern = re.compile(
        rf"(?:^|;)\s*{re.escape(prop)}\s*:\s*(?P<value>[^;]+)", re.MULTILINE
    )
    return any(
        match.group("value").strip().rstrip(";") == value for match in pattern.finditer(body)
    )


class TheMessageListOwnsTheOverflow(unittest.TestCase):
    def test_the_grid_row_is_bounded(self):
        # Without an explicit bounded row the row is auto-sized and grows to the
        # tallest item, so nothing downstream can ever overflow.
        self.assertTrue(
            declares(".chatLayout", "grid-template-rows", "minmax(0, 1fr)"),
            ".chatLayout must bound its row or the chat pane grows past the viewport",
        )

    def test_the_chat_pane_may_shrink_below_its_content(self):
        # A grid item's default min-height: auto is what pinned this open.
        self.assertTrue(
            declares(".chatShell", "min-height", "0"),
            ".chatShell needs min-height: 0 to shrink as a grid item",
        )

    def test_the_message_list_scrolls_and_may_shrink(self):
        self.assertTrue(declares(".messages", "overflow", "auto"))
        self.assertTrue(
            declares(".messages", "min-height", "0"),
            ".messages needs min-height: 0 to shrink as a flex item",
        )

    def test_the_sidebar_may_shrink_too(self):
        # Its conversation list scrolls on the same principle.
        self.assertTrue(declares(".chatSidebar", "min-height", "0"))


class TheScrollbarSitsAtTheEdgeOfTheColumn(unittest.TestCase):
    """A centred scrolling box puts its scrollbar in the middle of the window.

    .chatShell used to carry max-width: 1080px with auto margins, which made the
    scrolling element itself a centred 1080px box: the scrollbar rendered at that
    box's right edge with a dead gutter beside it. The reading width is now an
    inset on the content instead, so the panes fill their grid column.
    """

    def test_the_chat_pane_is_not_narrowed(self):
        self.assertFalse(
            re.search(r"max-width\s*:", rule_body(".chatShell")),
            ".chatShell must fill its column; inset the content instead",
        )

    def test_the_chat_pane_is_not_centred_by_margins(self):
        self.assertFalse(
            re.search(r"margin\s*:\s*0\s+auto", rule_body(".chatShell")),
            "auto margins on .chatShell re-centre the scrolling box",
        )

    def test_the_reading_width_is_an_inset_on_the_message_list(self):
        self.assertIn("1080px", rule_body(".messages"))

    def test_the_composer_uses_the_same_inset(self):
        # Otherwise the input stops lining up with the messages above it.
        self.assertIn("1080px", rule_body(".composer"))


class TheComposerStaysReachable(unittest.TestCase):
    def test_the_composer_is_not_inside_the_scrolling_region(self):
        # The Ask button must not scroll away with the transcript.
        markup = (REPO / "Scripts" / "webui" / "index.html").read_text(encoding="utf-8")
        messages_at = markup.index('id="messages"')
        composer_at = markup.index('id="chatForm"')
        self.assertLess(messages_at, composer_at)
        between = markup[messages_at:composer_at]
        self.assertNotIn("<div class=\"messages\"", between)


if __name__ == "__main__":
    unittest.main()
