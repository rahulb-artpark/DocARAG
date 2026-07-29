"""What a leaf node's surroundings say about it.

A figure carries an author-written caption stating what it shows.  A displayed
equation carries none: the prose explaining it sits in the sentences on either
side, often running straight through, so the sentence after an equation can
begin mid-clause.  Both the summariser and the vector index need that
neighbourhood, and they must agree on it -- a summary written against one
window and a vector built from another would describe different things.

The window is not a fixed span.  In a worked derivation the neighbours are
themselves equations joined by "Therefore," or "i.e.,", and the sentence naming
the subject can be several nodes back: measured over one probability textbook,
50 of 1066 equations had fewer than six words of prose within one sibling, and
walking outward past the symbol-only nodes recovered real sentences for 45 of
them, two to six siblings away.
"""

from __future__ import annotations

import re


#: A neighbour needs at least this many words of natural language to be worth
#: reading.  "i.e.," and "Therefore," are connectives, not context.
PROSE_WORDS = 6

#: How far to walk past symbol-only siblings before giving up.  Six covered
#: every chain observed; further risks reaching the next derivation.
MAX_STEPS = 6

_MATH = re.compile(r"\$\$.*?\$\$|\$[^$]*\$|\\[a-zA-Z]+|[{}^_\\]", re.S)
_WORD = re.compile(r"[A-Za-z]{2,}")


def node_content(node) -> str:
    """The node's own text, as the producer wrote it."""

    return str(getattr(getattr(node, "meta_info", None), "content", "") or "").strip()


def context_contribution(node) -> str:
    """What one neighbour contributes, read the way that node is indexed.

    A figure's prose is its caption, which some producers keep in a field of
    its own rather than folding into ``content``.  Reading ``content`` alone
    would leave the vision model's account of the pixels without the sentence
    naming the subject.
    """

    meta = getattr(node, "meta_info", None)
    caption = str(getattr(meta, "caption", "") or "").strip()
    footnote = str(getattr(meta, "footnote", "") or "").strip()
    content = node_content(node)
    # Some producers already fold the caption into the content; joining them
    # again would feed the model the same sentence twice.
    if caption and caption in content:
        caption = ""
    if footnote and footnote in content:
        footnote = ""
    return " ".join(part for part in (caption, footnote, content) if part)


def prose_word_count(text: str) -> int:
    """Words remaining once LaTeX is stripped: how much language is here."""

    return len(_WORD.findall(_MATH.sub(" ", text or "")))


def is_prose(node, *, threshold: int = PROSE_WORDS) -> bool:
    return prose_word_count(context_contribution(node)) >= threshold


def parent_title(node) -> str:
    """The section heading a node sits under, when it has one."""

    return node_content(getattr(node, "parent", None))


def _siblings(node) -> tuple[list, int | None]:
    siblings = list(getattr(getattr(node, "parent", None), "children", ()) or ())
    position = next(
        (index for index, item in enumerate(siblings) if item is node), None
    )
    return siblings, position


def adjacent_context(node) -> tuple[str, str]:
    """The immediate neighbour on each side, whatever it is."""

    siblings, position = _siblings(node)
    if position is None:
        return "", ""
    before = context_contribution(siblings[position - 1]) if position > 0 else ""
    after = (
        context_contribution(siblings[position + 1])
        if position + 1 < len(siblings)
        else ""
    )
    return before, after


def nearest_prose(node, *, threshold: int = PROSE_WORDS, max_steps: int = MAX_STEPS):
    """The closest sentence on each side, walking past symbol-only siblings.

    Returns the adjacent neighbour when it is already prose, so a healthy node
    keeps the tight window that serves it and only a chain reaches further.
    """

    siblings, position = _siblings(node)
    if position is None:
        return "", ""
    before = after = ""
    for step in range(1, max_steps + 1):
        if not before:
            index = position - step
            if index >= 0 and is_prose(siblings[index], threshold=threshold):
                before = context_contribution(siblings[index])
        if not after:
            index = position + step
            if index < len(siblings) and is_prose(siblings[index], threshold=threshold):
                after = context_contribution(siblings[index])
        if before and after:
            break
    return before, after
