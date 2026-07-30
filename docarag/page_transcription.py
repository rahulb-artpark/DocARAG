"""Decide which scanned pages need re-transcribing, and transcribe them.

MinerU 3.4.4's ``hybrid-engine`` is a good layout model and a weaker recogniser.
On a scanned page it finds the headings -- 14 on 10P_CH11 where 2.1.11 found 3 --
and then reads ``Near point of the eye gradually reduces away`` as ``Neaoint of
th ey gradally sudues way``.  Layout, geometry, block types and reading order
stay with MinerU; only the characters are replaced.

Deliberately not a handwriting detector.  A scan carries whatever was on the
paper -- printed text, cursive, or both on one page -- and faded print, a skewed
feed or a low-resolution scan degrade recognition just as cursive does.  Since
the remedy is the same in every case, the question asked here is only *was this
page scanned, and did the recogniser do badly on it*, which needs no script
classification and no second model.

A page is transcribed in horizontal bands rather than block by block.  Measured
on 10P_CH11 page 2 that is two inferences against roughly twenty, and banding is
also the more accurate: the surrounding words are what disambiguate a doubtful
glyph, so a three-word crop gives the model room to invent, while a whole dense
sheet lets it stop early.  The cost is that transcribed lines must be matched
back onto MinerU's blocks.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import re
import urllib.request
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'’\-]{2,}")
_RENDER_DPI = 200
_MAX_EDGE = 1600

TRANSCRIBE_PROMPT = (
    "Transcribe every line of text on this scanned page exactly as written, "
    "whether it is printed or handwritten. Read the page in its natural reading "
    "order; if it is in columns, finish the left column before starting the "
    "right. Write each line on its own line, and render bullet markers as '-'. "
    "Do not correct spelling, do not summarise, and do not add commentary. "
    "Ignore any faint printed watermark across the page. Output only the "
    "transcription."
)


@dataclass(frozen=True, slots=True)
class PageTextQuality:
    """Why a page was or was not judged to need re-transcription."""

    page_index: int
    has_text_layer: bool
    word_count: int
    lexicon_validity: float
    needs_transcription: bool
    reason: str


@dataclass(slots=True)
class TranscriptionOutcome:
    """What a transcription pass changed, for the run's diagnostics."""

    pages_examined: int = 0
    pages_transcribed: int = 0
    blocks_replaced: int = 0
    blocks_unmatched: int = 0
    failures: list[str] = field(default_factory=list)
    pages: list[dict[str, object]] = field(default_factory=list)


def _load_lexicon() -> frozenset[str]:
    """Words judged real. A system word list when there is one, else nothing.

    With no lexicon every page would score 0.0 validity, which would route
    perfectly good typed documents to the VLM.  ``page_text_quality`` therefore
    declines to judge by validity at all unless a lexicon was found.
    """

    for path in ("/usr/share/dict/words", "/usr/dict/words"):
        try:
            with open(path, encoding="utf-8", errors="ignore") as handle:
                words = {line.strip().casefold() for line in handle}
            if len(words) > 1000:
                return frozenset(words)
        except OSError:
            continue
    return frozenset()


_LEXICON = _load_lexicon()


def lexicon_validity(text: str) -> tuple[int, float]:
    """Fraction of a page's words that are real words, with the word count.

    This is the signal that separates a bad recognition from a good one, and it
    is indifferent to whether the page was printed or written.  MinerU's own
    span confidence is not usable for this: it scored 0.957 and 0.824 on the
    Presbyopia lines it had largely misread.
    """

    words = _WORD_RE.findall(text or "")
    if not words or not _LEXICON:
        return len(words), 0.0
    known = sum(1 for word in words if word.casefold().strip("'’-") in _LEXICON)
    return len(words), known / len(words)


def page_text_quality(
    page_index: int,
    *,
    parsed_text: str,
    has_text_layer: bool,
    min_validity: float,
    min_words: int,
) -> PageTextQuality:
    """Judge one page. No model: only the parser's own output and a word list."""

    words, validity = lexicon_validity(parsed_text)

    def verdict(needs: bool, reason: str) -> PageTextQuality:
        return PageTextQuality(
            page_index=page_index,
            has_text_layer=has_text_layer,
            word_count=words,
            lexicon_validity=round(validity, 4),
            needs_transcription=needs,
            reason=reason,
        )

    # A page with a real text layer was not scanned, so whatever MinerU read it
    # read from the document's own characters rather than from pixels. There is
    # nothing for a recogniser to get wrong and nothing for a VLM to improve.
    if has_text_layer:
        return verdict(False, "page has an extractable text layer")
    if not _LEXICON:
        return verdict(False, "no system lexicon available to judge validity")
    if words < min_words:
        return verdict(False, f"only {words} words, too few to judge")
    # A clean scan of printed text scores near 1.0 here, so one threshold sorts
    # good scans from bad without asking what the page contains.
    if validity >= min_validity:
        return verdict(False, f"lexicon validity {validity:.2f} is acceptable")
    return verdict(True, f"scanned page with lexicon validity {validity:.2f}")


def page_has_text_layer(pdf_path: str, page_index: int, *, min_chars: int = 24) -> bool:
    """True when the page carries extractable characters of its own.

    A scanner that ran its own OCR leaves a text layer, so this is not "was it
    scanned" but "does the PDF already know its own characters" -- which is the
    question that matters, since that layer is what MinerU reads.
    """

    import pypdfium2 as pdfium

    document = pdfium.PdfDocument(pdf_path)
    try:
        page = document[page_index]
        textpage = page.get_textpage()
        try:
            extracted = textpage.get_text_bounded() or ""
        finally:
            textpage.close()
    finally:
        document.close()
    return len(extracted.strip()) >= min_chars


def render_page_png(
    pdf_path: str,
    page_index: int,
    *,
    dpi: int = _RENDER_DPI,
    bands: int = 1,
    overlap: float = 0.06,
) -> list[bytes]:
    """Render a page for the VLM as horizontal bands, top to bottom.

    One image per page is not reliable on a dense page: asked for a whole sheet
    of two-column notes the model transcribed the upper two thirds and stopped
    at ``Astigmatism.``, well short of any token limit -- and non-deterministically,
    since an earlier call on the same page had reached the bottom. Bands bound
    how much it can decide it has finished with.

    Bands overlap slightly so a line falling on a cut is whole in one of them.
    The duplicate lines that produces need no removal: alignment is monotonic
    and advances past a line once claimed, so a repeat is simply never used.
    """

    import pypdfium2 as pdfium
    from PIL import Image

    document = pdfium.PdfDocument(pdf_path)
    try:
        image = document[page_index].render(scale=dpi / 72).to_pil().convert("RGB")
    finally:
        document.close()

    width, height = image.size
    count = max(1, bands)
    step = height / count
    margin = step * max(0.0, overlap)
    rendered: list[bytes] = []
    for index in range(count):
        top = max(0, int(index * step - margin))
        bottom = min(height, int((index + 1) * step + margin))
        crop = image.crop((0, top, width, bottom))
        if max(crop.size) > _MAX_EDGE:
            crop.thumbnail((_MAX_EDGE, _MAX_EDGE), Image.LANCZOS)
        buffer = io.BytesIO()
        crop.save(buffer, format="JPEG", quality=92)
        rendered.append(buffer.getvalue())
    return rendered


def transcribe_page_image(
    image_bytes: bytes,
    *,
    model: str,
    api_base: str,
    timeout: float = 600.0,
    temperature: float = 0.1,
    max_tokens: int = 2400,
) -> str:
    """Transcribe a rendered page with an Ollama vision model.

    ``think`` is sent false deliberately.  A reasoning-capable model otherwise
    returns an empty ``response`` with its output in ``thinking``, which looks
    exactly like a silent failure -- it cost two apparently-successful calls
    that produced nothing at all.
    """

    payload = json.dumps(
        {
            "model": model,
            "prompt": TRANSCRIBE_PROMPT,
            "images": [base64.b64encode(image_bytes).decode("ascii")],
            "stream": False,
            "think": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{api_base.rstrip('/')}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read())
    text = body.get("response") or ""
    if not text.strip():
        raise RuntimeError(
            "vision model returned no transcription "
            f"(done_reason={body.get('done_reason')!r})"
        )
    return text


def _alignment_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (value or "").casefold())


def _score(left: str, right: str) -> float:
    """How alike two readings of the same line are, ignoring length.

    Not reconciliation's ``_similarity``: that gates on a 0.85 length ratio,
    which is right for pairing two renderings of identical text and wrong here.
    ``Power of accomodation of eye devreases. Caures`` against its correct
    reading scores 0.83 on that gate and would be discarded, when it is in fact
    the match.
    """

    from difflib import SequenceMatcher

    a, b = _alignment_key(left), _alignment_key(right)
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def align_transcription(
    block_texts: list[str],
    lines: list[str],
    *,
    min_score: float = 0.55,
    lookahead: int | None = None,
    max_span: int = 4,
    as_lines: bool = False,
) -> list[str | None] | list[tuple[str, ...] | None]:
    """Match each block, in reading order, to the lines that re-read it.

    Both sequences describe the same page in the same order, so the alignment is
    monotonic: a block can only claim lines after the ones already claimed.  That
    ordering is what makes this reliable where independent best-match is not --
    ``↓ Causes`` and ``↓ Correction`` appear three times each on this page and are
    indistinguishable to a text score, but never ambiguous in sequence.

    A block may correspond to several lines, because MinerU groups wrapped lines
    into one block while the transcription keeps them separate; spans of up to
    ``max_span`` lines are tried and the best-scoring one wins.  A block that
    finds nothing convincing returns ``None`` and keeps the parser's own text.

    ``lookahead`` searches to the end of the page by default.  A narrow window
    fails here: the transcription reads lines MinerU dropped altogether, and an
    unmatched line does not advance the cursor, so a run of them wider than the
    window breaks the chain for every block after it.  On 10P_CH11 page 2 a
    six-line window matched 9 of 21 blocks and the whole page matched 15.  The
    search is O(blocks x lines x max_span), a few thousand comparisons per page.
    """

    matched: list[str | None] = []
    cursor = 0
    for text in block_texts:
        best: tuple[float, int, int] | None = None
        if _alignment_key(text):
            limit = len(lines) if lookahead is None else min(len(lines), cursor + lookahead)
            for start in range(cursor, limit):
                for span in range(1, max_span + 1):
                    if start + span > len(lines):
                        break
                    candidate = " ".join(lines[start : start + span])
                    score = _score(text, candidate)
                    if best is None or score > best[0]:
                        best = (score, start, span)
        if best is not None and best[0] >= min_score:
            _, start, span = best
            claimed = tuple(lines[start : start + span])
            # A list block keeps its items separate: its text lives in
            # list_items, and the adapter decides to build Markdown from that
            # by checking that "text" is absent. Handing back a joined string
            # would flip that check and cost the list its structure.
            matched.append(claimed if as_lines else " ".join(claimed))
            cursor = start + span
        else:
            matched.append(None)
    return matched


def transcription_lines(text: str) -> list[str]:
    """One entry per transcribed line, bullet markers and blanks removed."""

    lines: list[str] = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        line = re.sub(r"^[-•·▪◦‣*]+\s*", "", line).strip()
        if line:
            lines.append(line)
    return lines


def _block_text(block: dict) -> str:
    """A reconciled block's current text, wherever the parser put it."""

    from docarag.adapters.mineru_compat import content_block_text_parts

    return " ".join(content_block_text_parts(block))


def _replace_block_text(block: dict, lines: tuple[str, ...]) -> None:
    """Install a re-read of this block, keeping what the parser said."""

    block.setdefault("parser_text", _block_text(block))
    block["text_source"] = "vlm_page_transcription"
    if isinstance(block.get("list_items"), list):
        block["list_items"] = list(lines)
    elif isinstance(block.get("code_body"), str):
        block["code_body"] = "\n".join(lines)
    else:
        block["text"] = " ".join(lines)


def retranscribe_scanned_pages(
    blocks: list[dict],
    *,
    pdf_path: str,
    model: str,
    api_base: str,
    min_validity: float = 0.90,
    min_words: int = 12,
    dpi: int = _RENDER_DPI,
    bands: int = 2,
    timeout: float = 600.0,
) -> TranscriptionOutcome:
    """Re-read every scanned page whose text the recogniser got wrong.

    Operates on reconciled blocks, after middle-json and content-list have been
    paired.  Doing it earlier would mean rewriting one view and not the other,
    and reconciliation matches the two by text -- the rewritten view would stop
    matching and every block on the page would be emitted twice, which is the
    duplication this pipeline just finished removing.

    Blocks are mutated in place.  A block whose line cannot be found keeps the
    parser's text untouched, so a partial transcription degrades rather than
    blanks the page.
    """

    outcome = TranscriptionOutcome()
    by_page: dict[int, list[dict]] = {}
    for block in blocks:
        page = block.get("page_idx")
        if isinstance(page, int):
            by_page.setdefault(page, []).append(block)

    for page_index in sorted(by_page):
        page_blocks = [b for b in by_page[page_index] if _block_text(b).strip()]
        if not page_blocks:
            continue
        outcome.pages_examined += 1
        try:
            has_layer = page_has_text_layer(pdf_path, page_index)
        except Exception as error:  # a render/parse problem must not fail the run
            log.warning("page %d text-layer probe failed: %s", page_index, error)
            has_layer = True
        quality = page_text_quality(
            page_index,
            parsed_text=" ".join(_block_text(b) for b in page_blocks),
            has_text_layer=has_layer,
            min_validity=min_validity,
            min_words=min_words,
        )
        record: dict[str, object] = {
            "page_index": page_index,
            "lexicon_validity": quality.lexicon_validity,
            "word_count": quality.word_count,
            "needs_transcription": quality.needs_transcription,
            "reason": quality.reason,
        }
        if not quality.needs_transcription:
            outcome.pages.append(record)
            continue

        try:
            transcript = "\n".join(
                transcribe_page_image(
                    band, model=model, api_base=api_base, timeout=timeout
                )
                for band in render_page_png(
                    pdf_path, page_index, dpi=dpi, bands=bands
                )
            )
        except Exception as error:
            message = f"page {page_index}: {type(error).__name__}: {error}"
            log.warning("page transcription failed, keeping parser text: %s", message)
            outcome.failures.append(message)
            record["failed"] = message
            outcome.pages.append(record)
            continue

        lines = transcription_lines(transcript)
        matches = align_transcription(
            [_block_text(b) for b in page_blocks], lines, as_lines=True
        )
        replaced = 0
        for block, claimed in zip(page_blocks, matches):
            if claimed:
                _replace_block_text(block, claimed)
                replaced += 1
        after = " ".join(_block_text(b) for b in page_blocks)
        outcome.pages_transcribed += 1
        outcome.blocks_replaced += replaced
        outcome.blocks_unmatched += len(page_blocks) - replaced
        record.update(
            {
                "transcribed_lines": len(lines),
                "blocks": len(page_blocks),
                "blocks_replaced": replaced,
                "lexicon_validity_after": round(lexicon_validity(after)[1], 4),
            }
        )
        outcome.pages.append(record)
        log.info(
            "page %d re-transcribed: %d/%d blocks replaced, validity %.2f -> %.2f",
            page_index, replaced, len(page_blocks),
            quality.lexicon_validity, lexicon_validity(after)[1],
        )
    return outcome
