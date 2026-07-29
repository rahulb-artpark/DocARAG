from dataclasses import dataclass


@dataclass
class FigureConfig:
    """Recovery of text and meaning from figure regions.

    MinerU emits an image region as a crop with no interior text.  ``ocr``
    re-reads that crop with the detection/recognition models MinerU already
    ships; ``describe`` asks the configured vision model what the figure shows,
    grounded on the recovered text; ``diagrams`` asks that model for the
    node/edge topology a description flattens away.
    ``transcribe_formulas`` has that model re-read each equation crop, because
    the dedicated formula recognizer drops lines it cannot segment.
    ``transcribe_tables`` does the same for table crops, whose recognizer fails
    the same way on notation: a trigonometric ratios table came back with
    ``1|2`` for a half and empty cells where one over root two belonged.  The
    re-reading is admitted only when its row and column counts match what the
    recognizer found, so an invented table cannot replace a damaged one.
    """

    ocr: bool = True
    describe: bool = True
    diagrams: bool = True
    transcribe_formulas: bool = True
    transcribe_tables: bool = True
    ocr_lang: str = "en"
    min_confidence: float = 0.5

    def __post_init__(self):
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("figures.min_confidence must be within [0, 1]")
        if not self.ocr_lang or self.ocr_lang != self.ocr_lang.strip():
            raise ValueError("figures.ocr_lang must be a trimmed language code")
