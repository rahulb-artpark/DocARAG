from dataclasses import dataclass


@dataclass
class FigureConfig:
    """Recovery of text and meaning from figure regions.

    MinerU emits an image region as a crop with no interior text.  ``ocr``
    re-reads that crop with the detection/recognition models MinerU already
    ships; ``describe`` asks the configured vision model what the figure shows,
    grounded on the recovered text.
    """

    ocr: bool = True
    describe: bool = True
    ocr_lang: str = "en"
    min_confidence: float = 0.5

    def __post_init__(self):
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("figures.min_confidence must be within [0, 1]")
        if not self.ocr_lang or self.ocr_lang != self.ocr_lang.strip():
            raise ValueError("figures.ocr_lang must be a trimmed language code")
