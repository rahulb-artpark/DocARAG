"""Figure OCR backed by the detection/recognition models MinerU already ships.

MinerU bundles PaddleOCR's detection and recognition models ported to PyTorch
and downloads their weights for the pipeline backend.  Its pipeline never
applies them to the interior of an image region, so this drives the same
in-process models directly over a figure crop.
"""

from __future__ import annotations

import logging
import threading

from document_services.enrichment.figure_enrichment import FigureOCRResult, OCRSpan

log = logging.getLogger(__name__)

DEFAULT_DET_DB_BOX_THRESH = 0.3


class MinerUFigureOCR:
    """Recognize text inside a figure crop with MinerU's bundled OCR models."""

    def __init__(
        self,
        *,
        lang: str = "en",
        det_db_box_thresh: float = DEFAULT_DET_DB_BOX_THRESH,
    ) -> None:
        self._lang = lang or "en"
        self._det_db_box_thresh = det_db_box_thresh
        self._engine: object | None = None
        self._lock = threading.Lock()

    def _ocr_engine(self) -> object:
        if self._engine is None:
            with self._lock:
                if self._engine is None:
                    from mineru.backend.pipeline.model_init import (
                        AtomicModel,
                        atom_model_init,
                    )

                    self._engine = atom_model_init(
                        AtomicModel.OCR,
                        det_db_box_thresh=self._det_db_box_thresh,
                        lang=self._lang,
                    )
        return self._engine

    def __call__(self, data: bytes) -> FigureOCRResult:
        import cv2
        import numpy as np

        image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("figure crop is not a decodable image")
        height, width = image.shape[:2]
        raw = self._ocr_engine().ocr(image, det=True, rec=True)
        page = raw[0] if raw else None
        spans: list[OCRSpan] = []
        for entry in page or ():
            if not entry:
                continue
            polygon, recognition = entry[0], entry[1]
            if not polygon or not recognition:
                continue
            text = recognition[0]
            if not isinstance(text, str) or not text.strip():
                continue
            xs = [float(point[0]) for point in polygon]
            ys = [float(point[1]) for point in polygon]
            spans.append(
                OCRSpan(
                    text=text.strip(),
                    confidence=float(recognition[1]),
                    bbox=(min(xs), min(ys), max(xs), max(ys)),
                )
            )
        return FigureOCRResult(spans=tuple(spans), width=int(width), height=int(height))


__all__ = ["MinerUFigureOCR"]
