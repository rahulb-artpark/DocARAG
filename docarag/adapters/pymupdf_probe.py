"""Optional lazy PyMuPDF backend for the source-bound PDF page probe.

Importing this module never imports PyMuPDF.  The optional runtime is loaded
only when ``open()`` is called.  Page inspection uses native text blocks,
image metadata without hashes/xrefs, and referenced fonts; it never renders a
page, extracts image bytes, invokes OCR, or invokes a parser/model.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import importlib
from pathlib import Path

from docarag.page_probe import (
    PageProbeBackendError,
    PageProbeDependencyError,
    PdfPageSnapshot,
    PdfProbeBackend,
    PdfProbeSession,
)


def _load_pymupdf() -> object:
    """Load modern PyMuPDF, with its legacy ``fitz`` import as a fallback."""

    try:
        return importlib.import_module("pymupdf")
    except ModuleNotFoundError:
        try:
            return importlib.import_module("fitz")
        except ModuleNotFoundError as legacy_error:
            raise PageProbeDependencyError(
                "PyMuPDF page probing requires the optional 'PyMuPDF' package"
            ) from legacy_error
        except Exception as exc:
            raise PageProbeDependencyError(
                f"legacy PyMuPDF module 'fitz' could not be imported: {exc}"
            ) from exc
    except Exception as exc:
        raise PageProbeDependencyError(
            f"PyMuPDF module could not be imported: {exc}"
        ) from exc


class PyMuPdfProbeBackend(PdfProbeBackend):
    """Cheap PyMuPDF object-model inspection with a lazy vendor import."""

    def __init__(
        self,
        *,
        module_loader: Callable[[], object] | None = None,
    ) -> None:
        self._module_loader = module_loader or _load_pymupdf
        self._runtime_version = "unloaded"

    @property
    def backend_name(self) -> str:
        return "pymupdf"

    @property
    def backend_version(self) -> str:
        return self._runtime_version

    def open(self, source_path: Path) -> PdfProbeSession:
        try:
            module = self._module_loader()
        except PageProbeDependencyError:
            raise
        except ModuleNotFoundError as exc:
            raise PageProbeDependencyError(
                "PyMuPDF page probing requires the optional 'PyMuPDF' package"
            ) from exc
        except Exception as exc:
            raise PageProbeDependencyError(
                f"PyMuPDF module could not be loaded: {exc}"
            ) from exc
        open_document = getattr(module, "open", None)
        if not callable(open_document):
            raise PageProbeDependencyError("PyMuPDF module does not define open()")
        self._runtime_version = _module_version(module)
        try:
            document = open_document(filename=str(source_path))
        except Exception as exc:
            raise PageProbeBackendError(f"PyMuPDF could not open the PDF: {exc}") from exc
        try:
            if not bool(getattr(document, "is_pdf", False)):
                raise PageProbeBackendError("PyMuPDF source is not a PDF")
            if bool(getattr(document, "needs_pass", False)):
                raise PageProbeBackendError(
                    "PyMuPDF source is encrypted and requires a password"
                )
            return _PyMuPdfProbeSession(
                document=document,
                runtime_version=self._runtime_version,
            )
        except BaseException:
            try:
                document.close()
            except Exception:
                pass
            raise


class _PyMuPdfProbeSession(PdfProbeSession):
    def __init__(self, *, document: object, runtime_version: str) -> None:
        self._document = document
        self._runtime_version = runtime_version
        self._closed = False

    @property
    def page_count(self) -> int:
        self._require_open()
        return getattr(self._document, "page_count")

    def inspect_page(self, page_index: int) -> PdfPageSnapshot:
        self._require_open()
        try:
            page = self._document.load_page(page_index)  # type: ignore[attr-defined]
            cropbox = page.cropbox
            cropbox_position = page.cropbox_position
            width = float(cropbox.width)
            height = float(cropbox.height)
            offset_x = float(cropbox_position.x)
            offset_y = float(cropbox_position.y)
            blocks = page.get_text("blocks", sort=False)
            text_parts: list[str] = []
            text_boxes: list[tuple[float, float, float, float]] = []
            for block in blocks:
                if not isinstance(block, Sequence) or len(block) < 7:
                    continue
                if block[6] != 0:
                    continue
                value = block[4]
                if isinstance(value, str):
                    text_parts.append(value)
                text_boxes.append(
                    _offset_box(block[:4], offset_x=offset_x, offset_y=offset_y)
                )

            image_info = page.get_image_info(hashes=False, xrefs=False)
            image_boxes: list[tuple[float, float, float, float]] = []
            for item in image_info:
                if not isinstance(item, Mapping) or "bbox" not in item:
                    continue
                image_boxes.append(
                    _offset_box(
                        item["bbox"], offset_x=offset_x, offset_y=offset_y
                    )
                )

            fonts = page.get_fonts(full=False)
            font_keys = {
                tuple(str(part) for part in font)
                if isinstance(font, Sequence) and not isinstance(font, str)
                else (str(font),)
                for font in fonts
            }
            return PdfPageSnapshot(
                page_index=page_index,
                width=width,
                height=height,
                rotation_degrees=int(page.rotation),
                native_text="\n".join(text_parts),
                text_boxes=tuple(text_boxes),
                image_boxes=tuple(image_boxes),
                font_count=len(font_keys),
                # Counting vector paths requires a separate, more expensive
                # display-list walk.  It is deliberately unknown in this probe.
                vector_path_count=None,
                metadata={
                    "pymupdf_version": self._runtime_version,
                    "text_extraction": "get_text_blocks_source_order",
                    "image_extraction": "get_image_info_without_hashes_or_xrefs",
                    "cropbox_offset_x": offset_x,
                    "cropbox_offset_y": offset_y,
                    "rendered": False,
                    "ocr_invoked": False,
                    "image_bytes_extracted": False,
                },
            )
        except PageProbeBackendError:
            raise
        except Exception as exc:
            raise PageProbeBackendError(
                f"PyMuPDF could not inspect page {page_index}: {exc}"
            ) from exc

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._document.close()  # type: ignore[attr-defined]
        except Exception as exc:
            raise PageProbeBackendError(
                f"PyMuPDF could not close the PDF: {exc}"
            ) from exc

    def _require_open(self) -> None:
        if self._closed:
            raise PageProbeBackendError("PyMuPDF probe session is closed")


def _offset_box(
    value: object,
    *,
    offset_x: float,
    offset_y: float,
) -> tuple[float, float, float, float]:
    if isinstance(value, (str, bytes, bytearray)):
        # Preserve a visibly invalid sentinel so the generic trust boundary
        # marks coverage unknown instead of silently treating it as zero.
        return (float("nan"),) * 4
    try:
        items = tuple(value)  # type: ignore[arg-type]
        if len(items) != 4:
            return (float("nan"),) * 4
        x0, y0, x1, y1 = (float(item) for item in items)
    except (TypeError, ValueError):
        return (float("nan"),) * 4
    return (
        x0 - offset_x,
        y0 - offset_y,
        x1 - offset_x,
        y1 - offset_y,
    )


def _module_version(module: object) -> str:
    for attribute in ("__version__", "VersionBind"):
        value = getattr(module, attribute, None)
        if isinstance(value, str) and value.strip():
            return value
    version = getattr(module, "version", None)
    if isinstance(version, Sequence) and not isinstance(version, str) and version:
        return str(version[0])
    return "unknown"


__all__ = ["PyMuPdfProbeBackend"]
