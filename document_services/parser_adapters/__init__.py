"""Optional parser/storage/model adapter implementations."""

from document_services.parser_adapters.docling import (
    ADAPTER_NAME as DOCLING_EXPORT_ADAPTER_NAME,
    DoclingExportAdapter,
    DoclingExportLoader,
    translate_docling_export,
)
from document_services.parser_adapters.mineru_compat import (
    ADAPTER_NAME as MINERU_NORMALIZED_ADAPTER_NAME,
    MinerUNormalizedAdapter,
    NormalizedBlockLoader,
    translate_normalized_blocks,
)
from document_services.parser_adapters.paddleocr_vl import (
    ADAPTER_NAME as PADDLEOCR_VL_PAGE_ADAPTER_NAME,
    PaddleOCRVLPageAdapter,
    PaddlePageLoader,
    translate_page_results,
)
from document_services.parser_adapters.pymupdf_probe import PyMuPdfProbeBackend

__all__ = [
    "DOCLING_EXPORT_ADAPTER_NAME",
    "DoclingExportAdapter",
    "DoclingExportLoader",
    "MINERU_NORMALIZED_ADAPTER_NAME",
    "MinerUNormalizedAdapter",
    "NormalizedBlockLoader",
    "PADDLEOCR_VL_PAGE_ADAPTER_NAME",
    "PaddleOCRVLPageAdapter",
    "PaddlePageLoader",
    "PyMuPdfProbeBackend",
    "translate_docling_export",
    "translate_normalized_blocks",
    "translate_page_results",
]
