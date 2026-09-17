"""Dependency-free application ports."""

from document_services.ports.page_probe import PageProbe, PageRange
from document_services.ports.parser import (
    ParserAdapter,
    ParserAlreadyRegisteredError,
    ParserNotRegisteredError,
    ParserRegistry,
    ParserRegistryError,
    UnsupportedParserRequestError,
    validate_result_binding,
)

__all__ = [
    "PageProbe",
    "PageRange",
    "ParserAdapter",
    "ParserAlreadyRegisteredError",
    "ParserNotRegisteredError",
    "ParserRegistry",
    "ParserRegistryError",
    "UnsupportedParserRequestError",
    "validate_result_binding",
]
