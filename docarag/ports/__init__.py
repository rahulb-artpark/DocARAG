"""Dependency-free application ports."""

from .page_probe import PageProbe, PageRange
from .parser import (
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
