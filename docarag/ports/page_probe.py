"""Dependency-light port for streaming, source-bound page probes."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from os import PathLike
from typing import Protocol, TypeVar, runtime_checkable

from docarag.domain.base import ContractValidationError
from docarag.domain.evidence import DocumentRef


FeatureT_co = TypeVar("FeatureT_co", covariant=True)


@dataclass(frozen=True, slots=True)
class PageRange:
    """A zero-based, half-open page range ``[start_index, stop_index)``."""

    start_index: int
    stop_index: int

    def __post_init__(self) -> None:
        for value, field_name in (
            (self.start_index, "start_index"),
            (self.stop_index, "stop_index"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ContractValidationError(
                    f"PageRange.{field_name} must be a non-negative integer"
                )
        if self.stop_index <= self.start_index:
            raise ContractValidationError(
                "PageRange.stop_index must be greater than start_index"
            )


@runtime_checkable
class PageProbe(Protocol[FeatureT_co]):
    """Stream cheap page features without parsing or rendering the document.

    Implementations must bind every result to ``document.document_version_id``.
    ``page_indices`` and ``page_ranges`` are mutually exclusive.  An omitted
    selection means every page, but results must still be produced lazily.
    """

    def probe(
        self,
        document: DocumentRef,
        *,
        source_path: str | PathLike[str] | None = None,
        page_indices: Sequence[int] | None = None,
        page_ranges: Sequence[PageRange] | None = None,
    ) -> Iterator[FeatureT_co]:
        ...


__all__ = ["PageProbe", "PageRange"]
