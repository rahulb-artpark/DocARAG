"""Dependency-inverted parser port and explicit capability registry."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol, runtime_checkable

from document_services.domain.base import ContractValidationError
from document_services.domain.evidence import BlockType
from document_services.domain.parsing import (
    ParseRequest,
    ParseResult,
    ParserCapabilityManifest,
    ParserModality,
)


class ParserRegistryError(LookupError):
    """Base class for explicit adapter registration and resolution failures."""


class ParserAlreadyRegisteredError(ParserRegistryError):
    pass


class ParserNotRegisteredError(ParserRegistryError):
    pass


class UnsupportedParserRequestError(ParserRegistryError):
    pass


@runtime_checkable
class ParserAdapter(Protocol):
    """Small parser boundary suitable for an in-process or RPC adapter.

    Implementations own vendor imports.  Domain/application code depends only
    on this protocol and receives the same validated ParseResult either way.
    """

    def manifest(self) -> ParserCapabilityManifest:
        ...

    def parse(self, request: ParseRequest) -> ParseResult:
        ...


def validate_result_binding(
    manifest: ParserCapabilityManifest,
    request: ParseRequest,
    result: ParseResult,
) -> None:
    """Validate adapter identity and source bindings at the trust boundary."""

    if result.request_id != request.request_id:
        raise ContractValidationError("adapter result is bound to a different request")
    if result.adapter_name != manifest.adapter_name:
        raise ContractValidationError("adapter result name differs from its manifest")
    if result.adapter_version != manifest.adapter_version:
        raise ContractValidationError("adapter result version differs from its manifest")
    for artifact in result.artifacts:
        if artifact.document_version_id != request.document.document_version_id:
            raise ContractValidationError(
                "adapter artifact receipt is bound to a different document version"
            )
        if artifact.source_sha256 != request.document.content_sha256:
            raise ContractValidationError(
                "adapter artifact receipt is bound to a different source digest"
            )
    pages = request.page_by_index
    for observation in result.observations:
        if observation.document_version_id != request.document.document_version_id:
            raise ContractValidationError(
                "adapter observation is bound to a different document version"
            )
        page = pages.get(observation.page_index)
        if page is None or observation.page_id != page.page_id:
            raise ContractValidationError(
                "adapter observation is bound to a page outside the request"
            )
        if observation.provenance.parser_name != manifest.parser_name:
            raise ContractValidationError(
                "observation parser name differs from adapter manifest"
            )
        if observation.provenance.parser_version != manifest.parser_version:
            raise ContractValidationError(
                "observation parser version differs from adapter manifest"
            )
        if observation.provenance.model_version != manifest.model_version:
            raise ContractValidationError(
                "observation model version differs from adapter manifest"
            )
        if observation.block_type not in manifest.block_types:
            raise ContractValidationError(
                f"adapter emitted undeclared block type {observation.block_type.value!r}"
            )


class ParserRegistry:
    """Explicit, deterministic adapter registry; no import-time global plugins."""

    def __init__(self, adapters: Iterable[ParserAdapter] = ()) -> None:
        self._adapters: dict[str, ParserAdapter] = {}
        for adapter in adapters:
            self.register(adapter)

    def register(self, adapter: ParserAdapter, *, replace: bool = False) -> None:
        if not isinstance(adapter, ParserAdapter):
            raise TypeError("adapter must implement ParserAdapter")
        manifest = adapter.manifest()
        name = manifest.adapter_name
        if name in self._adapters and not replace:
            raise ParserAlreadyRegisteredError(f"parser adapter already registered: {name}")
        self._adapters[name] = adapter

    def unregister(self, name: str) -> None:
        try:
            del self._adapters[name]
        except KeyError as exc:
            raise ParserNotRegisteredError(f"parser adapter is not registered: {name}") from exc

    def get(self, name: str) -> ParserAdapter:
        try:
            return self._adapters[name]
        except KeyError as exc:
            available = ", ".join(sorted(self._adapters)) or "none"
            raise ParserNotRegisteredError(
                f"parser adapter is not registered: {name}; available: {available}"
            ) from exc

    def manifests(self) -> tuple[ParserCapabilityManifest, ...]:
        return tuple(
            self._adapters[name].manifest() for name in sorted(self._adapters)
        )

    def matching(
        self,
        *,
        mime_type: str,
        modalities: frozenset[ParserModality] = frozenset(),
        block_types: frozenset[BlockType] = frozenset(),
        regions: bool = False,
    ) -> tuple[ParserAdapter, ...]:
        return tuple(
            self._adapters[manifest.adapter_name]
            for manifest in self.manifests()
            if manifest.supports(
                mime_type=mime_type,
                modalities=modalities,
                block_types=block_types,
                regions=regions,
            )
        )

    def parse(self, name: str, request: ParseRequest) -> ParseResult:
        adapter = self.get(name)
        manifest = adapter.manifest()
        if request.document.mime_type not in manifest.supported_mime_types:
            raise UnsupportedParserRequestError(
                f"{name} does not support MIME type {request.document.mime_type}"
            )
        if request.regions and not manifest.supports_region_selection:
            raise UnsupportedParserRequestError(f"{name} does not support region parsing")
        if not manifest.supports_page_selection and request.document.page_count != len(
            request.pages
        ):
            raise UnsupportedParserRequestError(f"{name} requires the complete document")
        if (
            manifest.max_pages_per_request is not None
            and len(request.pages) > manifest.max_pages_per_request
        ):
            raise UnsupportedParserRequestError(
                f"{name} accepts at most {manifest.max_pages_per_request} pages per request"
            )
        result = adapter.parse(request)
        if not isinstance(result, ParseResult):
            raise ContractValidationError("adapter.parse() must return ParseResult")
        validate_result_binding(manifest, request, result)
        return result
