"""Verified source-byte handoff for isolated parser workers.

``DocumentRef.source_uri`` is the logical/original source used for citations.
It is deliberately *not* assumed to be a worker-readable file: it may be an
object URI, an inaccessible user path, or a mutable location.  A routed run
therefore binds a second locator in ``DocumentRef.metadata`` for the immutable
snapshot that workers are allowed to consume.

This module keeps that deployment concern out of the parser DTO and reuses the
same descriptor-relative, no-symlink, bounded hashing boundary used for other
canonical assets.  Missing, malformed, out-of-root, oversized, changed, or
digest-mismatched material fails closed.  There is intentionally no fallback
from a missing material binding to ``source_uri``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import os
from pathlib import Path
from types import MappingProxyType
from typing import NoReturn

from docarag.asset_material import (
    AssetMaterialErrorCode,
    AssetMaterialResolver,
    AssetMaterializationError,
)
from docarag.domain import AssetKind, AssetRef, DocumentRef


SOURCE_MATERIAL_BINDING_KEY = "docarag_source_material"
SOURCE_MATERIAL_BINDING_SCHEMA_VERSION = "docarag.source-material-binding.v1"
SOURCE_MATERIAL_VERIFICATION_SCHEMA_VERSION = "docarag.source-material-verification.v1"
_BINDING_FIELDS = frozenset({"schema_version", "uri", "sha256", "media_type"})


class SourceMaterialError(ValueError):
    """The exact source bytes could not be safely supplied to a parser."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        material_code: AssetMaterialErrorCode | None = None,
    ) -> None:
        self.code = code
        self.material_code = material_code
        super().__init__(message)


def _fail(code: str, message: str) -> NoReturn:
    raise SourceMaterialError(code, message)


def _binding(
    *, uri: str, sha256: str, media_type: str
) -> Mapping[str, object]:
    if not isinstance(uri, str) or not uri:
        _fail("invalid_binding", "source material URI must be non-empty")
    if not isinstance(sha256, str) or len(sha256) != 64 or any(
        character not in "0123456789abcdef" for character in sha256
    ):
        _fail(
            "invalid_binding",
            "source material SHA-256 must be a lowercase 64-character digest",
        )
    if not isinstance(media_type, str) or not media_type:
        _fail("invalid_binding", "source material media type must be non-empty")
    return MappingProxyType(
        {
            "schema_version": SOURCE_MATERIAL_BINDING_SCHEMA_VERSION,
            "uri": uri,
            "sha256": sha256,
            "media_type": media_type,
        }
    )


def local_source_material_binding(
    path: str | os.PathLike[str],
    *,
    sha256: str,
    media_type: str = "application/pdf",
) -> Mapping[str, object]:
    """Create a JSON-safe binding for an absolute local snapshot path.

    The function constructs identity only; the consumer reopens and verifies
    the bytes through :class:`SourceMaterialResolver`.  This separation avoids
    claiming that a pathname remains immutable after an earlier stat/hash.
    """

    try:
        candidate = Path(os.fspath(path)).expanduser()
    except (TypeError, ValueError) as exc:
        raise SourceMaterialError(
            "invalid_binding", "source material path is invalid"
        ) from exc
    if not candidate.is_absolute():
        _fail("invalid_binding", "source material path must be absolute")
    # Do not resolve here: resolution could hide a symbolic-link component.
    absolute = Path(os.path.abspath(os.fspath(candidate)))
    return _binding(uri=absolute.as_uri(), sha256=sha256, media_type=media_type)


def bind_source_material(
    document: DocumentRef,
    binding: Mapping[str, object],
) -> DocumentRef:
    """Return the same logical document identity with one material locator.

    Existing metadata is preserved.  Rebinding an existing locator is refused
    unless it is byte-for-byte equivalent, preventing composition roots from
    silently replacing the source after identity was established.
    """

    if not isinstance(document, DocumentRef):
        raise TypeError("document must be a DocumentRef")
    normalized = _parse_binding(binding, document=document)
    metadata = dict(document.metadata)
    existing = metadata.get(SOURCE_MATERIAL_BINDING_KEY)
    if existing is not None:
        existing_normalized = _parse_binding(existing, document=document)
        if dict(existing_normalized) != dict(normalized):
            _fail(
                "binding_conflict",
                "document already has a different source material binding",
            )
        return document
    metadata[SOURCE_MATERIAL_BINDING_KEY] = dict(normalized)
    return DocumentRef(
        document_id=document.document_id,
        document_version_id=document.document_version_id,
        content_sha256=document.content_sha256,
        mime_type=document.mime_type,
        source_uri=document.source_uri,
        display_name=document.display_name,
        page_count=document.page_count,
        metadata=metadata,
    )


def _parse_binding(
    value: object,
    *,
    document: DocumentRef,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        _fail("invalid_binding", "source material binding must be a JSON object")
    unknown = set(value).difference(_BINDING_FIELDS)
    if unknown:
        _fail(
            "invalid_binding",
            "source material binding has unknown fields: "
            + ", ".join(sorted(unknown)),
        )
    if value.get("schema_version") != SOURCE_MATERIAL_BINDING_SCHEMA_VERSION:
        _fail("invalid_binding", "source material binding schema is unsupported")
    normalized = _binding(
        uri=value.get("uri"),  # type: ignore[arg-type]
        sha256=value.get("sha256"),  # type: ignore[arg-type]
        media_type=value.get("media_type"),  # type: ignore[arg-type]
    )
    if normalized["sha256"] != document.content_sha256:
        _fail(
            "identity_mismatch",
            "source material digest differs from the document version",
        )
    if normalized["media_type"] != document.mime_type:
        _fail(
            "identity_mismatch",
            "source material media type differs from the document version",
        )
    return normalized


@dataclass(frozen=True, slots=True)
class VerifiedSourceMaterial:
    """A bounded descriptor read proven to match one ``DocumentRef``."""

    document_version_id: str
    logical_source_uri: str
    material_uri: str
    media_type: str
    sha256: str
    size_bytes: int
    path: Path
    data: bytes | None = None
    schema_version: str = SOURCE_MATERIAL_VERIFICATION_SCHEMA_VERSION

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "document_version_id": self.document_version_id,
            "logical_source_uri": self.logical_source_uri,
            "material_uri": self.material_uri,
            "media_type": self.media_type,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "verified_path": os.fspath(self.path),
            "bytes_retained": self.data is not None,
            "verification_scope": "descriptor_read_snapshot",
        }


class SourceMaterialResolver:
    """Resolve one explicitly bound local source beneath allow-listed roots."""

    __slots__ = ("_assets",)

    def __init__(
        self,
        *,
        allowed_roots: Sequence[str | os.PathLike[str]],
        max_source_bytes: int = 512 * 1024 * 1024,
        chunk_size: int = 1024 * 1024,
    ) -> None:
        self._assets = AssetMaterialResolver(
            allowed_roots=allowed_roots,
            max_assets=1,
            max_asset_bytes=max_source_bytes,
            max_total_bytes=max_source_bytes,
            chunk_size=chunk_size,
        )

    def resolve(
        self,
        document: DocumentRef,
        *,
        include_bytes: bool = False,
    ) -> VerifiedSourceMaterial:
        if not isinstance(document, DocumentRef):
            raise TypeError("document must be a DocumentRef")
        raw_binding = document.metadata.get(SOURCE_MATERIAL_BINDING_KEY)
        if raw_binding is None:
            _fail(
                "binding_required",
                "document has no verified source material binding; refusing to "
                "treat the logical source URI as worker-readable bytes",
            )
        binding = _parse_binding(raw_binding, document=document)
        asset = AssetRef(
            kind=AssetKind.SOURCE,
            uri=binding["uri"],  # type: ignore[arg-type]
            media_type=binding["media_type"],  # type: ignore[arg-type]
            sha256=binding["sha256"],  # type: ignore[arg-type]
            metadata={"document_version_id": document.document_version_id},
        )
        try:
            material = self._assets.resolve(asset, include_bytes=include_bytes)
        except AssetMaterializationError as exc:
            raise SourceMaterialError(
                "materialization_failed",
                f"source material verification failed: {exc}",
                material_code=exc.code,
            ) from exc
        return VerifiedSourceMaterial(
            document_version_id=document.document_version_id,
            logical_source_uri=document.source_uri,
            material_uri=asset.uri,
            media_type=document.mime_type,
            sha256=material.sha256,
            size_bytes=material.size_bytes,
            path=material.path,
            data=material.data,
        )


__all__ = [
    "SOURCE_MATERIAL_BINDING_KEY",
    "SOURCE_MATERIAL_BINDING_SCHEMA_VERSION",
    "SOURCE_MATERIAL_VERIFICATION_SCHEMA_VERSION",
    "SourceMaterialError",
    "SourceMaterialResolver",
    "VerifiedSourceMaterial",
    "bind_source_material",
    "local_source_material_binding",
]
