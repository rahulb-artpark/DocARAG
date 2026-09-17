"""Bounded, fail-closed verification of canonical local asset bytes.

``AssetRef`` values are references, not proof that the bytes currently found at
their URI are canonical evidence.  This module provides the deliberately
opt-in boundary that turns a local reference into a *read snapshot* verified
against its declared SHA-256 digest.

Only absolute local paths beneath explicitly allowlisted roots, ``file:`` URIs
with the same restriction, and configured content-addressed references are
accepted.  Files are opened component-by-component relative to a bound root
directory descriptor.  Symbolic links, path traversal, non-regular files,
oversized batches, replacement races, and mutation during streaming reads all
fail closed.

The default result retains no bytes and therefore uses O(chunk_size) working
memory.  ``include_bytes=True`` returns an immutable byte snapshot, bounded by
``max_asset_bytes`` and ``max_total_bytes``.  A returned filesystem path is a
point-in-time audit location only; consumers that need immunity from a change
after verification must consume ``data`` or copy it into an immutable,
content-addressed store before trusting it.

The limits in this module apply to encoded file bytes only.  ``media_type`` is
an unverified declaration copied from ``AssetRef``: this boundary does not
sniff MIME types, invoke an image/document decoder, or bound decoded pixel
dimensions.  A consumer that decodes verified bytes must enforce separate
format, dimension, decompression-memory, and decoder-isolation limits.  In
particular, a small compressed file is not proof of a small decoded image.

This module intentionally does not change ``CanonicalEvidenceHydrator``'s
``reference_only`` default.  Promoting an answer citation to byte-verified
requires an explicit consumer that can use the verified snapshot without
reopening an attacker-mutable path.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
import errno
import hashlib
import hmac
import os
from pathlib import Path
import re
import stat as stat_module
from typing import NoReturn
from urllib.parse import unquote_to_bytes, urlsplit

from document_services.domain import AssetRef


ASSET_MATERIAL_SCHEMA_VERSION = "docarag.asset-material.v1"
ASSET_MATERIAL_VERIFIED = "verified"
MAX_ALLOWLIST_ROOTS = 64
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_CAS_URI_PATTERN = re.compile(r"cas://sha256/([0-9a-f]{64})")
_SHA256_URI_PATTERN = re.compile(r"sha256:([0-9a-f]{64})")


class AssetMaterialErrorCode(str, Enum):
    """Stable classifications suitable for logs, metrics, and retry policy."""

    INVALID_CONFIGURATION = "invalid_configuration"
    INVALID_ASSET = "invalid_asset"
    UNSUPPORTED_URI = "unsupported_uri"
    DIGEST_REQUIRED = "digest_required"
    DIGEST_MISMATCH = "digest_mismatch"
    PATH_ESCAPE = "path_escape"
    SYMBOLIC_LINK = "symbolic_link"
    NOT_FOUND = "not_found"
    NOT_REGULAR = "not_regular"
    TOO_LARGE = "too_large"
    BATCH_LIMIT = "batch_limit"
    TARGET_CONFLICT = "target_conflict"
    CHANGED_DURING_READ = "changed_during_read"
    IO_ERROR = "io_error"


class AssetMaterializationError(ValueError):
    """An asset could not be proven to match its canonical reference."""

    def __init__(
        self,
        code: AssetMaterialErrorCode,
        message: str,
        *,
        asset_id: str | None = None,
        asset_uri: str | None = None,
    ) -> None:
        self.code = AssetMaterialErrorCode(code)
        self.asset_id = asset_id
        self.asset_uri = asset_uri
        suffix = f" [asset_id={asset_id}]" if asset_id else ""
        super().__init__(f"{message}{suffix}")


def _fail(
    code: AssetMaterialErrorCode,
    message: str,
    *,
    asset: AssetRef | None = None,
) -> NoReturn:
    raise AssetMaterializationError(
        code,
        message,
        asset_id=asset.asset_id if asset is not None else None,
        asset_uri=asset.uri if asset is not None else None,
    )


def _positive_integer(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        _fail(
            AssetMaterialErrorCode.INVALID_CONFIGURATION,
            f"{field_name} must be a positive integer",
        )
    return value


def _stable_stat(left: os.stat_result, right: os.stat_result) -> bool:
    fields = (
        "st_dev",
        "st_ino",
        "st_mode",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    return all(getattr(left, name) == getattr(right, name) for name in fields)


@dataclass(frozen=True, slots=True)
class VerifiedAssetMaterial:
    """One bounded byte-read whose digest matched its canonical ``AssetRef``.

    ``path`` records where the verified descriptor was opened.  It is not a
    promise that a later open of that pathname returns the same inode.  ``data``
    is populated only when the caller explicitly requests a bounded immutable
    snapshot.
    """

    asset_id: str
    asset_uri: str
    media_type: str | None
    sha256: str
    size_bytes: int
    path: Path
    reference_kind: str
    data: bytes | None = field(default=None, repr=False)
    status: str = field(default=ASSET_MATERIAL_VERIFIED, init=False)
    schema_version: str = field(default=ASSET_MATERIAL_SCHEMA_VERSION, init=False)

    def to_dict(self) -> dict[str, object]:
        """Return JSON-safe audit metadata without duplicating byte material.

        The validation-boundary fields prevent encoded-byte verification from
        being mistaken for MIME sniffing or safe decoded-image validation.
        """

        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "asset_id": self.asset_id,
            "asset_uri": self.asset_uri,
            "media_type": self.media_type,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "verified_path": os.fspath(self.path),
            "reference_kind": self.reference_kind,
            "bytes_retained": self.data is not None,
            "verification_scope": "descriptor_read_snapshot",
            "byte_limit_scope": "encoded_file_bytes_only",
            "media_type_source": "asset_ref_declaration_unverified",
            "decoded_content_validation": "not_performed",
        }


@dataclass(frozen=True, slots=True)
class _RootBinding:
    path: Path
    device: int
    inode: int


@dataclass(frozen=True, slots=True)
class _AssetTarget:
    asset: AssetRef
    root: _RootBinding
    parts: tuple[str, ...]
    path: Path
    reference_kind: str

    @property
    def key(self) -> tuple[str, tuple[str, ...]]:
        return (os.fspath(self.root.path), self.parts)


@dataclass(frozen=True, slots=True)
class _VerifiedRead:
    sha256: str
    size_bytes: int
    path: Path
    data: bytes | None


class AssetMaterialResolver:
    """Resolve and stream-verify a bounded batch of canonical local assets.

    Content-addressed references use either ``sha256:<digest>`` or
    ``cas://sha256/<digest>``.  With the default ``prefix2`` layout, those map
    to ``<content_addressed_root>/<first-two-hex>/<digest>``; ``flat`` maps to
    ``<content_addressed_root>/<digest>``.
    """

    __slots__ = (
        "_allowed_roots",
        "_content_root",
        "content_layout",
        "max_assets",
        "max_asset_bytes",
        "max_total_bytes",
        "chunk_size",
    )

    def __init__(
        self,
        *,
        allowed_roots: Sequence[str | os.PathLike[str]] = (),
        content_addressed_root: str | os.PathLike[str] | None = None,
        content_layout: str = "prefix2",
        max_assets: int = 64,
        max_asset_bytes: int = 32 * 1024 * 1024,
        max_total_bytes: int = 128 * 1024 * 1024,
        chunk_size: int = 1024 * 1024,
    ) -> None:
        if isinstance(allowed_roots, (str, bytes, bytearray, os.PathLike)):
            _fail(
                AssetMaterialErrorCode.INVALID_CONFIGURATION,
                "allowed_roots must be an array of directory paths",
            )
        try:
            raw_roots = tuple(allowed_roots)
        except TypeError as exc:
            raise AssetMaterializationError(
                AssetMaterialErrorCode.INVALID_CONFIGURATION,
                "allowed_roots must be an array of directory paths",
            ) from exc
        if len(raw_roots) > MAX_ALLOWLIST_ROOTS:
            _fail(
                AssetMaterialErrorCode.INVALID_CONFIGURATION,
                f"allowed_roots exceeds the {MAX_ALLOWLIST_ROOTS}-root limit",
            )
        if content_layout not in {"prefix2", "flat"}:
            _fail(
                AssetMaterialErrorCode.INVALID_CONFIGURATION,
                "content_layout must be 'prefix2' or 'flat'",
            )
        self.max_assets = _positive_integer(max_assets, "max_assets")
        self.max_asset_bytes = _positive_integer(
            max_asset_bytes, "max_asset_bytes"
        )
        self.max_total_bytes = _positive_integer(
            max_total_bytes, "max_total_bytes"
        )
        self.chunk_size = _positive_integer(chunk_size, "chunk_size")
        self.content_layout = content_layout

        if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
            _fail(
                AssetMaterialErrorCode.INVALID_CONFIGURATION,
                "this platform cannot provide no-follow directory traversal",
            )
        if os.open not in os.supports_dir_fd or os.stat not in os.supports_dir_fd:
            _fail(
                AssetMaterialErrorCode.INVALID_CONFIGURATION,
                "this platform cannot provide descriptor-relative traversal",
            )

        roots: list[_RootBinding] = []
        seen_paths: set[Path] = set()
        for raw_root in raw_roots:
            binding = self._bind_root(raw_root, field_name="allowed_roots[]")
            if binding.path not in seen_paths:
                roots.append(binding)
                seen_paths.add(binding.path)
        roots.sort(key=lambda value: len(value.path.parts), reverse=True)

        content_root = None
        if content_addressed_root is not None:
            content_root = self._bind_root(
                content_addressed_root,
                field_name="content_addressed_root",
            )
        if not roots and content_root is None:
            _fail(
                AssetMaterialErrorCode.INVALID_CONFIGURATION,
                "at least one allowlisted or content-addressed root is required",
            )
        self._allowed_roots = tuple(roots)
        self._content_root = content_root

    @staticmethod
    def _bind_root(
        value: str | os.PathLike[str], *, field_name: str
    ) -> _RootBinding:
        try:
            raw = Path(os.fspath(value)).expanduser()
        except (TypeError, ValueError) as exc:
            raise AssetMaterializationError(
                AssetMaterialErrorCode.INVALID_CONFIGURATION,
                f"{field_name} is not a valid filesystem path",
            ) from exc
        if not raw.is_absolute():
            _fail(
                AssetMaterialErrorCode.INVALID_CONFIGURATION,
                f"{field_name} must be absolute",
            )
        absolute = Path(os.path.abspath(os.fspath(raw)))
        try:
            resolved = absolute.resolve(strict=True)
            metadata = os.lstat(absolute)
        except OSError as exc:
            raise AssetMaterializationError(
                AssetMaterialErrorCode.INVALID_CONFIGURATION,
                f"{field_name} is unavailable",
            ) from exc
        # An allowlist should name the actual directory, not an alias whose
        # target can be changed independently.
        if resolved != absolute or stat_module.S_ISLNK(metadata.st_mode):
            _fail(
                AssetMaterialErrorCode.INVALID_CONFIGURATION,
                f"{field_name} must not contain symbolic-link components",
            )
        if not stat_module.S_ISDIR(metadata.st_mode):
            _fail(
                AssetMaterialErrorCode.INVALID_CONFIGURATION,
                f"{field_name} must name a directory",
            )
        return _RootBinding(
            path=absolute,
            device=metadata.st_dev,
            inode=metadata.st_ino,
        )

    def resolve(
        self, asset: AssetRef, *, include_bytes: bool = False
    ) -> VerifiedAssetMaterial:
        """Verify one asset through the same batch-bounded implementation."""

        return self.resolve_many((asset,), include_bytes=include_bytes)[0]

    def resolve_many(
        self,
        assets: Sequence[AssetRef],
        *,
        include_bytes: bool = False,
    ) -> tuple[VerifiedAssetMaterial, ...]:
        """Verify assets in input order with bounded total I/O and memory.

        Repeated references to the same target and digest are read once.  The
        input-count limit still applies, preventing an unbounded output tuple.
        Conflicting digest declarations for one target fail before any asset is
        opened.
        """

        if not isinstance(include_bytes, bool):
            _fail(
                AssetMaterialErrorCode.INVALID_CONFIGURATION,
                "include_bytes must be a boolean",
            )
        if not isinstance(assets, Sequence) or isinstance(
            assets, (str, bytes, bytearray)
        ):
            _fail(
                AssetMaterialErrorCode.INVALID_ASSET,
                "assets must be an array of AssetRef values",
            )
        if len(assets) > self.max_assets:
            _fail(
                AssetMaterialErrorCode.BATCH_LIMIT,
                f"asset batch exceeds max_assets ({len(assets)} > {self.max_assets})",
            )
        if not assets:
            return ()

        targets: list[_AssetTarget] = []
        digest_by_target: dict[tuple[str, tuple[str, ...]], str] = {}
        unique_targets: dict[tuple[str, tuple[str, ...]], _AssetTarget] = {}
        for position, asset in enumerate(assets):
            if not isinstance(asset, AssetRef):
                _fail(
                    AssetMaterialErrorCode.INVALID_ASSET,
                    f"assets[{position}] must be an AssetRef",
                )
            target = self._target_for(asset)
            expected_digest = asset.sha256
            assert expected_digest is not None  # checked by _target_for
            previous_digest = digest_by_target.get(target.key)
            if previous_digest is not None and previous_digest != expected_digest:
                _fail(
                    AssetMaterialErrorCode.TARGET_CONFLICT,
                    "one asset target has conflicting declared digests",
                    asset=asset,
                )
            digest_by_target[target.key] = expected_digest
            unique_targets.setdefault(target.key, target)
            targets.append(target)

        reads: dict[tuple[str, tuple[str, ...]], _VerifiedRead] = {}
        total_bytes = 0
        for key, target in unique_targets.items():
            remaining = self.max_total_bytes - total_bytes
            if remaining < 1:
                _fail(
                    AssetMaterialErrorCode.BATCH_LIMIT,
                    "asset batch exceeds max_total_bytes",
                    asset=target.asset,
                )
            verified = self._read_target(
                target,
                max_read_bytes=min(self.max_asset_bytes, remaining),
                include_bytes=include_bytes,
                total_limit_is_tighter=remaining < self.max_asset_bytes,
            )
            reads[key] = verified
            total_bytes += verified.size_bytes

        return tuple(
            VerifiedAssetMaterial(
                asset_id=target.asset.asset_id,
                asset_uri=target.asset.uri,
                media_type=target.asset.media_type,
                sha256=reads[target.key].sha256,
                size_bytes=reads[target.key].size_bytes,
                path=reads[target.key].path,
                reference_kind=target.reference_kind,
                data=reads[target.key].data,
            )
            for target in targets
        )

    def _target_for(self, asset: AssetRef) -> _AssetTarget:
        declared_digest = asset.sha256
        if declared_digest is None:
            _fail(
                AssetMaterialErrorCode.DIGEST_REQUIRED,
                "asset.sha256 is required before byte materialization",
                asset=asset,
            )
        if _SHA256_PATTERN.fullmatch(declared_digest) is None:
            # AssetRef currently enforces this itself, but retain the boundary
            # check if the domain object is ever deserialized by another layer.
            _fail(
                AssetMaterialErrorCode.INVALID_ASSET,
                "asset.sha256 must be a lowercase SHA-256 digest",
                asset=asset,
            )

        cas_match = _CAS_URI_PATTERN.fullmatch(asset.uri)
        if cas_match is None:
            cas_match = _SHA256_URI_PATTERN.fullmatch(asset.uri)
        if cas_match is not None:
            uri_digest = cas_match.group(1)
            if not hmac.compare_digest(uri_digest, declared_digest):
                _fail(
                    AssetMaterialErrorCode.DIGEST_MISMATCH,
                    "content-addressed URI digest differs from asset.sha256",
                    asset=asset,
                )
            if self._content_root is None:
                _fail(
                    AssetMaterialErrorCode.UNSUPPORTED_URI,
                    "content-addressed URI requires content_addressed_root",
                    asset=asset,
                )
            parts = (
                (uri_digest[:2], uri_digest)
                if self.content_layout == "prefix2"
                else (uri_digest,)
            )
            return _AssetTarget(
                asset=asset,
                root=self._content_root,
                parts=parts,
                path=self._content_root.path.joinpath(*parts),
                reference_kind="content_addressed",
            )

        path_text = self._local_path_from_uri(asset)
        # Reject traversal lexically rather than silently normalizing it into an
        # allowlisted path.
        raw_segments = path_text.replace("\\", "/").split("/")
        if any(segment in {".", ".."} for segment in raw_segments):
            _fail(
                AssetMaterialErrorCode.PATH_ESCAPE,
                "asset URI contains a traversal component",
                asset=asset,
            )
        path = Path(path_text)
        if not path.is_absolute():
            _fail(
                AssetMaterialErrorCode.UNSUPPORTED_URI,
                "local asset URI must be an absolute path",
                asset=asset,
            )
        path = Path(os.path.abspath(os.fspath(path)))
        for root in self._allowed_roots:
            try:
                parts = path.relative_to(root.path).parts
            except ValueError:
                continue
            if not parts:
                _fail(
                    AssetMaterialErrorCode.NOT_REGULAR,
                    "asset URI names an allowlisted directory, not a file",
                    asset=asset,
                )
            if any(part in {"", ".", ".."} or "/" in part for part in parts):
                _fail(
                    AssetMaterialErrorCode.PATH_ESCAPE,
                    "asset URI escapes its allowlisted root",
                    asset=asset,
                )
            return _AssetTarget(
                asset=asset,
                root=root,
                parts=tuple(parts),
                path=path,
                reference_kind="local_file",
            )
        _fail(
            AssetMaterialErrorCode.PATH_ESCAPE,
            "asset URI is outside every allowlisted root",
            asset=asset,
        )

    @staticmethod
    def _local_path_from_uri(asset: AssetRef) -> str:
        uri = asset.uri
        if "\x00" in uri:
            _fail(
                AssetMaterialErrorCode.INVALID_ASSET,
                "asset URI contains a NUL byte",
                asset=asset,
            )
        try:
            split = urlsplit(uri)
        except ValueError as exc:
            raise AssetMaterializationError(
                AssetMaterialErrorCode.INVALID_ASSET,
                "asset URI is malformed",
                asset_id=asset.asset_id,
                asset_uri=asset.uri,
            ) from exc
        if uri.startswith("file:"):
            if (
                split.scheme != "file"
                or split.netloc
                or split.query
                or split.fragment
                or not split.path
            ):
                _fail(
                    AssetMaterialErrorCode.UNSUPPORTED_URI,
                    "file URI must be an absolute local URI without authority, query, or fragment",
                    asset=asset,
                )
            try:
                path = unquote_to_bytes(split.path).decode("utf-8", errors="strict")
            except UnicodeDecodeError as exc:
                raise AssetMaterializationError(
                    AssetMaterialErrorCode.INVALID_ASSET,
                    "file URI path is not valid UTF-8",
                    asset_id=asset.asset_id,
                    asset_uri=asset.uri,
                ) from exc
            if "\x00" in path:
                _fail(
                    AssetMaterialErrorCode.INVALID_ASSET,
                    "file URI path contains a NUL byte",
                    asset=asset,
                )
            return path
        if split.scheme:
            _fail(
                AssetMaterialErrorCode.UNSUPPORTED_URI,
                f"unsupported asset URI scheme: {split.scheme}",
                asset=asset,
            )
        return uri

    def _read_target(
        self,
        target: _AssetTarget,
        *,
        max_read_bytes: int,
        include_bytes: bool,
        total_limit_is_tighter: bool,
    ) -> _VerifiedRead:
        root_flags = (
            os.O_RDONLY
            | os.O_DIRECTORY
            | os.O_NOFOLLOW
            | getattr(os, "O_CLOEXEC", 0)
        )
        file_flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
        directory_fds: list[int] = []
        directory_entries: list[tuple[int, str, os.stat_result]] = []
        file_descriptor: int | None = None
        parent_fd: int | None = None
        final_name = target.parts[-1]
        try:
            try:
                root_fd = os.open(target.root.path, root_flags)
            except OSError as exc:
                self._raise_os_error(exc, target, component="allowlisted root")
            directory_fds.append(root_fd)
            root_stat = self._descriptor_stat(
                root_fd, target, component="allowlisted root"
            )
            if (
                not stat_module.S_ISDIR(root_stat.st_mode)
                or root_stat.st_dev != target.root.device
                or root_stat.st_ino != target.root.inode
            ):
                _fail(
                    AssetMaterialErrorCode.CHANGED_DURING_READ,
                    "allowlisted root binding changed",
                    asset=target.asset,
                )

            parent_fd = root_fd
            for part in target.parts[:-1]:
                before = self._entry_stat(parent_fd, part, target)
                if stat_module.S_ISLNK(before.st_mode):
                    _fail(
                        AssetMaterialErrorCode.SYMBOLIC_LINK,
                        "asset path contains a symbolic-link directory",
                        asset=target.asset,
                    )
                if not stat_module.S_ISDIR(before.st_mode):
                    _fail(
                        AssetMaterialErrorCode.NOT_REGULAR,
                        "asset path component is not a directory",
                        asset=target.asset,
                    )
                try:
                    next_fd = os.open(part, root_flags, dir_fd=parent_fd)
                except OSError as exc:
                    self._raise_os_error(exc, target, component=part)
                # Register ownership immediately after a successful open.  In
                # particular, a subsequent fstat failure must not leak this
                # intermediate directory descriptor.
                directory_fds.append(next_fd)
                after = self._descriptor_stat(next_fd, target, component=part)
                if (
                    before.st_dev != after.st_dev
                    or before.st_ino != after.st_ino
                    or not stat_module.S_ISDIR(after.st_mode)
                ):
                    _fail(
                        AssetMaterialErrorCode.CHANGED_DURING_READ,
                        "asset directory changed during traversal",
                        asset=target.asset,
                    )
                directory_entries.append((parent_fd, part, after))
                parent_fd = next_fd

            before = self._entry_stat(parent_fd, final_name, target)
            if stat_module.S_ISLNK(before.st_mode):
                _fail(
                    AssetMaterialErrorCode.SYMBOLIC_LINK,
                    "asset path is a symbolic link",
                    asset=target.asset,
                )
            if not stat_module.S_ISREG(before.st_mode):
                _fail(
                    AssetMaterialErrorCode.NOT_REGULAR,
                    "asset path is not a regular file",
                    asset=target.asset,
                )
            try:
                file_descriptor = os.open(final_name, file_flags, dir_fd=parent_fd)
            except OSError as exc:
                self._raise_os_error(exc, target, component=final_name)
            opened = self._descriptor_stat(
                file_descriptor, target, component=final_name
            )
            if (
                before.st_dev != opened.st_dev
                or before.st_ino != opened.st_ino
                or not stat_module.S_ISREG(opened.st_mode)
            ):
                _fail(
                    AssetMaterialErrorCode.CHANGED_DURING_READ,
                    "asset changed while it was being opened",
                    asset=target.asset,
                )
            if opened.st_size > self.max_asset_bytes:
                _fail(
                    AssetMaterialErrorCode.TOO_LARGE,
                    "asset exceeds max_asset_bytes "
                    f"({opened.st_size} > {self.max_asset_bytes})",
                    asset=target.asset,
                )
            if opened.st_size > max_read_bytes:
                code = (
                    AssetMaterialErrorCode.BATCH_LIMIT
                    if total_limit_is_tighter
                    else AssetMaterialErrorCode.TOO_LARGE
                )
                limit_name = (
                    "remaining max_total_bytes"
                    if total_limit_is_tighter
                    else "max_asset_bytes"
                )
                _fail(
                    code,
                    f"asset exceeds {limit_name} ({opened.st_size} > {max_read_bytes})",
                    asset=target.asset,
                )

            digest = hashlib.sha256()
            bytes_read = 0
            snapshot: bytes | bytearray | None = None
            if include_bytes:
                # A regular-file read normally fills this request.  In that
                # common case the returned object is already the immutable
                # snapshot, avoiding a chunk list plus a full-size join copy.
                # A POSIX short read is legal, so use a pre-sized, hard-bounded
                # bytearray only as a fallback and continue in chunk-sized
                # increments.
                expected_size = opened.st_size
                try:
                    initial = os.read(file_descriptor, expected_size)
                except OSError as exc:
                    self._raise_os_error(exc, target, component=final_name)
                bytes_read = len(initial)
                digest.update(initial)
                if bytes_read == expected_size:
                    snapshot = initial
                else:
                    mutable_snapshot = bytearray(expected_size)
                    mutable_snapshot[:bytes_read] = initial
                    del initial
                    while bytes_read < expected_size:
                        try:
                            chunk = os.read(
                                file_descriptor,
                                min(self.chunk_size, expected_size - bytes_read),
                            )
                        except OSError as exc:
                            self._raise_os_error(
                                exc, target, component=final_name
                            )
                        if not chunk:
                            break
                        end = bytes_read + len(chunk)
                        mutable_snapshot[bytes_read:end] = chunk
                        bytes_read = end
                        digest.update(chunk)
                    snapshot = mutable_snapshot

                # The opened size was checked against both hard limits before
                # allocation.  A one-byte sentinel detects growth without an
                # unbounded read or retaining bytes beyond the configured cap.
                if bytes_read == expected_size:
                    try:
                        sentinel = os.read(file_descriptor, 1)
                    except OSError as exc:
                        self._raise_os_error(
                            exc, target, component=final_name
                        )
                    if sentinel:
                        bytes_read += len(sentinel)
                        if bytes_read > max_read_bytes:
                            _fail(
                                AssetMaterialErrorCode.TOO_LARGE,
                                "asset grew beyond its configured read bound",
                                asset=target.asset,
                            )
                        _fail(
                            AssetMaterialErrorCode.CHANGED_DURING_READ,
                            "asset grew while bytes were being read",
                            asset=target.asset,
                        )
            else:
                while True:
                    remaining_with_sentinel = max_read_bytes + 1 - bytes_read
                    if remaining_with_sentinel <= 0:
                        _fail(
                            AssetMaterialErrorCode.TOO_LARGE,
                            "asset grew beyond its configured read bound",
                            asset=target.asset,
                        )
                    try:
                        chunk = os.read(
                            file_descriptor,
                            min(self.chunk_size, remaining_with_sentinel),
                        )
                    except OSError as exc:
                        self._raise_os_error(exc, target, component=final_name)
                    if not chunk:
                        break
                    bytes_read += len(chunk)
                    if bytes_read > max_read_bytes:
                        _fail(
                            AssetMaterialErrorCode.TOO_LARGE,
                            "asset grew beyond its configured read bound",
                            asset=target.asset,
                        )
                    digest.update(chunk)

            after = self._descriptor_stat(
                file_descriptor, target, component=final_name
            )
            if bytes_read != after.st_size or not _stable_stat(opened, after):
                _fail(
                    AssetMaterialErrorCode.CHANGED_DURING_READ,
                    "asset changed while bytes were being read",
                    asset=target.asset,
                )

            # Verify every held namespace binding at the end of the read.  This
            # catches a rename-and-replace even when the opened inode itself did
            # not change.
            self._verify_root_binding(target)
            for entry_parent, entry_name, expected in directory_entries:
                current = self._entry_stat(entry_parent, entry_name, target)
                if (
                    current.st_dev != expected.st_dev
                    or current.st_ino != expected.st_ino
                    or not stat_module.S_ISDIR(current.st_mode)
                ):
                    _fail(
                        AssetMaterialErrorCode.CHANGED_DURING_READ,
                        "asset directory binding changed during read",
                        asset=target.asset,
                    )
            current = self._entry_stat(parent_fd, final_name, target)
            if (
                not _stable_stat(after, current)
                or stat_module.S_ISLNK(current.st_mode)
            ):
                _fail(
                    AssetMaterialErrorCode.CHANGED_DURING_READ,
                    "asset pathname was replaced during read",
                    asset=target.asset,
                )

            actual_digest = digest.hexdigest()
            expected_digest = target.asset.sha256
            assert expected_digest is not None
            if not hmac.compare_digest(actual_digest, expected_digest):
                _fail(
                    AssetMaterialErrorCode.DIGEST_MISMATCH,
                    "asset bytes differ from the declared SHA-256 digest",
                    asset=target.asset,
                )
            data = (
                bytes(snapshot)
                if isinstance(snapshot, bytearray)
                else snapshot
            )
            return _VerifiedRead(
                sha256=actual_digest,
                size_bytes=bytes_read,
                path=target.path,
                data=data,
            )
        finally:
            if file_descriptor is not None:
                os.close(file_descriptor)
            for descriptor in reversed(directory_fds):
                os.close(descriptor)

    @staticmethod
    def _descriptor_stat(
        descriptor: int, target: _AssetTarget, *, component: str
    ) -> os.stat_result:
        try:
            return os.fstat(descriptor)
        except OSError as exc:
            AssetMaterialResolver._raise_os_error(
                exc, target, component=component
            )

    @staticmethod
    def _entry_stat(
        parent_fd: int, name: str, target: _AssetTarget
    ) -> os.stat_result:
        try:
            return os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        except OSError as exc:
            AssetMaterialResolver._raise_os_error(
                exc, target, component=name
            )

    @staticmethod
    def _raise_os_error(
        exc: OSError, target: _AssetTarget, *, component: str
    ) -> NoReturn:
        if exc.errno in {errno.ENOENT}:
            code = AssetMaterialErrorCode.NOT_FOUND
            message = f"asset path component is unavailable: {component}"
        elif exc.errno in {errno.ELOOP}:
            code = AssetMaterialErrorCode.SYMBOLIC_LINK
            message = f"asset path component is a symbolic link: {component}"
        elif exc.errno in {errno.ENOTDIR}:
            code = AssetMaterialErrorCode.NOT_REGULAR
            message = f"asset path component is not a directory: {component}"
        else:
            code = AssetMaterialErrorCode.IO_ERROR
            message = f"asset path could not be read safely: {component}"
        raise AssetMaterializationError(
            code,
            message,
            asset_id=target.asset.asset_id,
            asset_uri=target.asset.uri,
        ) from exc

    @staticmethod
    def _verify_root_binding(target: _AssetTarget) -> None:
        try:
            current = os.stat(target.root.path, follow_symlinks=False)
        except OSError as exc:
            raise AssetMaterializationError(
                AssetMaterialErrorCode.CHANGED_DURING_READ,
                "allowlisted root path became unavailable during read",
                asset_id=target.asset.asset_id,
                asset_uri=target.asset.uri,
            ) from exc
        if (
            current.st_dev != target.root.device
            or current.st_ino != target.root.inode
            or not stat_module.S_ISDIR(current.st_mode)
            or stat_module.S_ISLNK(current.st_mode)
        ):
            _fail(
                AssetMaterialErrorCode.CHANGED_DURING_READ,
                "allowlisted root path was replaced during read",
                asset=target.asset,
            )


__all__ = [
    "ASSET_MATERIAL_SCHEMA_VERSION",
    "ASSET_MATERIAL_VERIFIED",
    "AssetMaterialErrorCode",
    "AssetMaterialResolver",
    "AssetMaterializationError",
    "VerifiedAssetMaterial",
]
