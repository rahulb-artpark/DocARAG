"""Dependency-light immutable storage for parser-produced byte artifacts.

The parser services exchange references, while canonical evidence requires exact
bytes.  This module supplies the small boundary between those concerns: bytes
are hashed, bounded, durably published once beneath an explicitly configured
root, and addressed as ``cas://sha256/<digest>``.  That URI is understood by
``AssetMaterialResolver`` when it is configured with the same root.

Publication is idempotent and safe across processes.  A fully written,
``fsync``-ed temporary inode is hard-linked to the final digest name, which is
an atomic no-overwrite operation.  A concurrent winner is accepted only after
its existing bytes are re-read and proven to have the expected digest.  Store
directories are traversed descriptor-relative with ``O_NOFOLLOW``; symbolic
links and conflicting existing targets fail closed.

Vendor-specific callables intentionally remain thin.  MinerU ZIP results are
validated independently before every regular member and the result archive are
published.  PaddleOCR-VL artifacts are already typed byte values and are
adapted directly to the same content-addressed store.  No ML, image, PDF, or
vendor runtime is imported.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from io import BytesIO
import errno
import hashlib
import os
from pathlib import Path, PurePosixPath
import re
import secrets
import stat
from types import MappingProxyType
from typing import NoReturn
import zipfile

from document_services.evidence.assets import AssetMaterialResolver
from document_services.domain import AssetKind, AssetRef
from document_services.parser_clients.mineru import MinerUArchive, PublishedMinerUArchive
from document_services.parser_clients.paddleocr_vl import (
    PaddleOCRVLArtifact,
    StoredPaddleOCRVLArtifact,
)


ARTIFACT_STORE_SCHEMA_VERSION = "docarag.local-cas.v1"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_CAS_PREFIX = "cas://sha256/"


class ArtifactStoreErrorCode(StrEnum):
    """Stable failure classes for logs and service error translation."""

    INVALID_CONFIGURATION = "invalid_configuration"
    INVALID_ARTIFACT = "invalid_artifact"
    DIGEST_MISMATCH = "digest_mismatch"
    TOO_LARGE = "too_large"
    UNSAFE_PATH = "unsafe_path"
    TARGET_CONFLICT = "target_conflict"
    IO_ERROR = "io_error"
    INVALID_ARCHIVE = "invalid_archive"
    ARCHIVE_LIMIT = "archive_limit"


class ArtifactStoreError(ValueError):
    """Artifact bytes could not be safely or immutably published."""

    def __init__(self, code: ArtifactStoreErrorCode, message: str) -> None:
        self.code = ArtifactStoreErrorCode(code)
        super().__init__(message)


def _fail(code: ArtifactStoreErrorCode, message: str) -> NoReturn:
    raise ArtifactStoreError(code, message)


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        _fail(
            ArtifactStoreErrorCode.INVALID_CONFIGURATION,
            f"{name} must be a positive integer",
        )
    return value


def _require_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        _fail(
            ArtifactStoreErrorCode.INVALID_ARTIFACT,
            f"{name} must be a lowercase SHA-256 digest",
        )
    return value


def _stable_stat(left: os.stat_result, right: os.stat_result) -> bool:
    return all(
        getattr(left, field_name) == getattr(right, field_name)
        for field_name in (
            "st_dev",
            "st_ino",
            "st_mode",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        )
    )


@dataclass(frozen=True, slots=True)
class StoredArtifact:
    """Immutable identity returned after successful content publication."""

    uri: str
    sha256: str
    size_bytes: int
    path: Path
    media_type: str | None = None
    schema_version: str = field(default=ARTIFACT_STORE_SCHEMA_VERSION, init=False)

    def to_asset_ref(
        self,
        *,
        kind: AssetKind = AssetKind.EXTRACTED_IMAGE,
        metadata: Mapping[str, object] | None = None,
    ) -> AssetRef:
        """Create a canonical reference resolvable against this store's root."""

        return AssetRef(
            kind=kind,
            uri=self.uri,
            media_type=self.media_type,
            sha256=self.sha256,
            metadata={} if metadata is None else metadata,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "uri": self.uri,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "media_type": self.media_type,
        }


class LocalContentAddressedArtifactStore:
    """Atomic, no-overwrite SHA-256 store rooted at one explicit directory.

    The on-disk layout is ``<root>/<first-two-hex>/<digest>``.  ``create=True``
    creates missing directory components without following symbolic links.
    Every later operation reopens the bound root and verifies its device/inode
    identity before descriptor-relative access.
    """

    __slots__ = (
        "root",
        "max_artifact_bytes",
        "chunk_size",
        "_root_device",
        "_root_inode",
    )

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        create: bool = True,
        max_artifact_bytes: int = 512 * 1024 * 1024,
        chunk_size: int = 1024 * 1024,
    ) -> None:
        if not isinstance(create, bool):
            _fail(
                ArtifactStoreErrorCode.INVALID_CONFIGURATION,
                "create must be a boolean",
            )
        if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
            _fail(
                ArtifactStoreErrorCode.INVALID_CONFIGURATION,
                "this platform cannot provide no-follow directory traversal",
            )
        if (
            os.open not in os.supports_dir_fd
            or os.mkdir not in os.supports_dir_fd
            or os.link not in os.supports_dir_fd
            or os.stat not in os.supports_dir_fd
        ):
            _fail(
                ArtifactStoreErrorCode.INVALID_CONFIGURATION,
                "this platform lacks required descriptor-relative filesystem operations",
            )
        try:
            raw = Path(os.fspath(root)).expanduser()
        except (TypeError, ValueError) as exc:
            raise ArtifactStoreError(
                ArtifactStoreErrorCode.INVALID_CONFIGURATION,
                "artifact root is not a valid filesystem path",
            ) from exc
        if not raw.is_absolute():
            _fail(
                ArtifactStoreErrorCode.INVALID_CONFIGURATION,
                "artifact root must be absolute",
            )
        raw_parts = os.fspath(raw).replace("\\", "/").split("/")
        if any(part in {".", ".."} for part in raw_parts):
            _fail(
                ArtifactStoreErrorCode.UNSAFE_PATH,
                "artifact root must not contain traversal components",
            )
        self.root = Path(os.path.abspath(os.fspath(raw)))
        if self.root == Path(os.path.sep):
            _fail(
                ArtifactStoreErrorCode.INVALID_CONFIGURATION,
                "artifact root may not be the filesystem root",
            )
        self.max_artifact_bytes = _positive_integer(
            max_artifact_bytes, "max_artifact_bytes"
        )
        self.chunk_size = _positive_integer(chunk_size, "chunk_size")
        descriptor = self._create_and_open_root(create=create)
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISDIR(metadata.st_mode):
                _fail(
                    ArtifactStoreErrorCode.INVALID_CONFIGURATION,
                    "artifact root is not a directory",
                )
            self._root_device = metadata.st_dev
            self._root_inode = metadata.st_ino
        finally:
            os.close(descriptor)

    def _create_and_open_root(self, *, create: bool) -> int:
        flags = (
            os.O_RDONLY
            | os.O_DIRECTORY
            | os.O_NOFOLLOW
            | getattr(os, "O_CLOEXEC", 0)
        )
        try:
            descriptor = os.open(os.path.sep, flags)
        except OSError as exc:  # pragma: no cover - unusable host filesystem
            raise ArtifactStoreError(
                ArtifactStoreErrorCode.IO_ERROR,
                "could not open filesystem root",
            ) from exc
        try:
            for component in self.root.parts[1:]:
                try:
                    child = os.open(component, flags, dir_fd=descriptor)
                except FileNotFoundError:
                    if not create:
                        _fail(
                            ArtifactStoreErrorCode.INVALID_CONFIGURATION,
                            "artifact root does not exist",
                        )
                    try:
                        os.mkdir(component, mode=0o700, dir_fd=descriptor)
                    except FileExistsError:
                        # A concurrent creator won.  The no-follow open below
                        # still establishes that it created a real directory.
                        pass
                    except OSError as exc:
                        raise ArtifactStoreError(
                            ArtifactStoreErrorCode.IO_ERROR,
                            "could not create artifact root",
                        ) from exc
                    try:
                        child = os.open(component, flags, dir_fd=descriptor)
                    except OSError as exc:
                        raise ArtifactStoreError(
                            ArtifactStoreErrorCode.UNSAFE_PATH,
                            "artifact root contains an unsafe directory component",
                        ) from exc
                except OSError as exc:
                    code = (
                        ArtifactStoreErrorCode.UNSAFE_PATH
                        if exc.errno in {errno.ELOOP, errno.ENOTDIR}
                        else ArtifactStoreErrorCode.IO_ERROR
                    )
                    raise ArtifactStoreError(
                        code,
                        "artifact root contains an unavailable or unsafe component",
                    ) from exc
                os.close(descriptor)
                descriptor = child
            return descriptor
        except BaseException:
            os.close(descriptor)
            raise

    def _open_bound_root(self) -> int:
        flags = (
            os.O_RDONLY
            | os.O_DIRECTORY
            | os.O_NOFOLLOW
            | getattr(os, "O_CLOEXEC", 0)
        )
        try:
            descriptor = os.open(self.root, flags)
            metadata = os.fstat(descriptor)
        except OSError as exc:
            raise ArtifactStoreError(
                ArtifactStoreErrorCode.UNSAFE_PATH,
                "artifact root is unavailable or was replaced",
            ) from exc
        if (
            metadata.st_dev != self._root_device
            or metadata.st_ino != self._root_inode
            or not stat.S_ISDIR(metadata.st_mode)
        ):
            os.close(descriptor)
            _fail(
                ArtifactStoreErrorCode.UNSAFE_PATH,
                "artifact root identity changed after store initialization",
            )
        return descriptor

    def publish(
        self,
        data: bytes,
        *,
        expected_sha256: str | None = None,
        media_type: str | None = None,
    ) -> StoredArtifact:
        """Hash and atomically publish one immutable byte string.

        Existing content is never overwritten.  If the digest target already
        exists, it is streamed and verified before the publication is treated
        as an idempotent success.
        """

        if not isinstance(data, bytes):
            _fail(
                ArtifactStoreErrorCode.INVALID_ARTIFACT,
                "artifact data must be immutable bytes",
            )
        if len(data) > self.max_artifact_bytes:
            _fail(
                ArtifactStoreErrorCode.TOO_LARGE,
                "artifact exceeds max_artifact_bytes",
            )
        if media_type is not None and (
            not isinstance(media_type, str) or not media_type.strip()
        ):
            _fail(
                ArtifactStoreErrorCode.INVALID_ARTIFACT,
                "media_type must be a non-empty string when provided",
            )
        digest = hashlib.sha256(data).hexdigest()
        if expected_sha256 is not None:
            expected = _require_sha256(expected_sha256, "expected_sha256")
            if digest != expected:
                _fail(
                    ArtifactStoreErrorCode.DIGEST_MISMATCH,
                    "artifact digest differs from expected_sha256",
                )

        root_fd = self._open_bound_root()
        prefix_fd: int | None = None
        try:
            prefix_fd = self._open_prefix(root_fd, digest[:2])
            self._publish_in_prefix(prefix_fd, digest, data)
        finally:
            if prefix_fd is not None:
                os.close(prefix_fd)
            os.close(root_fd)
        return StoredArtifact(
            uri=f"{_CAS_PREFIX}{digest}",
            sha256=digest,
            size_bytes=len(data),
            path=self.root / digest[:2] / digest,
            media_type=media_type,
        )

    def put_bytes(
        self,
        data: bytes,
        *,
        media_type: str | None = None,
        expected_sha256: str | None = None,
    ) -> StoredArtifact:
        """Generic parser-facing alias for :meth:`publish`.

        This deliberately accepts already-bounded encoded bytes only.  It is
        suitable for materializing Docling data-URI payloads after their
        decoder has enforced encoded/decoded limits, as well as any future
        parser adapter that has no vendor-specific storage callback.
        """

        return self.publish(
            data,
            expected_sha256=expected_sha256,
            media_type=media_type,
        )

    @staticmethod
    def _open_prefix(root_fd: int, prefix: str) -> int:
        flags = (
            os.O_RDONLY
            | os.O_DIRECTORY
            | os.O_NOFOLLOW
            | getattr(os, "O_CLOEXEC", 0)
        )
        try:
            os.mkdir(prefix, mode=0o700, dir_fd=root_fd)
            os.fsync(root_fd)
        except FileExistsError:
            pass
        except OSError as exc:
            raise ArtifactStoreError(
                ArtifactStoreErrorCode.IO_ERROR,
                "could not create content-addressed prefix directory",
            ) from exc
        try:
            descriptor = os.open(prefix, flags, dir_fd=root_fd)
            metadata = os.fstat(descriptor)
        except OSError as exc:
            raise ArtifactStoreError(
                ArtifactStoreErrorCode.UNSAFE_PATH,
                "content-addressed prefix is not a safe directory",
            ) from exc
        if not stat.S_ISDIR(metadata.st_mode):  # pragma: no cover - guarded by open
            os.close(descriptor)
            _fail(
                ArtifactStoreErrorCode.UNSAFE_PATH,
                "content-addressed prefix is not a directory",
            )
        return descriptor

    def _publish_in_prefix(self, prefix_fd: int, digest: str, data: bytes) -> None:
        try:
            existing = os.stat(digest, dir_fd=prefix_fd, follow_symlinks=False)
        except FileNotFoundError:
            existing = None
        except OSError as exc:
            raise ArtifactStoreError(
                ArtifactStoreErrorCode.IO_ERROR,
                "could not inspect content-addressed target",
            ) from exc
        if existing is not None:
            self._verify_existing(prefix_fd, digest, expected_size=len(data))
            return

        temporary_name = (
            f".publish-{os.getpid()}-{secrets.token_hex(16)}"
        )
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | os.O_NOFOLLOW
            | getattr(os, "O_CLOEXEC", 0)
        )
        descriptor: int | None = None
        try:
            descriptor = os.open(temporary_name, flags, 0o600, dir_fd=prefix_fd)
            view = memoryview(data)
            written = 0
            while written < len(view):
                count = os.write(descriptor, view[written : written + self.chunk_size])
                if count < 1:  # pragma: no cover - defensive OS contract check
                    raise OSError(errno.EIO, "zero-byte artifact write")
                written += count
            os.fsync(descriptor)
            os.fchmod(descriptor, 0o400)
            os.fsync(descriptor)
            os.close(descriptor)
            descriptor = None
            try:
                os.link(
                    temporary_name,
                    digest,
                    src_dir_fd=prefix_fd,
                    dst_dir_fd=prefix_fd,
                    follow_symlinks=False,
                )
                os.fsync(prefix_fd)
            except FileExistsError:
                # Another publisher won after our first stat.  Its bytes must
                # still be verified before accepting idempotent success.
                self._verify_existing(prefix_fd, digest, expected_size=len(data))
        except ArtifactStoreError:
            raise
        except OSError as exc:
            raise ArtifactStoreError(
                ArtifactStoreErrorCode.IO_ERROR,
                "could not durably publish content-addressed artifact",
            ) from exc
        finally:
            if descriptor is not None:
                os.close(descriptor)
            try:
                os.unlink(temporary_name, dir_fd=prefix_fd)
            except FileNotFoundError:
                pass
            except OSError as exc:
                raise ArtifactStoreError(
                    ArtifactStoreErrorCode.IO_ERROR,
                    "could not remove staged artifact inode",
                ) from exc

    def _verify_existing(
        self, prefix_fd: int, digest: str, *, expected_size: int
    ) -> None:
        # A successful publisher creates the final hard link and then removes
        # its private staging name.  That link-count transition can change
        # ctime while a concurrent idempotent publisher is reading the final
        # inode.  Retry a small bounded number of complete reads so this safe
        # transition succeeds without dropping ctime from mutation detection.
        # A target that keeps changing fails closed.
        for _attempt in range(3):
            if self._verify_existing_once(
                prefix_fd, digest, expected_size=expected_size
            ):
                return
        _fail(
            ArtifactStoreErrorCode.TARGET_CONFLICT,
            "existing content-addressed target kept changing during verification",
        )

    def _verify_existing_once(
        self, prefix_fd: int, digest: str, *, expected_size: int
    ) -> bool:
        flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
        try:
            descriptor = os.open(digest, flags, dir_fd=prefix_fd)
        except OSError as exc:
            code = (
                ArtifactStoreErrorCode.UNSAFE_PATH
                if exc.errno == errno.ELOOP
                else ArtifactStoreErrorCode.IO_ERROR
            )
            raise ArtifactStoreError(
                code, "existing content-addressed target could not be opened safely"
            ) from exc
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode):
                _fail(
                    ArtifactStoreErrorCode.TARGET_CONFLICT,
                    "existing content-addressed target is not a regular file",
                )
            if before.st_size != expected_size:
                _fail(
                    ArtifactStoreErrorCode.TARGET_CONFLICT,
                    "existing content-addressed target has a conflicting size",
                )
            hasher = hashlib.sha256()
            total = 0
            while True:
                chunk = os.read(descriptor, self.chunk_size)
                if not chunk:
                    break
                total += len(chunk)
                if total > self.max_artifact_bytes:
                    _fail(
                        ArtifactStoreErrorCode.TARGET_CONFLICT,
                        "existing content-addressed target exceeds store limits",
                    )
                hasher.update(chunk)
            after = os.fstat(descriptor)
        except ArtifactStoreError:
            raise
        except OSError as exc:
            raise ArtifactStoreError(
                ArtifactStoreErrorCode.IO_ERROR,
                "existing content-addressed target could not be verified",
            ) from exc
        finally:
            os.close(descriptor)
        if not _stable_stat(before, after):
            return False
        if total != expected_size or hasher.hexdigest() != digest:
            _fail(
                ArtifactStoreErrorCode.TARGET_CONFLICT,
                "existing content-addressed target does not match its digest name",
            )
        return True

    def make_asset_resolver(
        self,
        *,
        max_assets: int = 64,
        max_asset_bytes: int | None = None,
        max_total_bytes: int = 128 * 1024 * 1024,
    ) -> AssetMaterialResolver:
        """Return a verifier configured for this exact CAS root and layout."""

        return AssetMaterialResolver(
            content_addressed_root=self.root,
            content_layout="prefix2",
            max_assets=max_assets,
            max_asset_bytes=(
                self.max_artifact_bytes
                if max_asset_bytes is None
                else max_asset_bytes
            ),
            max_total_bytes=max_total_bytes,
            chunk_size=self.chunk_size,
        )


@dataclass(frozen=True, slots=True)
class MinerUArchivePublicationLimits:
    """Independent decompression bounds for one MinerU result archive."""

    max_archive_bytes: int = 512 * 1024 * 1024
    max_members: int = 4096
    max_member_bytes: int = 64 * 1024 * 1024
    max_total_uncompressed_bytes: int = 512 * 1024 * 1024
    max_compression_ratio: int = 200
    read_chunk_size: int = 1024 * 1024

    def __post_init__(self) -> None:
        for name in (
            "max_archive_bytes",
            "max_members",
            "max_member_bytes",
            "max_total_uncompressed_bytes",
            "max_compression_ratio",
            "read_chunk_size",
        ):
            _positive_integer(getattr(self, name), name)


def _safe_zip_name(name: object) -> PurePosixPath:
    if (
        not isinstance(name, str)
        or not name
        or "\\" in name
        or "\x00" in name
        or len(name) > 4096
    ):
        _fail(
            ArtifactStoreErrorCode.INVALID_ARCHIVE,
            "MinerU ZIP contains an unsafe member name",
        )
    path = PurePosixPath(name)
    if (
        path.is_absolute()
        or name.startswith("/")
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        _fail(
            ArtifactStoreErrorCode.INVALID_ARCHIVE,
            "MinerU ZIP contains a traversal member",
        )
    return path


class LocalMinerUArchivePublisher:
    """Validate and publish a MinerU result ZIP and all regular members."""

    __slots__ = ("store", "limits")

    def __init__(
        self,
        store: LocalContentAddressedArtifactStore,
        *,
        limits: MinerUArchivePublicationLimits | None = None,
    ) -> None:
        if not isinstance(store, LocalContentAddressedArtifactStore):
            raise TypeError("store must be a LocalContentAddressedArtifactStore")
        self.store = store
        self.limits = limits or MinerUArchivePublicationLimits()
        if not isinstance(self.limits, MinerUArchivePublicationLimits):
            raise TypeError("limits must be MinerUArchivePublicationLimits")

    def __call__(self, value: MinerUArchive) -> PublishedMinerUArchive:
        if not isinstance(value, MinerUArchive):
            raise TypeError("MinerU publisher requires a MinerUArchive")
        if not isinstance(value.data, bytes) or not value.data:
            _fail(
                ArtifactStoreErrorCode.INVALID_ARCHIVE,
                "MinerU archive data must be non-empty immutable bytes",
            )
        if len(value.data) > self.limits.max_archive_bytes:
            _fail(
                ArtifactStoreErrorCode.ARCHIVE_LIMIT,
                "MinerU archive exceeds max_archive_bytes",
            )
        declared = _require_sha256(value.archive_sha256, "archive_sha256")
        if hashlib.sha256(value.data).hexdigest() != declared:
            _fail(
                ArtifactStoreErrorCode.DIGEST_MISMATCH,
                "MinerU archive digest differs from downloaded bytes",
            )
        try:
            archive = zipfile.ZipFile(BytesIO(value.data))
        except (OSError, zipfile.BadZipFile) as exc:
            raise ArtifactStoreError(
                ArtifactStoreErrorCode.INVALID_ARCHIVE,
                "MinerU result is not a valid ZIP archive",
            ) from exc

        member_uris: dict[str, str] = {}
        with archive:
            infos = archive.infolist()
            if len(infos) > self.limits.max_members:
                _fail(
                    ArtifactStoreErrorCode.ARCHIVE_LIMIT,
                    "MinerU ZIP has too many members",
                )
            files: list[zipfile.ZipInfo] = []
            seen: set[str] = set()
            total = 0
            for info in infos:
                _safe_zip_name(info.filename)
                if info.filename in seen:
                    _fail(
                        ArtifactStoreErrorCode.INVALID_ARCHIVE,
                        "MinerU ZIP contains duplicate member names",
                    )
                seen.add(info.filename)
                if info.flag_bits & 0x1:
                    _fail(
                        ArtifactStoreErrorCode.INVALID_ARCHIVE,
                        "MinerU ZIP contains an encrypted member",
                    )
                mode = info.external_attr >> 16
                file_type = stat.S_IFMT(mode)
                if file_type == stat.S_IFLNK:
                    _fail(
                        ArtifactStoreErrorCode.INVALID_ARCHIVE,
                        "MinerU ZIP contains a symbolic-link member",
                    )
                if info.is_dir():
                    if file_type not in {0, stat.S_IFDIR}:
                        _fail(
                            ArtifactStoreErrorCode.INVALID_ARCHIVE,
                            "MinerU ZIP directory has an unsafe file type",
                        )
                    continue
                if file_type not in {0, stat.S_IFREG}:
                    _fail(
                        ArtifactStoreErrorCode.INVALID_ARCHIVE,
                        "MinerU ZIP contains a non-regular member",
                    )
                if info.file_size > self.limits.max_member_bytes:
                    _fail(
                        ArtifactStoreErrorCode.ARCHIVE_LIMIT,
                        "MinerU ZIP member exceeds max_member_bytes",
                    )
                total += info.file_size
                if total > self.limits.max_total_uncompressed_bytes:
                    _fail(
                        ArtifactStoreErrorCode.ARCHIVE_LIMIT,
                        "MinerU ZIP exceeds max_total_uncompressed_bytes",
                    )
                ratio = info.file_size / max(1, info.compress_size)
                if ratio > self.limits.max_compression_ratio:
                    _fail(
                        ArtifactStoreErrorCode.ARCHIVE_LIMIT,
                        "MinerU ZIP member exceeds max_compression_ratio",
                    )
                files.append(info)

            actual_members = tuple(info.filename for info in files)
            if (
                not isinstance(value.members, tuple)
                or any(not isinstance(item, str) for item in value.members)
                or value.members != actual_members
            ):
                _fail(
                    ArtifactStoreErrorCode.INVALID_ARCHIVE,
                    "MinerUArchive member declaration differs from ZIP contents",
                )
            if not files:
                _fail(
                    ArtifactStoreErrorCode.INVALID_ARCHIVE,
                    "MinerU ZIP contains no regular files",
                )

            for info in files:
                data = self._read_member(archive, info)
                computed = hashlib.sha256(data).hexdigest()
                try:
                    expected_member_digest = value.member_sha256s[info.filename]
                except KeyError as exc:  # pragma: no cover - guarded by DTO + names
                    raise ArtifactStoreError(
                        ArtifactStoreErrorCode.INVALID_ARCHIVE,
                        "MinerU archive omitted a member digest",
                    ) from exc
                if computed != expected_member_digest:
                    _fail(
                        ArtifactStoreErrorCode.DIGEST_MISMATCH,
                        "MinerU member digest differs from decompressed bytes",
                    )
                stored = self.store.publish(
                    data, expected_sha256=expected_member_digest
                )
                member_uris[info.filename] = stored.uri

        stored_archive = self.store.publish(
            value.data,
            expected_sha256=declared,
            media_type="application/zip",
        )
        return PublishedMinerUArchive(
            archive_uri=stored_archive.uri,
            archive_sha256=stored_archive.sha256,
            member_uris=MappingProxyType(member_uris),
        )

    def _read_member(self, archive: zipfile.ZipFile, info: zipfile.ZipInfo) -> bytes:
        output = bytearray()
        try:
            with archive.open(info, "r") as source:
                while True:
                    chunk = source.read(
                        min(
                            self.limits.read_chunk_size,
                            self.limits.max_member_bytes + 1 - len(output),
                        )
                    )
                    if not chunk:
                        break
                    output.extend(chunk)
                    if len(output) > self.limits.max_member_bytes:
                        _fail(
                            ArtifactStoreErrorCode.ARCHIVE_LIMIT,
                            "MinerU ZIP member exceeds max_member_bytes while reading",
                        )
        except ArtifactStoreError:
            raise
        except (OSError, EOFError, RuntimeError, NotImplementedError, zipfile.BadZipFile) as exc:
            raise ArtifactStoreError(
                ArtifactStoreErrorCode.INVALID_ARCHIVE,
                "MinerU ZIP member could not be safely decompressed",
            ) from exc
        if len(output) != info.file_size:
            _fail(
                ArtifactStoreErrorCode.INVALID_ARCHIVE,
                "MinerU ZIP member size differs from its directory entry",
            )
        return bytes(output)


class LocalPaddleOCRVLArtifactSink:
    """Callable adapter from PaddleOCR-VL artifacts to the local CAS."""

    __slots__ = ("store",)

    def __init__(self, store: LocalContentAddressedArtifactStore) -> None:
        if not isinstance(store, LocalContentAddressedArtifactStore):
            raise TypeError("store must be a LocalContentAddressedArtifactStore")
        self.store = store

    def __call__(self, value: PaddleOCRVLArtifact) -> StoredPaddleOCRVLArtifact:
        if not isinstance(value, PaddleOCRVLArtifact):
            raise TypeError("Paddle sink requires a PaddleOCRVLArtifact")
        stored = self.store.publish(
            value.data,
            expected_sha256=value.sha256,
            media_type=value.media_type,
        )
        return StoredPaddleOCRVLArtifact(
            uri=stored.uri,
            sha256=stored.sha256,
            media_type=value.media_type,
        )


def make_mineru_archive_publisher(
    store: LocalContentAddressedArtifactStore,
    *,
    limits: MinerUArchivePublicationLimits | None = None,
) -> LocalMinerUArchivePublisher:
    """Factory with the exact callable shape expected by MinerUServiceLoader."""

    return LocalMinerUArchivePublisher(store, limits=limits)


def make_paddleocr_vl_artifact_sink(
    store: LocalContentAddressedArtifactStore,
) -> LocalPaddleOCRVLArtifactSink:
    """Factory with the exact callable shape expected by PaddleOCR-VL."""

    return LocalPaddleOCRVLArtifactSink(store)


__all__ = [
    "ARTIFACT_STORE_SCHEMA_VERSION",
    "ArtifactStoreError",
    "ArtifactStoreErrorCode",
    "LocalContentAddressedArtifactStore",
    "LocalMinerUArchivePublisher",
    "LocalPaddleOCRVLArtifactSink",
    "MinerUArchivePublicationLimits",
    "StoredArtifact",
    "make_mineru_archive_publisher",
    "make_paddleocr_vl_artifact_sink",
]
