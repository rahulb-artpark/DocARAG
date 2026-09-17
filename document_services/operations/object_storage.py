"""Immutable object-storage port for single-host and S3/MinIO deployments."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
from typing import Protocol

from document_services.persistence.artifact_store import LocalContentAddressedArtifactStore


OBJECT_STORE_SCHEMA_VERSION = "docarag.object-store.v1"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_PREFIX = re.compile(r"[a-z0-9](?:[a-z0-9._/-]{0,510}[a-z0-9])?")


class ObjectStoreError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ObjectReceipt:
    uri: str
    sha256: str
    size_bytes: int
    media_type: str
    backend: str

    def __post_init__(self) -> None:
        if _SHA256.fullmatch(self.sha256) is None or self.size_bytes < 0:
            raise ObjectStoreError("invalid object receipt")
        if not self.uri or not self.media_type or not self.backend:
            raise ObjectStoreError("object receipt fields must be non-empty")


class ImmutableObjectStore(Protocol):
    def put_bytes(
        self,
        data: bytes,
        *,
        media_type: str,
        expected_sha256: str | None = None,
    ) -> ObjectReceipt: ...

    def get_bytes(self, receipt: ObjectReceipt, *, max_bytes: int) -> bytes: ...


class LocalObjectStore:
    """Object-store adapter over DocARAG's hardened local content-addressed store."""

    def __init__(self, root: str | Path, *, max_artifact_bytes: int = 512 * 1024 * 1024):
        self.store = LocalContentAddressedArtifactStore(
            Path(root).expanduser().resolve(), max_artifact_bytes=max_artifact_bytes
        )

    def put_bytes(
        self,
        data: bytes,
        *,
        media_type: str,
        expected_sha256: str | None = None,
    ) -> ObjectReceipt:
        stored = self.store.put_bytes(
            data, media_type=media_type, expected_sha256=expected_sha256
        )
        return ObjectReceipt(
            uri=stored.uri,
            sha256=stored.sha256,
            size_bytes=stored.size_bytes,
            media_type=media_type,
            backend="local-cas",
        )

    def get_bytes(self, receipt: ObjectReceipt, *, max_bytes: int) -> bytes:
        if receipt.backend != "local-cas":
            raise ObjectStoreError("receipt belongs to another object-store backend")
        if receipt.size_bytes > max_bytes:
            raise ObjectStoreError("object exceeds caller max_bytes")
        try:
            path = self.store.root / receipt.sha256[:2] / receipt.sha256
            data = path.read_bytes()
        except OSError as exc:
            raise ObjectStoreError("local object is unavailable") from exc
        _verify_object_bytes(data, receipt, max_bytes=max_bytes)
        return data


class S3CompatibleObjectStore:
    """Small injected-client adapter for S3, MinIO, and compatible stores.

    The client is intentionally injected, keeping boto3 out of the controller's
    required dependency set.  It must expose ``put_object``, ``head_object`` and
    ``get_object`` methods with boto3-compatible shapes.  Objects are immutable:
    publication uses ``IfNoneMatch='*'`` and an existing key is accepted only
    after its stored digest and length are verified.
    """

    def __init__(self, client: object, *, bucket: str, prefix: str = "docarag/v1") -> None:
        if not bucket or len(bucket) > 255:
            raise ValueError("bucket must be a bounded non-empty name")
        prefix = prefix.strip("/")
        if _PREFIX.fullmatch(prefix) is None or ".." in prefix.split("/"):
            raise ValueError("prefix is invalid")
        for method in ("put_object", "head_object", "get_object"):
            if not callable(getattr(client, method, None)):
                raise TypeError(f"S3-compatible client lacks {method}")
        self.client = client
        self.bucket = bucket
        self.prefix = prefix

    def _key(self, digest: str) -> str:
        return f"{self.prefix}/sha256/{digest[:2]}/{digest}"

    def put_bytes(
        self,
        data: bytes,
        *,
        media_type: str,
        expected_sha256: str | None = None,
    ) -> ObjectReceipt:
        if not isinstance(data, bytes) or not media_type.strip():
            raise ValueError("immutable bytes and media_type are required")
        digest = hashlib.sha256(data).hexdigest()
        if expected_sha256 is not None and digest != expected_sha256:
            raise ObjectStoreError("object digest differs from expected_sha256")
        key = self._key(digest)
        try:
            self.client.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=data,
                ContentLength=len(data),
                ContentType=media_type,
                Metadata={
                    "docarag-sha256": digest,
                    "docarag-schema": OBJECT_STORE_SCHEMA_VERSION,
                },
                IfNoneMatch="*",
            )
        except Exception:
            # Conditional create commonly raises for an idempotent existing
            # object.  Verify it rather than relying on vendor-specific errors.
            self._verify_head(key, digest=digest, size_bytes=len(data))
        else:
            self._verify_head(key, digest=digest, size_bytes=len(data))
        return ObjectReceipt(
            uri=f"s3://{self.bucket}/{key}",
            sha256=digest,
            size_bytes=len(data),
            media_type=media_type,
            backend="s3-compatible",
        )

    def get_bytes(self, receipt: ObjectReceipt, *, max_bytes: int) -> bytes:
        if receipt.backend != "s3-compatible":
            raise ObjectStoreError("receipt belongs to another object-store backend")
        expected_uri = f"s3://{self.bucket}/{self._key(receipt.sha256)}"
        if receipt.uri != expected_uri:
            raise ObjectStoreError("object receipt URI is outside the configured namespace")
        if receipt.size_bytes > max_bytes:
            raise ObjectStoreError("object exceeds caller max_bytes")
        try:
            response = self.client.get_object(
                Bucket=self.bucket, Key=self._key(receipt.sha256)
            )
            body = response["Body"]
            data = body.read(max_bytes + 1)
        except Exception as exc:
            raise ObjectStoreError("object download failed") from exc
        _verify_object_bytes(data, receipt, max_bytes=max_bytes)
        return data

    def _verify_head(self, key: str, *, digest: str, size_bytes: int) -> None:
        try:
            head = self.client.head_object(Bucket=self.bucket, Key=key)
            metadata = {
                str(k).lower(): str(v) for k, v in dict(head.get("Metadata", {})).items()
            }
            stored_size = int(head["ContentLength"])
        except Exception as exc:
            raise ObjectStoreError("object publication could not be verified") from exc
        if metadata.get("docarag-sha256") != digest or stored_size != size_bytes:
            raise ObjectStoreError("immutable object key conflicts with different bytes")


def _verify_object_bytes(data: bytes, receipt: ObjectReceipt, *, max_bytes: int) -> None:
    if not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes < 1:
        raise ValueError("max_bytes must be positive")
    if len(data) > max_bytes:
        raise ObjectStoreError("object exceeds caller max_bytes")
    if len(data) != receipt.size_bytes or hashlib.sha256(data).hexdigest() != receipt.sha256:
        raise ObjectStoreError("object bytes do not match immutable receipt")


__all__ = [
    "ImmutableObjectStore",
    "LocalObjectStore",
    "OBJECT_STORE_SCHEMA_VERSION",
    "ObjectReceipt",
    "ObjectStoreError",
    "S3CompatibleObjectStore",
]
