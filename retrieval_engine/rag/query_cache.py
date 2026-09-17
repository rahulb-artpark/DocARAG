"""Fail-closed query cache identities and private atomic JSON publication.

This module deliberately keeps cache policy separate from RAG strategies.  A
query result is reusable only when a caller supplies a verified
``QueryCacheContext``.  Question text alone is never treated as sufficient
evidence that an answer belongs to the current document/index/model runtime.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, is_dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, IO


_SHA256 = re.compile(r"[0-9a-f]{64}")
_CACHE_SCHEMA_VERSION = 1
_SECRET_FIELD_NAMES = frozenset(
    {
        "api_key",
        "password",
        "secret",
        "token",
        "access_token",
        "auth_token",
        "authorization",
        "credential",
        "credentials",
        "private_key",
    }
)


def _require_sha256(value: object, field_name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
    return value


def _canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_payload(payload: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _feed_digest(digest: "hashlib._Hash", value: bytes) -> None:
    """Add exact bytes to a digest without concatenation ambiguity."""

    digest.update(len(value).to_bytes(8, "big"))
    digest.update(value)


def _json_value(value: Any) -> Any:
    """Return a deterministic JSON value from validated config objects."""

    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, set):
        converted = [_json_value(item) for item in value]
        return sorted(converted, key=lambda item: _canonical_json_bytes(item))
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    # Configuration models should have been handled above.  Rejecting an
    # unknown value is safer than deriving an unstable identity from repr().
    raise TypeError(f"configuration contains a non-serializable {type(value).__name__}")


def _is_secret_field(name: str) -> bool:
    normalized = name.lower()
    return normalized in _SECRET_FIELD_NAMES or any(
        normalized.endswith(suffix)
        for suffix in (
            "_api_key",
            "_password",
            "_secret",
            "_token",
            "_credential",
            "_private_key",
        )
    )


def _redact_secrets(value: Any) -> Any:
    """Remove credentials while retaining whether a credential was configured."""

    if isinstance(value, Mapping):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            name = str(key)
            if _is_secret_field(name):
                redacted[name] = {
                    "redacted": True,
                    "configured": item is not None and item != "",
                }
            else:
                redacted[name] = _redact_secrets(item)
        return redacted
    if isinstance(value, list):
        return [_redact_secrets(item) for item in value]
    return value


def _model_projection(value: Any, path: tuple[str, ...] = ()) -> list[dict[str, Any]]:
    """Collect every resolved model-bearing config block with its JSON path."""

    models: list[dict[str, Any]] = []
    if isinstance(value, Mapping):
        if any(
            isinstance(item, str)
            and (str(key) == "model_name" or str(key).endswith("_model_name"))
            for key, item in value.items()
        ):
            models.append({"path": list(path), "config": dict(value)})
        for key in sorted(value):
            models.extend(_model_projection(value[key], (*path, str(key))))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            models.extend(_model_projection(item, (*path, str(index))))
    return models


def query_prompt_code_sha256(project_root: Path | None = None) -> str:
    """Hash the current first-party query/prompt execution code.

    This is intentionally based on source bytes, not a package version that a
    developer might forget to bump.  Hashing both first-party Python packages
    is conservative: an indexing-only edit can invalidate answer caches, but a
    query dependency cannot be silently omitted from a hand-maintained list.
    The cost is bounded and paid once when constructing a run context.
    """

    root = (
        project_root.expanduser().resolve()
        if project_root is not None
        else Path(__file__).resolve().parents[2]
    )
    candidates = [
        *(root / "retrieval_engine").rglob("*.py"),
        *(root / "document_services").rglob("*.py"),
    ]
    files = sorted({path.resolve() for path in candidates if path.is_file()})
    if not files:
        raise FileNotFoundError("query prompt/code sources are unavailable")

    digest = hashlib.sha256()
    for path in files:
        try:
            relative = path.relative_to(root).as_posix().encode("utf-8")
        except ValueError as exc:
            raise ValueError("query source path escapes the project root") from exc
        _feed_digest(digest, relative)
        before = path.stat()
        with path.open("rb") as source:
            content_digest = hashlib.sha256()
            size = 0
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                size += len(chunk)
                content_digest.update(chunk)
        after = path.stat()
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise OSError(f"query source changed while hashing: {path}")
        _feed_digest(digest, size.to_bytes(8, "big"))
        _feed_digest(digest, content_digest.digest())
    return digest.hexdigest()


@dataclass(frozen=True)
class QueryCacheContext:
    """The complete non-question identity required for safe answer reuse."""

    artifact_set_sha256: str
    resolved_config_sha256: str
    model_identity_sha256: str
    query_prompt_code_sha256: str

    def __post_init__(self) -> None:
        for field_name in (
            "artifact_set_sha256",
            "resolved_config_sha256",
            "model_identity_sha256",
            "query_prompt_code_sha256",
        ):
            _require_sha256(getattr(self, field_name), field_name)

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": _CACHE_SCHEMA_VERSION,
            "artifact_set_sha256": self.artifact_set_sha256,
            "resolved_config_sha256": self.resolved_config_sha256,
            "model_identity_sha256": self.model_identity_sha256,
            "query_prompt_code_sha256": self.query_prompt_code_sha256,
        }

    @classmethod
    def from_system_config(
        cls,
        cfg: object,
        *,
        artifact_set_sha256: str,
        prompt_code_sha256: str | None = None,
        project_root: Path | None = None,
    ) -> "QueryCacheContext":
        """Bind a validated/default-resolved config without retaining secrets."""

        resolved = _redact_secrets(_json_value(cfg))
        models = _model_projection(resolved)
        return cls(
            artifact_set_sha256=_require_sha256(
                artifact_set_sha256, "artifact_set_sha256"
            ),
            resolved_config_sha256=_sha256_payload(resolved),
            model_identity_sha256=_sha256_payload(
                {"schema_version": 1, "models": models}
            ),
            query_prompt_code_sha256=(
                prompt_code_sha256
                if prompt_code_sha256 is not None
                else query_prompt_code_sha256(project_root)
            ),
        )


@dataclass(frozen=True)
class QueryCacheBinding:
    """Question-specific immutable cache identity."""

    cache_key: str
    question_sha256: str
    question_utf8_bytes: int
    context: QueryCacheContext

    @classmethod
    def create(cls, question: str, context: QueryCacheContext) -> "QueryCacheBinding":
        if not isinstance(question, str):
            raise TypeError("question must be a string")
        question_bytes = question.encode("utf-8", errors="strict")
        question_sha256 = hashlib.sha256(question_bytes).hexdigest()
        digest = hashlib.sha256()
        _feed_digest(digest, b"docarag-query-cache-binding-v1")
        # Feed the exact question bytes, rather than a normalized or stripped
        # spelling.  Visually similar Unicode and trailing whitespace differ.
        _feed_digest(digest, question_bytes)
        for field_name, value in context.as_dict().items():
            _feed_digest(digest, str(field_name).encode("utf-8"))
            _feed_digest(digest, str(value).encode("utf-8"))
        return cls(
            cache_key=digest.hexdigest(),
            question_sha256=question_sha256,
            question_utf8_bytes=len(question_bytes),
            context=context,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": _CACHE_SCHEMA_VERSION,
            "cache_key": self.cache_key,
            "question_sha256": self.question_sha256,
            "question_utf8_bytes": self.question_utf8_bytes,
            "context": self.context.as_dict(),
        }


def question_directory_key(question: str, context: QueryCacheContext | None) -> str:
    """Return an identity-derived directory key, even when reuse is disabled."""

    if context is not None:
        return QueryCacheBinding.create(question, context).cache_key
    question_bytes = question.encode("utf-8", errors="strict")
    digest = hashlib.sha256()
    _feed_digest(digest, b"docarag-uncached-question-directory-v1")
    _feed_digest(digest, question_bytes)
    return digest.hexdigest()


def atomic_write_bytes(path: Path, payload: bytes, *, mode: int = 0o600) -> None:
    """Publish one private file atomically and durably within its directory."""

    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.parent.chmod(0o700)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
        path.chmod(mode)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_json(path: Path, payload: object, *, indent: int | None = 2) -> bytes:
    """Serialize and atomically publish JSON, returning the exact bytes written."""

    encoded = (
        json.dumps(
            payload,
            indent=indent,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    atomic_write_bytes(path, encoded)
    return encoded


class AtomicJsonArrayWriter:
    """Incrementally write a JSON array and publish it only after success."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._temporary: Path | None = None
        self._output: IO[str] | None = None
        self._first = True
        self.count = 0

    def __enter__(self) -> "AtomicJsonArrayWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.path.parent.chmod(0o700)
        # An aggregate from an earlier invocation must not be mistaken for the
        # output of this in-progress invocation if the latter later fails.
        self.path.unlink(missing_ok=True)
        descriptor, name = tempfile.mkstemp(
            prefix=f".{self.path.name}.", dir=self.path.parent
        )
        os.chmod(name, 0o600)
        self._temporary = Path(name)
        self._output = os.fdopen(descriptor, "w", encoding="utf-8", newline="\n")
        self._output.write("[\n")
        return self

    def append(self, payload: object) -> None:
        if self._output is None:
            raise RuntimeError("JSON array writer is not active")
        if not self._first:
            self._output.write(",\n")
        json.dump(
            payload,
            self._output,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        self._first = False
        self.count += 1

    def __exit__(self, exc_type, _exc, _traceback) -> bool:
        output = self._output
        temporary = self._temporary
        self._output = None
        self._temporary = None
        try:
            if output is not None:
                if exc_type is None:
                    output.write("\n]\n")
                    output.flush()
                    os.fsync(output.fileno())
                output.close()
            if exc_type is None and temporary is not None:
                os.replace(temporary, self.path)
                self.path.chmod(0o600)
                directory_fd = os.open(self.path.parent, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return False


def cache_commit_payload(binding: QueryCacheBinding, result_bytes: bytes) -> dict[str, object]:
    return {
        "schema_version": _CACHE_SCHEMA_VERSION,
        "binding": binding.as_dict(),
        "result_sha256": hashlib.sha256(result_bytes).hexdigest(),
        "result_size_bytes": len(result_bytes),
    }
