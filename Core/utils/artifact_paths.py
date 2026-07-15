"""Safe run-local artifacts and dependency-light Qdrant health probes."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
from typing import Any, Callable, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit, urlunsplit
from urllib.request import Request, urlopen


MAX_QDRANT_RESPONSE_BYTES = 64 * 1024
_ENVIRONMENT_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


class QdrantUnavailableError(RuntimeError):
    """The configured Qdrant service could not be reached in time."""


class QdrantProtocolError(RuntimeError):
    """Qdrant returned an unsafe or unsupported response."""


@dataclass(frozen=True)
class QdrantCollectionState:
    """One exact, collection-and-scope-specific point-count observation."""

    exists: bool
    point_count: int | None


def resolve_run_artifact_path(
    save_path: str | Path,
    configured_path: str | Path,
    *,
    label: str,
) -> Path:
    """Resolve an artifact path and reject escapes from the document run."""

    root = Path(save_path).expanduser().resolve()
    configured = Path(configured_path).expanduser()
    unresolved = configured if configured.is_absolute() else root / configured
    lexical = Path(os.path.abspath(unresolved))
    try:
        lexical_relative = lexical.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"{label} must stay below the document run {root}: {lexical}"
        ) from exc
    current = root
    for part in lexical_relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"{label} may not traverse symbolic links: {current}")

    candidate = lexical.resolve()
    try:
        relative = candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"{label} must stay below the document run {root}: {candidate}"
        ) from exc
    if relative == Path("."):
        raise ValueError(f"{label} cannot be the document run root: {root}")
    return candidate


def canonical_qdrant_url(value: str) -> str:
    """Validate and normalize a credential-free HTTP(S) Qdrant base URL."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError("qdrant_url must be a non-empty string")
    if value != value.strip():
        raise ValueError("qdrant_url may not have surrounding whitespace")
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("qdrant_url must be an absolute HTTP(S) URL")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("qdrant_url must not contain credentials")
    if parsed.query or parsed.fragment:
        raise ValueError("qdrant_url must not contain a query or fragment")
    path = parsed.path.rstrip("/")
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def qdrant_endpoint_sha256(value: str) -> str:
    """Return a portable endpoint binding without exposing deployment topology."""

    normalized = canonical_qdrant_url(value)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _scope_filter(scope_filter: Mapping[str, Any]) -> dict[str, object]:
    if not isinstance(scope_filter, Mapping) or not scope_filter:
        raise ValueError("Qdrant count requires a non-empty scope_filter")
    clauses: list[dict[str, object]] = []
    for key in sorted(scope_filter):
        value = scope_filter[key]
        if not isinstance(key, str) or not key or key != key.strip():
            raise ValueError("scope_filter keys must be non-empty trimmed strings")
        if value is None or isinstance(value, (dict, list, tuple, set)):
            raise ValueError("scope_filter values must be non-null scalars")
        if not isinstance(value, (str, int, float, bool)):
            raise ValueError("scope_filter values must be JSON scalars")
        if isinstance(value, str) and (not value or value != value.strip()):
            raise ValueError("string scope_filter values must be non-empty and trimmed")
        clauses.append({"key": key, "match": {"value": value}})
    return {"must": clauses}


def qdrant_collection_state(
    *,
    qdrant_url: str,
    collection_name: str,
    scope_filter: Mapping[str, Any],
    api_key_env: str = "QDRANT_API_KEY",
    timeout_seconds: float = 10.0,
    opener: Callable[..., Any] | None = None,
) -> QdrantCollectionState:
    """Read an exact scoped count from Qdrant without importing its SDK.

    A missing collection is a proven non-ready state. Transport failures and
    malformed responses raise distinct errors so callers can report an offline
    dependency separately from a content mismatch. The API key is read only
    from the named environment variable and is never returned or formatted.
    """

    base_url = canonical_qdrant_url(qdrant_url)
    if not isinstance(collection_name, str) or not collection_name.strip():
        raise ValueError("collection_name must be a non-empty string")
    if collection_name != collection_name.strip():
        raise ValueError("collection_name may not have surrounding whitespace")
    if not isinstance(api_key_env, str) or _ENVIRONMENT_NAME.fullmatch(api_key_env) is None:
        raise ValueError("api_key_env must be an environment-variable name")
    if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)):
        raise ValueError("timeout_seconds must be a positive number")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be a positive number")

    body = json.dumps(
        {"filter": _scope_filter(scope_filter), "exact": True},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    endpoint = f"{base_url}/collections/{quote(collection_name, safe='')}/points/count"
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    api_key = os.environ.get(api_key_env)
    if api_key:
        headers["api-key"] = api_key
    request = Request(endpoint, data=body, headers=headers, method="POST")
    open_request = opener or urlopen
    try:
        response = open_request(request, timeout=float(timeout_seconds))
        with response:
            status = getattr(response, "status", 200)
            if status == 404:
                return QdrantCollectionState(exists=False, point_count=None)
            if status != 200:
                raise QdrantProtocolError(
                    f"Qdrant count returned unexpected HTTP status {status}"
                )
            raw = response.read(MAX_QDRANT_RESPONSE_BYTES + 1)
    except HTTPError as exc:
        if exc.code == 404:
            return QdrantCollectionState(exists=False, point_count=None)
        if 500 <= exc.code <= 599:
            raise QdrantUnavailableError(
                f"Qdrant count service returned HTTP {exc.code}"
            ) from exc
        raise QdrantProtocolError(
            f"Qdrant count request was rejected with HTTP {exc.code}"
        ) from exc
    except (TimeoutError, URLError, OSError) as exc:
        raise QdrantUnavailableError("Qdrant count service is unreachable") from exc

    if len(raw) > MAX_QDRANT_RESPONSE_BYTES:
        raise QdrantProtocolError("Qdrant count response exceeds the size limit")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise QdrantProtocolError("Qdrant count response is not valid JSON") from exc
    result = payload.get("result") if isinstance(payload, dict) else None
    count = result.get("count") if isinstance(result, dict) else None
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise QdrantProtocolError("Qdrant count response has no valid point count")
    return QdrantCollectionState(exists=True, point_count=count)


def valid_lexical_index(directory: str | Path) -> bool:
    artifact = Path(directory) / "bm25_index.pkl"
    return (
        not artifact.is_symlink()
        and artifact.is_file()
        and artifact.stat().st_size > 0
    )


def remove_artifact_path(path: str | Path) -> None:
    """Remove an already-contained failed artifact regardless of file type."""

    target = Path(path)
    if target.is_symlink():
        raise ValueError(f"Refusing to remove an artifact through a symlink: {target}")
    if target.is_dir():
        shutil.rmtree(target)
    elif target.exists():
        target.unlink()
