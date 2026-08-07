"""Dependency-free primitives shared by DocARAG domain contracts.

The helpers in this module deliberately accept JSON values only.  That keeps
contract hashes portable across Python processes and future out-of-process
parser adapters.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import re
from types import MappingProxyType
from typing import Any, Self, TypeAlias


CONTRACT_SCHEMA_VERSION = "docarag.parser-contract.v1"

JsonScalar: TypeAlias = None | bool | int | float | str
JsonValue: TypeAlias = JsonScalar | tuple["JsonValue", ...] | Mapping[str, "JsonValue"]
JsonObject: TypeAlias = Mapping[str, JsonValue]

_SHA256 = re.compile(r"[0-9a-f]{64}")


class ContractValidationError(ValueError):
    """A DTO or adapter payload violates the parser contract."""


class UnsupportedSchemaVersionError(ContractValidationError):
    """A serialized DTO uses a schema this installation cannot read."""


class JsonDTOMixin:
    """Uniform JSON round-trip surface for versioned domain DTOs."""

    __slots__ = ()

    def to_dict(self) -> dict[str, object]:
        raise NotImplementedError

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> Self:
        raise NotImplementedError

    def to_json(self, *, indent: int | None = None) -> str:
        return dump_json(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, payload: str | bytes | bytearray) -> Self:
        return cls.from_dict(parse_json_object(payload))


def require_schema_version(payload: Mapping[str, object]) -> None:
    """Reject missing, older, and newer schemas instead of guessing."""

    actual = payload.get("schema_version")
    if actual != CONTRACT_SCHEMA_VERSION:
        raise UnsupportedSchemaVersionError(
            "unsupported parser contract schema_version "
            f"{actual!r}; expected {CONTRACT_SCHEMA_VERSION!r}"
        )


def require_non_empty(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractValidationError(f"{field} must be a non-empty string")
    return value


def require_sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ContractValidationError(f"{field} must be a lowercase SHA-256 hex digest")
    return value


def require_probability(value: float | None, field: str) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractValidationError(f"{field} must be a number between 0 and 1")
    if isinstance(value, int):
        if not 0 <= value <= 1:
            raise ContractValidationError(f"{field} must be between 0 and 1")
        return
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ContractValidationError(f"{field} must be between 0 and 1")


def freeze_json(value: object, *, field: str = "value") -> JsonValue:
    """Validate and recursively freeze a JSON-compatible value.

    Copying into tuples and mapping proxies prevents mutation after a stable
    content hash has been calculated.

    A NUL is dropped here, before any digest is taken over the value, because
    PostgreSQL's text and jsonb types both reject an embedded NUL outright and
    MinerU emits a literal NUL as its fallback for a glyph a source PDF's font
    has no usable Unicode mapping for (Greek letters and a degree sign, in
    Zernike-polynomial notation). Something has to remove it before the value
    can be stored, and the only safe moment is before it is hashed: dropping it
    downstream instead left six blocks of a 1871-page document carrying a
    content_sha256 for text the store no longer held, and every query against
    that document failed on "Representation content_sha256 mismatch" -- a whole
    index lost to a character that was never storable in the first place.
    """

    if isinstance(value, str):
        return value.replace("\x00", "") if "\x00" in value else value
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ContractValidationError(f"{field} contains a non-finite float")
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, JsonValue] = {}
        if any(not isinstance(key, str) for key in value):
            raise ContractValidationError(f"{field} contains a non-string key")
        for key in sorted(value):
            if not isinstance(key, str):
                raise ContractValidationError(f"{field} contains a non-string key")
            frozen[key] = freeze_json(value[key], field=f"{field}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(
            freeze_json(item, field=f"{field}[{index}]")
            for index, item in enumerate(value)
        )
    raise ContractValidationError(
        f"{field} must contain JSON-compatible values, got {type(value).__name__}"
    )


def json_ready(value: object) -> object:
    """Convert frozen domain data into ordinary JSON containers."""

    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [json_ready(item) for item in value]
    return value


def canonical_json_bytes(value: object) -> bytes:
    """Serialize a JSON value deterministically for IDs and hashes."""

    frozen = freeze_json(value)
    return json.dumps(
        json_ready(frozen),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def content_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def stable_id(kind: str, *identity_parts: object) -> str:
    """Create a compact deterministic ID from an explicitly stable identity."""

    require_non_empty(kind, "kind")
    digest = content_sha256(
        {"kind": kind, "schema_version": CONTRACT_SCHEMA_VERSION, "parts": identity_parts}
    )
    return f"{kind}_{digest[:32]}"


def parse_json_object(payload: str | bytes | bytearray) -> dict[str, Any]:
    try:
        parsed = json.loads(payload)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ContractValidationError(f"invalid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ContractValidationError("serialized DTO must be a JSON object")
    return parsed


def dump_json(payload: Mapping[str, object], *, indent: int | None = None) -> str:
    return json.dumps(
        json_ready(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":") if indent is None else None,
        indent=indent,
        allow_nan=False,
    )


def reject_unknown_fields(
    payload: Mapping[str, object], allowed: set[str], *, dto: str
) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ContractValidationError(f"{dto} has unknown field(s): {', '.join(unknown)}")


def require_mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ContractValidationError(f"{field} must be a JSON object")
    return value


def require_sequence(value: object, field: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ContractValidationError(f"{field} must be a JSON array")
    return value
