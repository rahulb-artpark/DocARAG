"""Strict, dependency-light profiles for opt-in routed PDF ingestion.

This module is deliberately separate from :mod:`docarag.settings`: merely
loading a profile has no side effects. The supported ``docarag index`` command
selects one only when ``--routed-profile`` is supplied, binds its exact bytes
and normalized digest into the run revision, and passes ``profile.config`` to
the runtime composition root.

The TOML file contains only public configuration and environment-variable
*names*.  Resolved bearer values never enter this DTO, its hashes, logs, or
revision binding.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import re
from types import MappingProxyType
import tomllib
from typing import Callable, TypeVar

from docarag.domain.parsing import ParseOptions
from docarag.page_probe import PdfProbePolicy
from docarag.parsing_orchestrator import ParsingOrchestrationPolicy
from docarag.presets import MINERU_ADAPTER
from docarag.reconciliation import ReconciliationPolicy
from docarag.routed_ingestion import RoutedIngestionPolicy
from docarag.runtime_composition import (
    BalancedRoutedRuntimeConfig,
    ParserWorkerEndpointConfig,
)


ROUTED_PROFILE_SCHEMA_VERSION = "docarag.routed-runtime-profile.v1"
MAX_ROUTED_PROFILE_BYTES = 256 * 1024

_PROFILE_ID = re.compile(r"[a-z0-9](?:[a-z0-9._-]{0,126}[a-z0-9])?")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_ZERO_SHA256 = "0" * 64
_UNQUALIFIED_ADAPTER_VERSIONS = frozenset(
    {
        "changeme",
        "dev",
        "latest",
        "main",
        "master",
        "replace-with-qualified-version",
        "todo",
        "unknown",
        "unqualified",
    }
)


def _is_unqualified_adapter_version(value: str) -> bool:
    normalized = value.strip().casefold()
    return (
        normalized in _UNQUALIFIED_ADAPTER_VERSIONS
        or normalized.startswith(("placeholder", "replace-", "todo-"))
        or (normalized.startswith("<") and normalized.endswith(">"))
    )
_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema_version",
        "profile_id",
        "runtime",
        "workers",
        "parse_options",
        "page_probe",
        "ingestion",
        "orchestration",
        "reconciliation",
    }
)
_RUNTIME_FIELDS = frozenset(
    {
        "max_primary_latency_ms",
        "max_total_latency_ms",
        "max_primary_cost_microusd",
        "max_total_cost_microusd",
    }
)
_WORKER_FIELDS = frozenset(
    {
        "runtime_id",
        "adapter_name",
        "base_url",
        "expected_adapter_version",
        "expected_manifest_sha256",
        "bearer_token_env",
        "connect_timeout_seconds",
        "operation_timeout_seconds",
        "max_request_bytes",
        "max_response_bytes",
        "allow_insecure_http",
        "tls_ca_file",
        "estimated_latency_ms",
        "estimated_cost_microusd",
    }
)
_PARSE_OPTION_FIELDS = frozenset(
    {
        "languages",
        "ocr_mode",
        "parse_tables",
        "parse_formulas",
        "parse_images",
        "mineru_effort",
        "timeout_seconds",
        "extra",
    }
)
_PAGE_PROBE_FIELDS = frozenset(
    {
        "sha256_chunk_bytes",
        "infer_scanned_pages",
        "scanned_max_native_characters",
        "scanned_min_raster_coverage_ratio",
    }
)
_INGESTION_FIELDS = frozenset(
    {
        "max_page_features",
        "max_audit_bytes",
        "max_canonical_artifact_bytes",
        "sha256_chunk_bytes",
        "probe_identity",
    }
)
_ORCHESTRATION_FIELDS = frozenset(
    {
        "max_batch_pages",
        "minimum_observations_per_page",
        "minimum_explicit_confidence",
        "require_expected_block_types",
        "review_degraded_canonical",
        "review_reconciliation_issues",
    }
)
_RECONCILIATION_FIELDS = frozenset(
    {
        "minimum_iou",
        "high_iou",
        "minimum_text_similarity",
        "cross_type_iou",
        "cross_type_text_similarity",
        "accepted_confidence",
        "grid_size",
        "max_candidates_per_observation",
        "max_raw_candidate_probes_per_observation",
        "parser_quality_weights",
        "parser_priority",
    }
)
_SECRET_FIELD_NAMES = frozenset(
    {
        "access_token",
        "api_key",
        "apikey",
        "authorization",
        "bearer_token",
        "client_secret",
        "credential",
        "credentials",
        "password",
        "private_key",
        "refresh_token",
        "secret",
        "token",
    }
)


class RoutedProfileError(ValueError):
    """Base failure while reading or validating a routed runtime profile."""


class RoutedProfileReadError(RoutedProfileError):
    """The selected profile could not be read as one stable bounded file."""


class RoutedProfileSyntaxError(RoutedProfileError):
    """The profile is not valid UTF-8 TOML."""


class RoutedProfileValidationError(RoutedProfileError):
    """The TOML shape or a composed runtime contract is invalid."""


@dataclass(frozen=True, slots=True)
class RoutedProfileConfigurationIssue:
    """One static reason a valid profile must not be used to build a runtime."""

    code: str
    message: str
    runtime_id: str
    field: str

    def to_dict(self) -> dict[str, str]:
        return {
            "code": self.code,
            "message": self.message,
            "runtime_id": self.runtime_id,
            "field": self.field,
        }


def _normalized_configuration_bytes(
    profile_id: str, config: BalancedRoutedRuntimeConfig
) -> bytes:
    normalized = {
        "schema_version": ROUTED_PROFILE_SCHEMA_VERSION,
        "profile_id": profile_id,
        "runtime": config.to_dict(),
    }
    try:
        return json.dumps(
            normalized,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:  # defensive; DTOs should be JSON-safe
        raise RoutedProfileValidationError(
            f"normalized routed profile is not deterministic JSON: {exc}"
        ) from exc


@dataclass(frozen=True, slots=True)
class RoutedRuntimeProfile:
    """One immutable, secret-free runtime profile and its revision identity."""

    profile_id: str
    source_path: Path
    source_sha256: str
    configuration_sha256: str
    config: BalancedRoutedRuntimeConfig
    _source_bytes: bytes = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.profile_id, str)
            or _PROFILE_ID.fullmatch(self.profile_id) is None
        ):
            raise RoutedProfileValidationError("profile_id is invalid")
        if not isinstance(self.source_path, Path) or not self.source_path.is_absolute():
            raise RoutedProfileValidationError("source_path must be an absolute Path")
        if not isinstance(self.config, BalancedRoutedRuntimeConfig):
            raise RoutedProfileValidationError(
                "config must be BalancedRoutedRuntimeConfig"
            )
        if not isinstance(self._source_bytes, bytes):
            raise RoutedProfileValidationError("source_bytes must be bytes")
        if _SHA256.fullmatch(self.source_sha256) is None or self.source_sha256 != (
            hashlib.sha256(self._source_bytes).hexdigest()
        ):
            raise RoutedProfileValidationError("source_sha256 does not match source_bytes")
        expected_configuration_sha256 = hashlib.sha256(
            _normalized_configuration_bytes(self.profile_id, self.config)
        ).hexdigest()
        if (
            _SHA256.fullmatch(self.configuration_sha256) is None
            or self.configuration_sha256 != expected_configuration_sha256
        ):
            raise RoutedProfileValidationError(
                "configuration_sha256 does not match normalized runtime configuration"
            )

    @property
    def source_bytes(self) -> bytes:
        """Exact TOML bytes to persist as the immutable run input snapshot."""

        return self._source_bytes

    def revision_binding(self) -> Mapping[str, str]:
        """Return the minimum identity fields every routed run must bind.

        ``source_sha256`` proves the exact operator-authored profile, including
        comments. ``configuration_sha256`` proves normalized runtime semantics
        independently of TOML ordering.  Neither digest includes secret values.
        """

        return MappingProxyType(
            {
                "schema_version": ROUTED_PROFILE_SCHEMA_VERSION,
                "profile_id": self.profile_id,
                "source_sha256": self.source_sha256,
                "configuration_sha256": self.configuration_sha256,
            }
        )

    def to_dict(self) -> dict[str, object]:
        """Return inspectable normalized configuration without resolved secrets."""

        return {
            **dict(self.revision_binding()),
            "source_path": str(self.source_path),
            "runtime": self.config.to_dict(),
        }

    def configuration_issues(self) -> tuple[RoutedProfileConfigurationIssue, ...]:
        """Return static blockers without contacting workers or reading secrets.

        Structural validation and operational readiness are intentionally
        different.  In particular, the checked-in example is valid TOML but
        contains conspicuous pins that must never be mistaken for a deployable
        runtime configuration.
        """

        issues: list[RoutedProfileConfigurationIssue] = []
        for endpoint in self.config.endpoints:
            if _is_unqualified_adapter_version(endpoint.expected_adapter_version):
                issues.append(
                    RoutedProfileConfigurationIssue(
                        code="unqualified_adapter_version",
                        message=(
                            "expected_adapter_version is a placeholder or mutable "
                            "label; replace it with an exact qualified worker version"
                        ),
                        runtime_id=endpoint.runtime_id,
                        field="expected_adapter_version",
                    )
                )
            if endpoint.expected_manifest_sha256 == _ZERO_SHA256:
                issues.append(
                    RoutedProfileConfigurationIssue(
                        code="placeholder_manifest_sha256",
                        message=(
                            "expected_manifest_sha256 is the all-zero template value; "
                            "replace it with the exact qualified capability-manifest digest"
                        ),
                        runtime_id=endpoint.runtime_id,
                        field="expected_manifest_sha256",
                    )
                )
            if (
                endpoint.adapter_name == MINERU_ADAPTER
                and self.config.parse_options.mineru_effort is None
            ):
                issues.append(
                    RoutedProfileConfigurationIssue(
                        code="mineru_effort_required",
                        message=(
                            "MinerU 3.x effort must be explicitly set to medium or "
                            "high so latency/accuracy behavior enters request identity"
                        ),
                        runtime_id=endpoint.runtime_id,
                        field="parse_options.mineru_effort",
                    )
                )
        return tuple(issues)

    @property
    def configuration_ready(self) -> bool:
        """Whether static pins are usable for a later live readiness check."""

        return not self.configuration_issues()


def _mapping(value: object, scope: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise RoutedProfileValidationError(f"{scope} must be a TOML table")
    if any(not isinstance(key, str) for key in value):
        raise RoutedProfileValidationError(f"{scope} keys must be strings")
    return value  # type: ignore[return-value]


def _table(root: Mapping[str, object], key: str) -> Mapping[str, object]:
    return _mapping(root.get(key, {}), f"[{key}]")


def _reject_unknown(
    value: Mapping[str, object], allowed: frozenset[str], *, scope: str
) -> None:
    unknown = sorted(key for key in value if key not in allowed)
    if unknown:
        raise RoutedProfileValidationError(
            f"unknown routed profile field(s) in {scope}: {', '.join(unknown)}"
        )


def _normalized_secret_name(key: str) -> str:
    # Split both ordinary camelCase (``accessToken``) and acronym boundaries
    # (``APIKey``) before applying the separator-independent comparison.  A
    # lower-case-only normalization would turn those into ``accesstoken`` and
    # ``apikey`` and could miss a future deny-list entry written in snake_case.
    value = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", "_", key.strip())
    value = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", value)
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _assert_secret_references_only(value: object, *, path: str = "profile") -> None:
    """Reject likely literal-secret fields while allowing ``bearer_token_env``."""

    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            key = str(raw_key)
            normalized = _normalized_secret_name(key)
            if normalized != "bearer_token_env" and (
                normalized in _SECRET_FIELD_NAMES
                or normalized.endswith("_password")
                or normalized.endswith("_secret")
                or normalized.endswith("_token")
                or normalized.endswith("_api_key")
            ):
                raise RoutedProfileValidationError(
                    f"literal secret field is forbidden at {path}.{key}; "
                    "use a supported *_env reference"
                )
            _assert_secret_references_only(item, path=f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        for index, item in enumerate(value):
            _assert_secret_references_only(item, path=f"{path}[{index}]")
    elif isinstance(value, str) and "-----BEGIN PRIVATE KEY-----" in value:
        raise RoutedProfileValidationError(
            f"literal private key material is forbidden at {path}"
        )


T = TypeVar("T")


def _construct(
    factory: Callable[..., T],
    values: Mapping[str, object],
    allowed: frozenset[str],
    *,
    scope: str,
    transforms: Mapping[str, Callable[[object], object]] = MappingProxyType({}),
) -> T:
    _reject_unknown(values, allowed, scope=scope)
    kwargs = dict(values)
    for key, transform in transforms.items():
        if key in kwargs:
            kwargs[key] = transform(kwargs[key])
    try:
        return factory(**kwargs)
    except (TypeError, ValueError) as exc:
        raise RoutedProfileValidationError(f"invalid {scope}: {exc}") from exc


def _tuple(value: object) -> tuple[object, ...]:
    if not isinstance(value, list):
        raise RoutedProfileValidationError("value must be a TOML array")
    return tuple(value)


def _parse_profile(
    payload: Mapping[str, object], *, source_path: Path, source_bytes: bytes
) -> RoutedRuntimeProfile:
    _assert_secret_references_only(payload)
    _reject_unknown(payload, _TOP_LEVEL_FIELDS, scope="the root table")
    version = payload.get("schema_version")
    if version != ROUTED_PROFILE_SCHEMA_VERSION:
        raise RoutedProfileValidationError(
            f"unsupported routed profile schema {version!r}; "
            f"expected {ROUTED_PROFILE_SCHEMA_VERSION!r}"
        )
    profile_id = payload.get("profile_id")
    if not isinstance(profile_id, str) or _PROFILE_ID.fullmatch(profile_id) is None:
        raise RoutedProfileValidationError(
            "profile_id must contain lowercase letters, digits, '.', '_', or '-'"
        )

    workers_value = payload.get("workers")
    if not isinstance(workers_value, list):
        raise RoutedProfileValidationError(
            "workers must be a TOML array of tables ([[workers]])"
        )
    workers = tuple(
        _construct(
            ParserWorkerEndpointConfig,
            _mapping(item, f"[[workers]][{index}]"),
            _WORKER_FIELDS,
            scope=f"[[workers]][{index}]",
        )
        for index, item in enumerate(workers_value)
    )

    parse_option_values = _table(payload, "parse_options")
    extra_options = parse_option_values.get("extra")
    if extra_options not in (None, {}):
        # ParseOptions supports programmatic vendor extensions, but a profile
        # that is normalized, hashed, and printable by the CLI cannot safely
        # accept arbitrary key/value payloads.  A value such as
        # ``{headers = "Bearer ..."}`` bypasses field-name secret checks and
        # would be copied into diagnostics.  Future file-based extensions must
        # use an explicit typed schema with *_env references.
        raise RoutedProfileValidationError(
            "[parse_options].extra is not allowed in routed profile files; "
            "add a typed profile field with an environment-variable reference"
        )
    parse_options = _construct(
        ParseOptions,
        parse_option_values,
        _PARSE_OPTION_FIELDS,
        scope="[parse_options]",
        transforms={"languages": _tuple},
    )
    page_probe_policy = _construct(
        PdfProbePolicy,
        _table(payload, "page_probe"),
        _PAGE_PROBE_FIELDS,
        scope="[page_probe]",
    )
    ingestion_policy = _construct(
        RoutedIngestionPolicy,
        _table(payload, "ingestion"),
        _INGESTION_FIELDS,
        scope="[ingestion]",
    )
    orchestration_policy = _construct(
        ParsingOrchestrationPolicy,
        _table(payload, "orchestration"),
        _ORCHESTRATION_FIELDS,
        scope="[orchestration]",
    )
    reconciliation_policy = _construct(
        ReconciliationPolicy,
        _table(payload, "reconciliation"),
        _RECONCILIATION_FIELDS,
        scope="[reconciliation]",
        transforms={"parser_priority": _tuple},
    )
    runtime = _table(payload, "runtime")
    _reject_unknown(runtime, _RUNTIME_FIELDS, scope="[runtime]")
    try:
        config = BalancedRoutedRuntimeConfig(
            endpoints=workers,
            parse_options=parse_options,
            page_probe_policy=page_probe_policy,
            ingestion_policy=ingestion_policy,
            orchestration_policy=orchestration_policy,
            reconciliation_policy=reconciliation_policy,
            **dict(runtime),
        )
    except (TypeError, ValueError) as exc:
        raise RoutedProfileValidationError(
            f"invalid routed runtime composition: {exc}"
        ) from exc

    canonical_bytes = _normalized_configuration_bytes(profile_id, config)
    return RoutedRuntimeProfile(
        profile_id=profile_id,
        source_path=source_path,
        source_sha256=hashlib.sha256(source_bytes).hexdigest(),
        configuration_sha256=hashlib.sha256(canonical_bytes).hexdigest(),
        config=config,
        _source_bytes=source_bytes,
    )


def _read_stable_bounded_file(source: Path, *, max_bytes: int) -> bytes:
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 1:
        raise ValueError("max_bytes must be a positive integer")
    try:
        with source.open("rb") as handle:
            before = os.fstat(handle.fileno())
            payload = handle.read(max_bytes + 1)
            after = os.fstat(handle.fileno())
    except OSError as exc:
        raise RoutedProfileReadError(
            f"routed profile could not be read: {source}: {exc}"
        ) from exc
    if len(payload) > max_bytes:
        raise RoutedProfileReadError(
            f"routed profile exceeds the {max_bytes}-byte limit: {source}"
        )
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if identity_before != identity_after or len(payload) != after.st_size:
        raise RoutedProfileReadError(
            f"routed profile changed while it was being read: {source}"
        )
    return payload


def load_routed_runtime_profile(
    path: str | os.PathLike[str],
    *,
    max_bytes: int = MAX_ROUTED_PROFILE_BYTES,
) -> RoutedRuntimeProfile:
    """Load one explicit profile; there is intentionally no implicit discovery."""

    source = Path(path).expanduser()
    try:
        source = source.resolve(strict=True)
    except OSError as exc:
        raise RoutedProfileReadError(
            f"routed profile could not be resolved: {source}: {exc}"
        ) from exc
    if not source.is_file():
        raise RoutedProfileReadError(f"routed profile is not a file: {source}")
    source_bytes = _read_stable_bounded_file(source, max_bytes=max_bytes)
    try:
        text = source_bytes.decode("utf-8")
        payload = tomllib.loads(text)
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise RoutedProfileSyntaxError(
            f"routed profile is not valid UTF-8 TOML: {source}: {exc}"
        ) from exc
    return _parse_profile(payload, source_path=source, source_bytes=source_bytes)


__all__ = [
    "MAX_ROUTED_PROFILE_BYTES",
    "ROUTED_PROFILE_SCHEMA_VERSION",
    "RoutedProfileError",
    "RoutedProfileReadError",
    "RoutedProfileSyntaxError",
    "RoutedProfileValidationError",
    "RoutedProfileConfigurationIssue",
    "RoutedRuntimeProfile",
    "load_routed_runtime_profile",
]
