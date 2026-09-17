"""Standard-library runtime compatibility contracts for DocARAG deployments.

The controller must be able to inspect a deployment before importing Docling,
MinerU, PaddlePaddle, Torch, Transformers, or vLLM.  This module therefore has
no dependencies outside the Python standard library.  It validates a checked-in
TOML/JSON compatibility declaration and the small JSON capability document
served by an out-of-process parser worker.

Compatibility and verification are deliberately separate concepts.  A worker
can speak the correct protocol and adapter contract while its vendor package,
model, image, or VRAM matrix is still only fixture-tested.  Such a worker is
reported as compatible-but-uncertified, never as production verified.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from pathlib import Path
import re
import tomllib
from types import MappingProxyType
from typing import Any, Mapping, Sequence
from urllib.parse import urljoin, urlsplit

from document_services.domain.base import ContractValidationError
from document_services.domain.parsing import ParserCapabilityManifest


MANIFEST_SCHEMA_VERSION = "docarag.runtime-compatibility.v1"
PARSER_CONTRACT_SCHEMA_VERSION = "docarag.parser-contract.v1"
PARSER_WORKER_PROTOCOL_VERSION = "docarag.parser-worker-http.v1"
DEFAULT_MANIFEST_FILENAME = "runtime_compatibility.toml"
DEFAULT_MANIFEST_PATH = Path(__file__).with_name(DEFAULT_MANIFEST_FILENAME)

_SHA256 = re.compile(r"[0-9a-f]{64}")
_RUNTIME_ID = re.compile(r"[a-z0-9](?:[a-z0-9._-]{0,126}[a-z0-9])?")
_ALLOWED_TOP_LEVEL_FIELDS = frozenset({"schema_version", "manifest_revision", "runtimes"})
_ALLOWED_RUNTIME_FIELDS = frozenset(
    {
        "runtime_id",
        "runtime_kind",
        "contract_schema",
        "worker_protocol",
        "adapter_name",
        "adapter_versions",
        "parser_name",
        "accelerator",
        "minimum_vram_mib",
        "minimum_vram_certified",
        "health_location",
        "capability_location",
        "verification_level",
        "verification_notes",
        "package_version_constraints",
        "model_version_constraints",
        "image_version_constraints",
    }
)
_ALLOWED_CONSTRAINT_FIELDS = frozenset(
    {"name", "allowed_versions", "certified", "notes"}
)


class CompatibilityManifestError(ValueError):
    """A runtime compatibility declaration is missing required semantics."""


class RuntimeKind(str, Enum):
    CONTROLLER = "controller"
    PARSER_WORKER = "parser-worker"


class AcceleratorRequirement(str, Enum):
    ANY = "any"
    CPU = "cpu"
    NVIDIA_CUDA = "nvidia-cuda"
    REMOTE = "remote"


class VerificationLevel(str, Enum):
    FIXTURE_ONLY = "fixture-only"
    REAL_MODEL_TESTED = "real-model-tested"


def _nonempty(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CompatibilityManifestError(f"{field} must be a non-empty string")
    return value.strip()


def _exact_integer(value: object, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise CompatibilityManifestError(
            f"{field} must be an integer greater than or equal to {minimum}"
        )
    return value


def _boolean(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise CompatibilityManifestError(f"{field} must be a boolean")
    return value


def _mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise CompatibilityManifestError(f"{field} must be an object/table")
    if any(not isinstance(key, str) for key in value):
        raise CompatibilityManifestError(f"{field} keys must be strings")
    return value  # type: ignore[return-value]


def _sequence(value: object, field: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise CompatibilityManifestError(f"{field} must be an array")
    return value


def _enum(enum_type: type[Enum], value: object, field: str) -> Any:
    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(repr(item.value) for item in enum_type)
        raise CompatibilityManifestError(f"{field} must be one of {allowed}") from exc


def _reject_unknown_fields(
    value: Mapping[str, object], allowed: frozenset[str], field: str
) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise CompatibilityManifestError(
            f"{field} contains unknown field(s): {', '.join(unknown)}"
        )


def _strings(value: object, field: str) -> tuple[str, ...]:
    result = tuple(_nonempty(item, f"{field}[]") for item in _sequence(value, field))
    if len(result) != len(set(result)):
        raise CompatibilityManifestError(f"{field} must not contain duplicates")
    return result


def _relative_http_location(value: object, field: str) -> str:
    location = _nonempty(value, field)
    parsed = urlsplit(location)
    if (
        not location.startswith("/")
        or location.startswith("//")
        or parsed.scheme
        or parsed.netloc
        or parsed.query
        or parsed.fragment
    ):
        raise CompatibilityManifestError(
            f"{field} must be an absolute HTTP path without host, query, or fragment"
        )
    return location


@dataclass(frozen=True, slots=True)
class ArtifactVersionConstraint:
    """An exact allow-list, or an explicit statement that no version is certified."""

    name: str
    allowed_versions: tuple[str, ...]
    certified: bool
    notes: str

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, object], *, field: str
    ) -> "ArtifactVersionConstraint":
        _reject_unknown_fields(payload, _ALLOWED_CONSTRAINT_FIELDS, field)
        name = _nonempty(payload.get("name"), f"{field}.name")
        allowed = _strings(
            payload.get("allowed_versions", ()), f"{field}.allowed_versions"
        )
        certified = _boolean(payload.get("certified"), f"{field}.certified")
        notes = _nonempty(payload.get("notes"), f"{field}.notes")
        if certified and not allowed:
            raise CompatibilityManifestError(
                f"{field}.allowed_versions must not be empty when certified is true"
            )
        if not certified and allowed:
            raise CompatibilityManifestError(
                f"{field}.allowed_versions must be empty when certified is false; "
                "put observations in notes instead of implying support"
            )
        return cls(name, allowed, certified, notes)

    def accepts(self, version: str) -> bool | None:
        """Return ``None`` when this artifact has no certified version matrix."""

        if not self.certified:
            return None
        return version in self.allowed_versions

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "allowed_versions": list(self.allowed_versions),
            "certified": self.certified,
            "notes": self.notes,
        }


def _constraints(value: object, field: str) -> tuple[ArtifactVersionConstraint, ...]:
    constraints = tuple(
        ArtifactVersionConstraint.from_dict(
            _mapping(item, f"{field}[{index}]"), field=f"{field}[{index}]"
        )
        for index, item in enumerate(_sequence(value, field))
    )
    names = [item.name for item in constraints]
    if len(names) != len(set(names)):
        raise CompatibilityManifestError(f"{field} artifact names must be unique")
    return constraints


@dataclass(frozen=True, slots=True)
class RuntimeCompatibility:
    runtime_id: str
    runtime_kind: RuntimeKind
    contract_schema: str
    worker_protocol: str | None
    adapter_name: str | None
    adapter_versions: tuple[str, ...]
    parser_name: str | None
    accelerator: AcceleratorRequirement
    minimum_vram_mib: int
    minimum_vram_certified: bool
    health_location: str | None
    capability_location: str | None
    verification_level: VerificationLevel
    verification_notes: str
    package_version_constraints: tuple[ArtifactVersionConstraint, ...]
    model_version_constraints: tuple[ArtifactVersionConstraint, ...]
    image_version_constraints: tuple[ArtifactVersionConstraint, ...]

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, object], *, field: str
    ) -> "RuntimeCompatibility":
        _reject_unknown_fields(payload, _ALLOWED_RUNTIME_FIELDS, field)
        runtime_id = _nonempty(payload.get("runtime_id"), f"{field}.runtime_id")
        if _RUNTIME_ID.fullmatch(runtime_id) is None:
            raise CompatibilityManifestError(
                f"{field}.runtime_id must contain lowercase letters, digits, '.', '_', or '-'"
            )
        runtime_kind = _enum(
            RuntimeKind, payload.get("runtime_kind"), f"{field}.runtime_kind"
        )
        contract_schema = _nonempty(
            payload.get("contract_schema"), f"{field}.contract_schema"
        )
        if contract_schema != PARSER_CONTRACT_SCHEMA_VERSION:
            raise CompatibilityManifestError(
                f"{field}.contract_schema {contract_schema!r} is not supported by this "
                f"controller ({PARSER_CONTRACT_SCHEMA_VERSION!r})"
            )
        accelerator = _enum(
            AcceleratorRequirement,
            payload.get("accelerator"),
            f"{field}.accelerator",
        )
        minimum_vram_mib = _exact_integer(
            payload.get("minimum_vram_mib"), f"{field}.minimum_vram_mib"
        )
        minimum_vram_certified = _boolean(
            payload.get("minimum_vram_certified"),
            f"{field}.minimum_vram_certified",
        )
        if not minimum_vram_certified and minimum_vram_mib != 0:
            raise CompatibilityManifestError(
                f"{field}.minimum_vram_mib must be 0 until the minimum is certified"
            )
        verification_level = _enum(
            VerificationLevel,
            payload.get("verification_level"),
            f"{field}.verification_level",
        )
        verification_notes = _nonempty(
            payload.get("verification_notes"), f"{field}.verification_notes"
        )
        package_constraints = _constraints(
            payload.get("package_version_constraints", ()),
            f"{field}.package_version_constraints",
        )
        model_constraints = _constraints(
            payload.get("model_version_constraints", ()),
            f"{field}.model_version_constraints",
        )
        image_constraints = _constraints(
            payload.get("image_version_constraints", ()),
            f"{field}.image_version_constraints",
        )

        if runtime_kind is RuntimeKind.PARSER_WORKER:
            worker_protocol = _nonempty(
                payload.get("worker_protocol"), f"{field}.worker_protocol"
            )
            if worker_protocol != PARSER_WORKER_PROTOCOL_VERSION:
                raise CompatibilityManifestError(
                    f"{field}.worker_protocol {worker_protocol!r} is unsupported; "
                    f"expected {PARSER_WORKER_PROTOCOL_VERSION!r}"
                )
            adapter_name = _nonempty(
                payload.get("adapter_name"), f"{field}.adapter_name"
            )
            adapter_versions = _strings(
                payload.get("adapter_versions"), f"{field}.adapter_versions"
            )
            if not adapter_versions:
                raise CompatibilityManifestError(
                    f"{field}.adapter_versions must not be empty"
                )
            parser_name = _nonempty(
                payload.get("parser_name"), f"{field}.parser_name"
            )
            health_location = _relative_http_location(
                payload.get("health_location"), f"{field}.health_location"
            )
            capability_location = _relative_http_location(
                payload.get("capability_location"),
                f"{field}.capability_location",
            )
            if health_location == capability_location:
                raise CompatibilityManifestError(
                    f"{field} health and capability locations must be different"
                )
            if not package_constraints or not model_constraints or not image_constraints:
                raise CompatibilityManifestError(
                    f"{field} must explicitly record package, model, and image version constraints"
                )
        else:
            forbidden = {
                key
                for key in (
                    "worker_protocol",
                    "adapter_name",
                    "adapter_versions",
                    "parser_name",
                    "health_location",
                    "capability_location",
                )
                if key in payload
            }
            if forbidden:
                raise CompatibilityManifestError(
                    f"{field} controller runtime must not declare parser-worker field(s): "
                    f"{', '.join(sorted(forbidden))}"
                )
            worker_protocol = None
            adapter_name = None
            adapter_versions = ()
            parser_name = None
            health_location = None
            capability_location = None

        return cls(
            runtime_id=runtime_id,
            runtime_kind=runtime_kind,
            contract_schema=contract_schema,
            worker_protocol=worker_protocol,
            adapter_name=adapter_name,
            adapter_versions=adapter_versions,
            parser_name=parser_name,
            accelerator=accelerator,
            minimum_vram_mib=minimum_vram_mib,
            minimum_vram_certified=minimum_vram_certified,
            health_location=health_location,
            capability_location=capability_location,
            verification_level=verification_level,
            verification_notes=verification_notes,
            package_version_constraints=package_constraints,
            model_version_constraints=model_constraints,
            image_version_constraints=image_constraints,
        )

    @property
    def all_artifact_constraints(self) -> tuple[ArtifactVersionConstraint, ...]:
        return (
            *self.package_version_constraints,
            *self.model_version_constraints,
            *self.image_version_constraints,
        )

    @property
    def is_production_verified(self) -> bool:
        """Whether this exact runtime matrix has real-model and version evidence."""

        return (
            self.verification_level is VerificationLevel.REAL_MODEL_TESTED
            and self.minimum_vram_certified
            and all(item.certified for item in self.all_artifact_constraints)
        )

    @property
    def uncertified_artifacts(self) -> tuple[str, ...]:
        return tuple(
            item.name for item in self.all_artifact_constraints if not item.certified
        )

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "runtime_id": self.runtime_id,
            "runtime_kind": self.runtime_kind.value,
            "contract_schema": self.contract_schema,
            "accelerator": self.accelerator.value,
            "minimum_vram_mib": self.minimum_vram_mib,
            "minimum_vram_certified": self.minimum_vram_certified,
            "verification_level": self.verification_level.value,
            "verification_notes": self.verification_notes,
            "package_version_constraints": [
                item.to_dict() for item in self.package_version_constraints
            ],
            "model_version_constraints": [
                item.to_dict() for item in self.model_version_constraints
            ],
            "image_version_constraints": [
                item.to_dict() for item in self.image_version_constraints
            ],
        }
        if self.runtime_kind is RuntimeKind.PARSER_WORKER:
            result.update(
                {
                    "worker_protocol": self.worker_protocol,
                    "adapter_name": self.adapter_name,
                    "adapter_versions": list(self.adapter_versions),
                    "parser_name": self.parser_name,
                    "health_location": self.health_location,
                    "capability_location": self.capability_location,
                }
            )
        return result


@dataclass(frozen=True, slots=True)
class RuntimeCompatibilityManifest:
    schema_version: str
    manifest_revision: int
    runtimes: tuple[RuntimeCompatibility, ...]
    source_sha256: str

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, object]
    ) -> "RuntimeCompatibilityManifest":
        _reject_unknown_fields(payload, _ALLOWED_TOP_LEVEL_FIELDS, "manifest")
        schema_version = _nonempty(payload.get("schema_version"), "schema_version")
        if schema_version != MANIFEST_SCHEMA_VERSION:
            raise CompatibilityManifestError(
                f"unsupported runtime compatibility schema {schema_version!r}; "
                f"expected {MANIFEST_SCHEMA_VERSION!r}"
            )
        revision = _exact_integer(
            payload.get("manifest_revision"), "manifest_revision", minimum=1
        )
        runtimes = tuple(
            RuntimeCompatibility.from_dict(
                _mapping(item, f"runtimes[{index}]"), field=f"runtimes[{index}]"
            )
            for index, item in enumerate(_sequence(payload.get("runtimes"), "runtimes"))
        )
        if not runtimes:
            raise CompatibilityManifestError("runtimes must not be empty")
        ids = [item.runtime_id for item in runtimes]
        if len(ids) != len(set(ids)):
            raise CompatibilityManifestError("runtime_id values must be unique")
        controllers = [
            item for item in runtimes if item.runtime_kind is RuntimeKind.CONTROLLER
        ]
        if len(controllers) != 1:
            raise CompatibilityManifestError(
                "manifest must contain exactly one controller runtime"
            )
        adapter_names = [
            item.adapter_name
            for item in runtimes
            if item.runtime_kind is RuntimeKind.PARSER_WORKER
        ]
        if len(adapter_names) != len(set(adapter_names)):
            raise CompatibilityManifestError(
                "parser-worker adapter_name values must be unique"
            )
        canonical = {
            "schema_version": schema_version,
            "manifest_revision": revision,
            "runtimes": [item.to_dict() for item in runtimes],
        }
        digest = hashlib.sha256(
            json.dumps(
                canonical,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        return cls(schema_version, revision, runtimes, digest)

    @property
    def by_runtime_id(self) -> Mapping[str, RuntimeCompatibility]:
        return MappingProxyType({item.runtime_id: item for item in self.runtimes})

    @property
    def by_adapter_name(self) -> Mapping[str, RuntimeCompatibility]:
        return MappingProxyType(
            {
                item.adapter_name: item
                for item in self.runtimes
                if item.adapter_name is not None
            }
        )

    @property
    def parser_workers(self) -> tuple[RuntimeCompatibility, ...]:
        return tuple(
            item
            for item in self.runtimes
            if item.runtime_kind is RuntimeKind.PARSER_WORKER
        )

    def to_dict(self) -> dict[str, object]:
        """Return the portable manifest payload (the derived digest is omitted)."""

        return {
            "schema_version": self.schema_version,
            "manifest_revision": self.manifest_revision,
            "runtimes": [item.to_dict() for item in self.runtimes],
        }


def load_runtime_compatibility_manifest(
    path: str | Path = DEFAULT_MANIFEST_PATH,
) -> RuntimeCompatibilityManifest:
    """Load and strictly validate one TOML or JSON compatibility manifest."""

    source = Path(path).expanduser()
    try:
        raw = source.read_bytes()
    except OSError as exc:
        raise CompatibilityManifestError(
            f"runtime compatibility manifest could not be read: {source}: {exc}"
        ) from exc
    try:
        if source.suffix.casefold() == ".toml":
            payload = tomllib.loads(raw.decode("utf-8"))
        elif source.suffix.casefold() == ".json":
            payload = json.loads(raw)
        else:
            raise CompatibilityManifestError(
                "runtime compatibility manifest must use .toml or .json"
            )
    except (UnicodeDecodeError, tomllib.TOMLDecodeError, json.JSONDecodeError) as exc:
        raise CompatibilityManifestError(
            f"runtime compatibility manifest syntax is invalid: {source}: {exc}"
        ) from exc
    return RuntimeCompatibilityManifest.from_dict(_mapping(payload, "manifest"))


@dataclass(frozen=True, slots=True)
class WorkerCompatibilityResult:
    compatible: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    details: Mapping[str, object]


def worker_endpoint_url(base_url: str, location: str) -> str:
    """Resolve a declared relative worker location below one HTTP(S) origin."""

    if not isinstance(base_url, str) or not base_url.strip():
        raise CompatibilityManifestError(
            "worker base_url must be a non-empty string"
        )
    parsed = urlsplit(base_url)
    if parsed.scheme not in {"http", "https"} or parsed.hostname is None:
        raise CompatibilityManifestError(
            "worker base_url must be an absolute http:// or https:// URL"
        )
    if parsed.username is not None or parsed.password is not None:
        raise CompatibilityManifestError(
            "worker base_url must not contain user information"
        )
    if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        raise CompatibilityManifestError(
            "worker base_url must not contain a path, query, or fragment"
        )
    try:
        parsed.port
    except ValueError as exc:
        raise CompatibilityManifestError(
            "worker base_url contains an invalid port"
        ) from exc
    relative = _relative_http_location(location, "worker location")
    origin = f"{parsed.scheme}://{parsed.netloc}/"
    return urljoin(origin, relative.lstrip("/"))


def validate_worker_capability_payload(
    expectation: RuntimeCompatibility,
    payload: object,
    *,
    expected_manifest_sha256: str | None = None,
) -> WorkerCompatibilityResult:
    """Compare an HTTP worker capability response to a runtime expectation."""

    if expectation.runtime_kind is not RuntimeKind.PARSER_WORKER:
        raise CompatibilityManifestError(
            "capability payloads can only be checked against parser-worker runtimes"
        )
    if expected_manifest_sha256 is not None and _SHA256.fullmatch(
        expected_manifest_sha256
    ) is None:
        raise CompatibilityManifestError(
            "expected_manifest_sha256 must be a lowercase SHA-256 digest"
        )
    errors: list[str] = []
    warnings: list[str] = []
    details: dict[str, object] = {
        "runtime_id": expectation.runtime_id,
        "verification_level": expectation.verification_level.value,
        "production_verified": expectation.is_production_verified,
    }
    if not isinstance(payload, Mapping):
        return WorkerCompatibilityResult(
            False,
            ("capability response must be a JSON object",),
            (),
            details,
        )
    protocol = payload.get("protocol_version")
    details["protocol_version"] = protocol
    if protocol != expectation.worker_protocol:
        errors.append(
            f"worker protocol {protocol!r} does not match {expectation.worker_protocol!r}"
        )
    unknown_envelope_fields = sorted(
        key for key in payload if key not in {"protocol_version", "manifest"}
    )
    if unknown_envelope_fields:
        errors.append(
            "capability response contains unknown field(s): "
            + ", ".join(unknown_envelope_fields)
        )

    advertised = payload.get("manifest")
    manifest: ParserCapabilityManifest | None = None
    if not isinstance(advertised, Mapping):
        errors.append("capability response manifest must be a JSON object")
    else:
        # This is the same dependency-light DTO validation used by
        # HTTPParserAdapter.refresh_manifest().  It validates every required
        # capability field, rejects unknown fields, recomputes the manifest
        # digest and capability ID, and compares both advertised identities.
        try:
            manifest = ParserCapabilityManifest.from_dict(advertised)
        except ContractValidationError as exc:
            errors.append(f"worker capability manifest is invalid: {exc}")

    if manifest is not None:
        fields = {
            "schema_version": manifest.SCHEMA_VERSION,
            "capability_id": manifest.capability_id,
            "adapter_name": manifest.adapter_name,
            "adapter_version": manifest.adapter_version,
            "parser_name": manifest.parser_name,
            "parser_version": manifest.parser_version,
            "model_version": manifest.model_version,
            "manifest_sha256": manifest.manifest_sha256,
        }
        details.update(fields)
        if fields["schema_version"] != expectation.contract_schema:
            errors.append(
                f"parser schema {fields['schema_version']!r} does not match "
                f"{expectation.contract_schema!r}"
            )
        if fields["adapter_name"] != expectation.adapter_name:
            errors.append(
                f"adapter {fields['adapter_name']!r} does not match "
                f"{expectation.adapter_name!r}"
            )
        if fields["adapter_version"] not in expectation.adapter_versions:
            errors.append(
                f"adapter version {fields['adapter_version']!r} is not in the allowed "
                f"set {list(expectation.adapter_versions)!r}"
            )
        if fields["parser_name"] != expectation.parser_name:
            errors.append(
                f"parser {fields['parser_name']!r} does not match "
                f"{expectation.parser_name!r}"
            )
        if expected_manifest_sha256 is not None:
            if manifest.manifest_sha256 != expected_manifest_sha256:
                errors.append(
                    "worker manifest_sha256 differs from the configured pinned identity"
                )

    if expectation.verification_level is VerificationLevel.FIXTURE_ONLY:
        warnings.append(
            "runtime is fixture-only; no real-model compatibility claim has been made"
        )
    if expectation.uncertified_artifacts:
        warnings.append(
            "uncertified package/model/image versions: "
            + ", ".join(expectation.uncertified_artifacts)
        )
    if not expectation.minimum_vram_certified:
        warnings.append("minimum VRAM has not been certified")
    return WorkerCompatibilityResult(
        not errors,
        tuple(errors),
        tuple(warnings),
        MappingProxyType(details),
    )


def validate_worker_health_payload(
    expectation: RuntimeCompatibility, payload: object
) -> WorkerCompatibilityResult:
    """Validate worker readiness and bind health identity to the expectation."""

    if expectation.runtime_kind is not RuntimeKind.PARSER_WORKER:
        raise CompatibilityManifestError(
            "health payloads can only be checked against parser-worker runtimes"
        )
    details: dict[str, object] = {"runtime_id": expectation.runtime_id}
    if not isinstance(payload, Mapping):
        return WorkerCompatibilityResult(
            False, ("health response must be a JSON object",), (), details
        )
    fields = {
        "protocol_version": payload.get("protocol_version"),
        "state": payload.get("state"),
        "accepting_requests": payload.get("accepting_requests"),
        "adapter_name": payload.get("adapter_name"),
        "adapter_version": payload.get("adapter_version"),
        "manifest_sha256": payload.get("manifest_sha256"),
    }
    details.update(fields)
    errors: list[str] = []
    if fields["protocol_version"] != expectation.worker_protocol:
        errors.append("health response uses an incompatible worker protocol")
    if fields["state"] != "ready" or fields["accepting_requests"] is not True:
        errors.append("worker is not ready and accepting requests")
    if fields["adapter_name"] != expectation.adapter_name:
        errors.append("health response advertises a different adapter")
    if fields["adapter_version"] not in expectation.adapter_versions:
        errors.append("health response advertises an unsupported adapter version")
    manifest_sha256 = fields["manifest_sha256"]
    if not isinstance(manifest_sha256, str) or _SHA256.fullmatch(manifest_sha256) is None:
        errors.append("health response manifest_sha256 is missing or invalid")
    return WorkerCompatibilityResult(
        not errors, tuple(errors), (), MappingProxyType(details)
    )


__all__ = [
    "AcceleratorRequirement",
    "ArtifactVersionConstraint",
    "CompatibilityManifestError",
    "DEFAULT_MANIFEST_FILENAME",
    "DEFAULT_MANIFEST_PATH",
    "MANIFEST_SCHEMA_VERSION",
    "PARSER_CONTRACT_SCHEMA_VERSION",
    "PARSER_WORKER_PROTOCOL_VERSION",
    "RuntimeCompatibility",
    "RuntimeCompatibilityManifest",
    "RuntimeKind",
    "VerificationLevel",
    "WorkerCompatibilityResult",
    "load_runtime_compatibility_manifest",
    "validate_worker_capability_payload",
    "validate_worker_health_payload",
    "worker_endpoint_url",
]
