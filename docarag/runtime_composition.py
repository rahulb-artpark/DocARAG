"""Opt-in dependency-light composition root for routed PDF ingestion.

The legacy BookRAG/MinerU path remains untouched.  Deployments that explicitly
choose routed parsing can construct this runtime with three pinned HTTP worker
endpoints, then inject ``runtime.provider`` through Core's structural canonical
ingestion seam.

Importing this module loads no parser vendor, ML framework, model, or PyMuPDF.
The default PDF probe backend imports PyMuPDF lazily only when a PDF is probed.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
import os
import re
import threading
from typing import Protocol, runtime_checkable
from urllib.parse import urlsplit

from docarag.application import (
    DocumentExtractionPolicy,
    ExtractDocumentUseCase,
    RoutedDocumentExtractionBackend,
)
from docarag.adapters.pymupdf_probe import PyMuPdfProbeBackend
from docarag.composition import RoutedBookRAGFacade
from docarag.domain.base import ContractValidationError, require_non_empty, require_sha256
from docarag.domain.parsing import (
    ParseOptions,
    ParseRequest,
    ParseResult,
    ParserCapabilityManifest,
)
from docarag.page_probe import PdfProbeBackend, PdfProbePolicy, SourceBoundPdfPageProbe
from docarag.parsing_orchestrator import ParsingOrchestrationPolicy
from docarag.presets import (
    DOCLING_ADAPTER,
    MINERU_ADAPTER,
    PADDLEOCR_VL_ADAPTER,
)
from docarag.reconciliation import ReconciliationPolicy
from docarag.routed_ingestion import (
    RoutedCanonicalIngestionProvider,
    RoutedIngestionPolicy,
)
from docarag.transport import (
    HTTPParserAdapter,
    HTTPParserClientConfig,
    WorkerHealth,
    WorkerNotReadyError,
    WorkerState,
)


BALANCED_ADAPTER_IDENTITIES = frozenset(
    {DOCLING_ADAPTER, MINERU_ADAPTER, PADDLEOCR_VL_ADAPTER}
)
RUNTIME_COMPOSITION_SCHEMA_VERSION = "docarag.balanced-runtime-composition.v1"

_RUNTIME_ID = re.compile(r"[a-z0-9](?:[a-z0-9._-]{0,126}[a-z0-9])?")
_ENVIRONMENT_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


class RuntimeCompositionError(RuntimeError):
    """Base failure while constructing or owning the opt-in routed runtime."""


class RuntimeCompositionConfigurationError(RuntimeCompositionError, ValueError):
    """The explicit immutable endpoint/runtime configuration is invalid."""


class RuntimeCompositionIdentityError(RuntimeCompositionError):
    """A configured endpoint resolved to an unexpected or duplicate worker."""


class RuntimeCompositionClosedError(RuntimeCompositionError):
    """A caller requested the provider after its owned clients were closed."""


class RuntimeCompositionCloseError(RuntimeCompositionError):
    """One or more owned parser clients could not be closed."""


def _optional_non_negative_int(value: object, field_name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RuntimeCompositionConfigurationError(
            f"{field_name} must be a non-negative integer or null"
        )
    return value


def _canonical_endpoint(base_url: str) -> tuple[str, str, int]:
    parsed = urlsplit(base_url)
    hostname = parsed.hostname
    if hostname is None:  # HTTPParserClientConfig already validates this.
        raise RuntimeCompositionConfigurationError(
            "worker base_url must include a hostname"
        )
    return (
        parsed.scheme.lower(),
        hostname.lower(),
        parsed.port or (443 if parsed.scheme.lower() == "https" else 80),
    )


@dataclass(frozen=True, slots=True)
class ParserWorkerEndpointConfig:
    """One fully pinned worker endpoint; secret values remain outside the DTO."""

    runtime_id: str
    adapter_name: str
    base_url: str
    expected_adapter_version: str
    expected_manifest_sha256: str
    bearer_token_env: str | None = None
    connect_timeout_seconds: float = 5.0
    operation_timeout_seconds: float = 300.0
    max_request_bytes: int = 16 * 1024 * 1024
    max_response_bytes: int = 64 * 1024 * 1024
    allow_insecure_http: bool = False
    tls_ca_file: str | None = None
    estimated_latency_ms: int | None = None
    estimated_cost_microusd: int | None = None

    def __post_init__(self) -> None:
        for value, field_name in (
            (self.runtime_id, "runtime_id"),
            (self.adapter_name, "adapter_name"),
            (self.expected_adapter_version, "expected_adapter_version"),
        ):
            try:
                require_non_empty(value, field_name)
            except ContractValidationError as exc:
                raise RuntimeCompositionConfigurationError(str(exc)) from exc
        if _RUNTIME_ID.fullmatch(self.runtime_id) is None:
            raise RuntimeCompositionConfigurationError(
                "runtime_id must contain lowercase letters, digits, '.', '_', or '-'"
            )
        if self.adapter_name not in BALANCED_ADAPTER_IDENTITIES:
            expected = ", ".join(sorted(BALANCED_ADAPTER_IDENTITIES))
            raise RuntimeCompositionConfigurationError(
                f"adapter_name must be one of: {expected}"
            )
        try:
            require_sha256(
                self.expected_manifest_sha256,
                "expected_manifest_sha256",
            )
        except ContractValidationError as exc:
            raise RuntimeCompositionConfigurationError(str(exc)) from exc
        if self.bearer_token_env is not None and (
            not isinstance(self.bearer_token_env, str)
            or _ENVIRONMENT_NAME.fullmatch(self.bearer_token_env) is None
        ):
            raise RuntimeCompositionConfigurationError(
                "bearer_token_env must be a valid environment-variable name"
            )
        object.__setattr__(
            self,
            "estimated_latency_ms",
            _optional_non_negative_int(
                self.estimated_latency_ms,
                "estimated_latency_ms",
            ),
        )
        object.__setattr__(
            self,
            "estimated_cost_microusd",
            _optional_non_negative_int(
                self.estimated_cost_microusd,
                "estimated_cost_microusd",
            ),
        )
        # Reuse the transport's URL, TLS, deadline, and body-bound validation.
        try:
            HTTPParserClientConfig(
                base_url=self.base_url,
                connect_timeout_seconds=self.connect_timeout_seconds,
                operation_timeout_seconds=self.operation_timeout_seconds,
                max_request_bytes=self.max_request_bytes,
                max_response_bytes=self.max_response_bytes,
                expected_adapter_name=self.adapter_name,
                expected_adapter_version=self.expected_adapter_version,
                expected_manifest_sha256=self.expected_manifest_sha256,
                allow_insecure_http=self.allow_insecure_http,
                tls_ca_file=self.tls_ca_file,
            )
        except (TypeError, ValueError) as exc:
            raise RuntimeCompositionConfigurationError(str(exc)) from exc

    @property
    def canonical_endpoint(self) -> tuple[str, str, int]:
        return _canonical_endpoint(self.base_url)

    def client_config(self, *, bearer_token: str | None) -> HTTPParserClientConfig:
        """Build a transport config without retaining the environment mapping."""

        return HTTPParserClientConfig(
            base_url=self.base_url,
            connect_timeout_seconds=self.connect_timeout_seconds,
            operation_timeout_seconds=self.operation_timeout_seconds,
            max_request_bytes=self.max_request_bytes,
            max_response_bytes=self.max_response_bytes,
            bearer_token=bearer_token,
            expected_adapter_name=self.adapter_name,
            expected_adapter_version=self.expected_adapter_version,
            expected_manifest_sha256=self.expected_manifest_sha256,
            allow_insecure_http=self.allow_insecure_http,
            tls_ca_file=self.tls_ca_file,
        )

    def to_dict(self) -> dict[str, object]:
        """Serialize configuration references, never resolved bearer values."""

        return {
            "runtime_id": self.runtime_id,
            "adapter_name": self.adapter_name,
            "base_url": self.base_url,
            "expected_adapter_version": self.expected_adapter_version,
            "expected_manifest_sha256": self.expected_manifest_sha256,
            "bearer_token_env": self.bearer_token_env,
            "connect_timeout_seconds": self.connect_timeout_seconds,
            "operation_timeout_seconds": self.operation_timeout_seconds,
            "max_request_bytes": self.max_request_bytes,
            "max_response_bytes": self.max_response_bytes,
            "allow_insecure_http": self.allow_insecure_http,
            "tls_ca_file": self.tls_ca_file,
            "estimated_latency_ms": self.estimated_latency_ms,
            "estimated_cost_microusd": self.estimated_cost_microusd,
        }


@dataclass(frozen=True, slots=True)
class BalancedRoutedRuntimeConfig:
    """Complete immutable policy and endpoint configuration for the factory."""

    endpoints: tuple[ParserWorkerEndpointConfig, ...]
    parse_options: ParseOptions = field(default_factory=ParseOptions)
    page_probe_policy: PdfProbePolicy = field(default_factory=PdfProbePolicy)
    ingestion_policy: RoutedIngestionPolicy = field(
        default_factory=RoutedIngestionPolicy
    )
    orchestration_policy: ParsingOrchestrationPolicy = field(
        default_factory=ParsingOrchestrationPolicy
    )
    reconciliation_policy: ReconciliationPolicy = field(
        default_factory=ReconciliationPolicy
    )
    max_primary_latency_ms: int | None = None
    max_total_latency_ms: int | None = None
    max_primary_cost_microusd: int | None = None
    max_total_cost_microusd: int | None = None

    def __post_init__(self) -> None:
        endpoints = tuple(self.endpoints)
        if any(not isinstance(item, ParserWorkerEndpointConfig) for item in endpoints):
            raise RuntimeCompositionConfigurationError(
                "endpoints must contain ParserWorkerEndpointConfig values"
            )
        adapter_names = [item.adapter_name for item in endpoints]
        if len(endpoints) != 3 or set(adapter_names) != BALANCED_ADAPTER_IDENTITIES:
            expected = ", ".join(sorted(BALANCED_ADAPTER_IDENTITIES))
            raise RuntimeCompositionConfigurationError(
                "balanced runtime requires exactly one endpoint for each adapter: "
                + expected
            )
        runtime_ids = [item.runtime_id for item in endpoints]
        if len(runtime_ids) != len(set(runtime_ids)):
            raise RuntimeCompositionConfigurationError(
                "worker runtime_id values must be unique"
            )
        canonical_endpoints = [item.canonical_endpoint for item in endpoints]
        if len(canonical_endpoints) != len(set(canonical_endpoints)):
            raise RuntimeCompositionConfigurationError(
                "worker endpoint URLs must be unique"
            )
        for value, field_name in (
            (self.parse_options, "parse_options"),
            (self.page_probe_policy, "page_probe_policy"),
            (self.ingestion_policy, "ingestion_policy"),
            (self.orchestration_policy, "orchestration_policy"),
            (self.reconciliation_policy, "reconciliation_policy"),
        ):
            expected_type = {
                "parse_options": ParseOptions,
                "page_probe_policy": PdfProbePolicy,
                "ingestion_policy": RoutedIngestionPolicy,
                "orchestration_policy": ParsingOrchestrationPolicy,
                "reconciliation_policy": ReconciliationPolicy,
            }[field_name]
            if not isinstance(value, expected_type):
                raise RuntimeCompositionConfigurationError(
                    f"{field_name} must be {expected_type.__name__}"
                )
        for field_name in (
            "max_primary_latency_ms",
            "max_total_latency_ms",
            "max_primary_cost_microusd",
            "max_total_cost_microusd",
        ):
            object.__setattr__(
                self,
                field_name,
                _optional_non_negative_int(getattr(self, field_name), field_name),
            )
        if (
            self.max_primary_latency_ms is not None
            and self.max_total_latency_ms is not None
            and self.max_total_latency_ms < self.max_primary_latency_ms
        ):
            raise RuntimeCompositionConfigurationError(
                "max_total_latency_ms must be at least max_primary_latency_ms"
            )
        if (
            self.max_primary_cost_microusd is not None
            and self.max_total_cost_microusd is not None
            and self.max_total_cost_microusd < self.max_primary_cost_microusd
        ):
            raise RuntimeCompositionConfigurationError(
                "max_total_cost_microusd must be at least max_primary_cost_microusd"
            )
        object.__setattr__(
            self,
            "endpoints",
            tuple(sorted(endpoints, key=lambda item: item.adapter_name)),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": RUNTIME_COMPOSITION_SCHEMA_VERSION,
            "endpoints": [item.to_dict() for item in self.endpoints],
            "parse_options": self.parse_options.to_dict(),
            "page_probe_policy": {
                "sha256_chunk_bytes": self.page_probe_policy.sha256_chunk_bytes,
                "infer_scanned_pages": self.page_probe_policy.infer_scanned_pages,
                "scanned_max_native_characters": (
                    self.page_probe_policy.scanned_max_native_characters
                ),
                "scanned_min_raster_coverage_ratio": (
                    self.page_probe_policy.scanned_min_raster_coverage_ratio
                ),
            },
            "ingestion_policy": {
                "max_page_features": self.ingestion_policy.max_page_features,
                "max_audit_bytes": self.ingestion_policy.max_audit_bytes,
                "max_canonical_artifact_bytes": (
                    self.ingestion_policy.max_canonical_artifact_bytes
                ),
                "sha256_chunk_bytes": self.ingestion_policy.sha256_chunk_bytes,
                "probe_identity": self.ingestion_policy.probe_identity,
            },
            "orchestration_policy": self.orchestration_policy.to_dict(),
            "reconciliation_policy": self.reconciliation_policy.to_dict(),
            "max_primary_latency_ms": self.max_primary_latency_ms,
            "max_total_latency_ms": self.max_total_latency_ms,
            "max_primary_cost_microusd": self.max_primary_cost_microusd,
            "max_total_cost_microusd": self.max_total_cost_microusd,
        }


@runtime_checkable
class ManagedParserClient(Protocol):
    """ParserAdapter plus the explicit lifecycle and readiness operations."""

    def refresh_manifest(self) -> ParserCapabilityManifest:
        ...

    def manifest(self) -> ParserCapabilityManifest:
        ...

    def health(self, *, require_ready: bool = False) -> WorkerHealth:
        ...

    def parse(self, request: ParseRequest) -> ParseResult:
        ...

    def close(self) -> None:
        ...


@dataclass(frozen=True, slots=True)
class WorkerRuntimeIdentity:
    """Secret-free identity proven during capability and health checks."""

    runtime_id: str
    base_url: str
    adapter_name: str
    adapter_version: str
    manifest_sha256: str
    worker_instance_id: str

    def to_dict(self) -> dict[str, object]:
        return {
            "runtime_id": self.runtime_id,
            "base_url": self.base_url,
            "adapter_name": self.adapter_name,
            "adapter_version": self.adapter_version,
            "manifest_sha256": self.manifest_sha256,
            "worker_instance_id": self.worker_instance_id,
        }


class BalancedRoutedRuntime:
    """Context-managed owner of ready clients and their ingestion provider."""

    def __init__(
        self,
        *,
        config: BalancedRoutedRuntimeConfig,
        provider: RoutedCanonicalIngestionProvider,
        facade: RoutedBookRAGFacade,
        clients: tuple[ManagedParserClient, ...],
        identities: tuple[WorkerRuntimeIdentity, ...],
    ) -> None:
        self._config = config
        self._provider = provider
        self._facade = facade
        self._clients = clients
        self._identities = identities
        self._document_extractor = ExtractDocumentUseCase(
            RoutedDocumentExtractionBackend(provider),
            policy=DocumentExtractionPolicy(
                max_canonical_artifact_bytes=(
                    config.ingestion_policy.max_canonical_artifact_bytes
                ),
                sha256_chunk_bytes=config.ingestion_policy.sha256_chunk_bytes,
            ),
        )
        self._lock = threading.RLock()
        self._closed = False

    @property
    def config(self) -> BalancedRoutedRuntimeConfig:
        return self._config

    @property
    def provider(self) -> RoutedCanonicalIngestionProvider:
        with self._lock:
            if self._closed:
                raise RuntimeCompositionClosedError("routed runtime is closed")
            return self._provider

    @property
    def facade(self) -> RoutedBookRAGFacade:
        with self._lock:
            if self._closed:
                raise RuntimeCompositionClosedError("routed runtime is closed")
            return self._facade

    @property
    def document_extractor(self) -> ExtractDocumentUseCase:
        """Return the CLI/API-neutral routed extraction application use case."""

        with self._lock:
            if self._closed:
                raise RuntimeCompositionClosedError("routed runtime is closed")
            return self._document_extractor

    @property
    def worker_identities(self) -> tuple[WorkerRuntimeIdentity, ...]:
        return self._identities

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        failures: list[str] = []
        for identity, client in reversed(tuple(zip(self._identities, self._clients))):
            try:
                client.close()
            except Exception as exc:
                failures.append(f"{identity.runtime_id}: {type(exc).__name__}")
        if failures:
            raise RuntimeCompositionCloseError(
                "could not close parser client(s): " + ", ".join(failures)
            )

    def __enter__(self) -> "BalancedRoutedRuntime":
        with self._lock:
            if self._closed:
                raise RuntimeCompositionClosedError("routed runtime is closed")
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        try:
            self.close()
        except RuntimeCompositionCloseError:
            if exc_type is None:
                raise
        return False


ClientFactory = Callable[[HTTPParserClientConfig], ManagedParserClient]
ProbeBackendFactory = Callable[[], PdfProbeBackend]


def _resolve_token(
    endpoint: ParserWorkerEndpointConfig,
    environment: Mapping[str, str],
) -> str | None:
    variable = endpoint.bearer_token_env
    if variable is None:
        return None
    value = environment.get(variable)
    if not isinstance(value, str) or not value:
        raise RuntimeCompositionConfigurationError(
            f"bearer token environment variable {variable!r} is missing or empty"
        )
    return value


def _validate_remote_identity(
    endpoint: ParserWorkerEndpointConfig,
    manifest: ParserCapabilityManifest,
    health: WorkerHealth,
) -> WorkerRuntimeIdentity:
    _validate_manifest_identity(endpoint, manifest)
    if not isinstance(health, WorkerHealth):
        raise RuntimeCompositionIdentityError(
            f"{endpoint.runtime_id} returned a non-health value"
        )
    if (
        health.adapter_name != manifest.adapter_name
        or health.adapter_version != manifest.adapter_version
        or health.manifest_sha256 != manifest.manifest_sha256
    ):
        raise RuntimeCompositionIdentityError(
            f"{endpoint.runtime_id} health and capability identities differ"
        )
    if health.state is not WorkerState.READY or not health.accepting_requests:
        raise WorkerNotReadyError(
            f"parser worker {endpoint.runtime_id} is not accepting requests"
        )
    return WorkerRuntimeIdentity(
        runtime_id=endpoint.runtime_id,
        base_url=endpoint.base_url,
        adapter_name=manifest.adapter_name,
        adapter_version=manifest.adapter_version,
        manifest_sha256=manifest.manifest_sha256,
        worker_instance_id=health.worker_instance_id,
    )


def _validate_manifest_identity(
    endpoint: ParserWorkerEndpointConfig,
    manifest: ParserCapabilityManifest,
) -> None:
    if not isinstance(manifest, ParserCapabilityManifest):
        raise RuntimeCompositionIdentityError(
            f"{endpoint.runtime_id} returned a non-manifest capability value"
        )
    expected = (
        ("adapter_name", endpoint.adapter_name, manifest.adapter_name),
        (
            "adapter_version",
            endpoint.expected_adapter_version,
            manifest.adapter_version,
        ),
        (
            "manifest_sha256",
            endpoint.expected_manifest_sha256,
            manifest.manifest_sha256,
        ),
    )
    for field_name, configured, actual in expected:
        if configured != actual:
            raise RuntimeCompositionIdentityError(
                f"{endpoint.runtime_id} {field_name} differs from its configured pin"
            )


def build_balanced_routed_runtime(
    config: BalancedRoutedRuntimeConfig,
    *,
    environment: Mapping[str, str] | None = None,
    probe_backend_factory: ProbeBackendFactory = PyMuPdfProbeBackend,
    client_factory: ClientFactory = HTTPParserAdapter,
) -> BalancedRoutedRuntime:
    """Build and verify the opt-in balanced runtime, or fail without leaking clients.

    Capability and ready-health calls happen before the provider is returned.
    Any construction or identity failure closes every client already created.
    Resolved bearer tokens are passed directly to transport configs and are
    retained by neither the immutable runtime config nor its serialization.
    """

    if not isinstance(config, BalancedRoutedRuntimeConfig):
        raise TypeError("config must be BalancedRoutedRuntimeConfig")
    if not callable(probe_backend_factory):
        raise TypeError("probe_backend_factory must be callable")
    if not callable(client_factory):
        raise TypeError("client_factory must be callable")
    active_environment = os.environ if environment is None else environment
    if not isinstance(active_environment, Mapping):
        raise TypeError("environment must be a mapping")

    # Resolve every secret reference before allocating a client.  Only the
    # three resolved strings enter their individual repr-redacted transports.
    client_configs = tuple(
        endpoint.client_config(
            bearer_token=_resolve_token(endpoint, active_environment)
        )
        for endpoint in config.endpoints
    )
    clients: list[ManagedParserClient] = []
    identities: list[WorkerRuntimeIdentity] = []
    try:
        for endpoint, client_config in zip(
            config.endpoints, client_configs, strict=True
        ):
            client = client_factory(client_config)
            if not isinstance(client, ManagedParserClient):
                close = getattr(client, "close", None)
                if callable(close):
                    try:
                        close()
                    except Exception:
                        pass
                raise RuntimeCompositionConfigurationError(
                    "client_factory must return a managed parser client"
                )
            clients.append(client)
            manifest = client.refresh_manifest()
            _validate_manifest_identity(endpoint, manifest)
            health = client.health(require_ready=True)
            identities.append(
                _validate_remote_identity(endpoint, manifest, health)
            )
        worker_instance_ids = [item.worker_instance_id for item in identities]
        if len(worker_instance_ids) != len(set(worker_instance_ids)):
            raise RuntimeCompositionIdentityError(
                "balanced endpoints resolved to duplicate worker_instance_id values"
            )

        backend = probe_backend_factory()
        page_probe = SourceBoundPdfPageProbe(
            backend,
            policy=config.page_probe_policy,
        )
        latency = {
            endpoint.adapter_name: endpoint.estimated_latency_ms
            for endpoint in config.endpoints
            if endpoint.estimated_latency_ms is not None
        }
        cost = {
            endpoint.adapter_name: endpoint.estimated_cost_microusd
            for endpoint in config.endpoints
            if endpoint.estimated_cost_microusd is not None
        }
        facade = RoutedBookRAGFacade.routed_default(
            adapters=tuple(clients),
            orchestration_policy=config.orchestration_policy,
            reconciliation_policy=config.reconciliation_policy,
            estimated_latency_ms=latency or None,
            estimated_cost_microusd=cost or None,
            max_primary_latency_ms=config.max_primary_latency_ms,
            max_total_latency_ms=config.max_total_latency_ms,
            max_primary_cost_microusd=config.max_primary_cost_microusd,
            max_total_cost_microusd=config.max_total_cost_microusd,
        )
        provider = RoutedCanonicalIngestionProvider(
            page_probe,
            facade,
            parse_options=config.parse_options,
            policy=config.ingestion_policy,
        )
        return BalancedRoutedRuntime(
            config=config,
            provider=provider,
            facade=facade,
            clients=tuple(clients),
            identities=tuple(identities),
        )
    except BaseException:
        for client in reversed(clients):
            try:
                client.close()
            except Exception:
                pass
        raise


__all__ = [
    "BALANCED_ADAPTER_IDENTITIES",
    "RUNTIME_COMPOSITION_SCHEMA_VERSION",
    "BalancedRoutedRuntime",
    "BalancedRoutedRuntimeConfig",
    "ManagedParserClient",
    "ParserWorkerEndpointConfig",
    "RuntimeCompositionCloseError",
    "RuntimeCompositionClosedError",
    "RuntimeCompositionConfigurationError",
    "RuntimeCompositionError",
    "RuntimeCompositionIdentityError",
    "WorkerRuntimeIdentity",
    "build_balanced_routed_runtime",
]
