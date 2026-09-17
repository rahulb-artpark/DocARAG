"""Dependency-light runtime diagnostics for DocARAG.

This module deliberately uses only the Python standard library.  In particular,
it must remain safe to import before Torch, PaddlePaddle, Transformers, MinerU,
or any model-serving client has been installed.

The public entry points are :func:`run_diagnostics` and
:func:`options_from_docarag_config`.  Probe functions are injectable so the
checks are deterministic in tests and embeddable in the CLI, API, or Web UI.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import tomllib
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence
from urllib.error import HTTPError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen

from document_services.runtime.compatibility_manifest import (
    CompatibilityManifestError,
    DEFAULT_MANIFEST_PATH,
    RuntimeCompatibility,
    RuntimeCompatibilityManifest,
    VerificationLevel,
    load_runtime_compatibility_manifest,
    validate_worker_capability_payload,
    validate_worker_health_payload,
    worker_endpoint_url,
)


MAX_DIAGNOSTIC_RESPONSE_BYTES = 1024 * 1024
MAX_WORKER_HEALTH_RESPONSE_BYTES = 64 * 1024
MAX_WORKER_CAPABILITY_RESPONSE_BYTES = 1024 * 1024
MAX_WORKER_DIAGNOSTIC_TIMEOUT_SECONDS = 10.0


class Status(str, Enum):
    """Traffic-light status used by every diagnostic check."""

    GREEN = "green"
    AMBER = "amber"
    RED = "red"


class CheckType(str, Enum):
    """Stable machine-readable check categories."""

    PYTHON = "python"
    CONFIGURATION = "configuration"
    PATH = "path"
    COMMAND = "command"
    HTTP_ENDPOINT = "http_endpoint"
    MODEL_ENDPOINT = "model_endpoint"
    GPU = "gpu"
    RUNTIME_COMPATIBILITY = "runtime_compatibility"
    WORKER_CAPABILITY = "worker_capability"


_STATUS_WEIGHT = {Status.GREEN: 0, Status.AMBER: 1, Status.RED: 2}


@dataclass(frozen=True)
class CheckResult:
    """One JSON-safe diagnostic result."""

    check_id: str
    check_type: CheckType
    status: Status
    message: str
    details: Mapping[str, Any] = field(default_factory=dict)
    remediation: Optional[str] = None
    duration_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "check_id": self.check_id,
            "check_type": self.check_type.value,
            "status": self.status.value,
            "message": self.message,
            "details": _json_safe(self.details),
            "duration_ms": round(self.duration_ms, 3),
        }
        if self.remediation:
            result["remediation"] = self.remediation
        return result


@dataclass(frozen=True)
class DiagnosticsReport:
    """Complete diagnostics result suitable for a CLI or HTTP response."""

    checks: tuple[CheckResult, ...]
    generated_at: str

    @property
    def overall_status(self) -> Status:
        if not self.checks:
            return Status.AMBER
        return max((check.status for check in self.checks), key=_STATUS_WEIGHT.get)

    @property
    def exit_code(self) -> int:
        """Return non-zero only when a required check is red."""

        return 1 if self.overall_status is Status.RED else 0

    @property
    def ok(self) -> bool:
        """Whether all required checks passed.

        Amber warnings remain usable in the default CLI mode; ``--strict`` can
        use this property to fail only on red checks.
        """

        return self.overall_status is not Status.RED

    def to_dict(self) -> dict[str, Any]:
        counts = {
            status.value: sum(check.status is status for check in self.checks)
            for status in Status
        }
        return {
            "schema_version": 1,
            "generated_at": self.generated_at,
            "overall_status": self.overall_status.value,
            "exit_code": self.exit_code,
            "counts": counts,
            "checks": [check.to_dict() for check in self.checks],
        }

    def as_dict(self) -> dict[str, Any]:
        """Compatibility alias used by the public CLI/API layer."""

        return self.to_dict()

    def to_json(self, *, indent: Optional[int] = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    def render(self, *, color: bool = False, include_details: bool = False) -> str:
        """Render a compact human-readable report.

        ANSI color is opt-in so captured logs, tests, and API responses remain
        clean by default.
        """

        def badge(status: Status) -> str:
            label = status.value.upper()
            if not color:
                return f"[{label}]"
            ansi = {
                Status.GREEN: "\033[32m",
                Status.AMBER: "\033[33m",
                Status.RED: "\033[31m",
            }[status]
            return f"{ansi}[{label}]\033[0m"

        lines = [f"DocARAG diagnostics: {self.overall_status.value.upper()}"]
        for check in self.checks:
            lines.append(f"{badge(check.status)} {check.check_id}: {check.message}")
            if check.remediation:
                lines.append(f"  Fix: {check.remediation}")
            if include_details and check.details:
                details = json.dumps(_json_safe(check.details), sort_keys=True)
                lines.append(f"  Details: {details}")
        counts = self.to_dict()["counts"]
        lines.append(
            "Summary: "
            f"{counts['green']} green, {counts['amber']} amber, {counts['red']} red"
        )
        return "\n".join(lines)


@dataclass(frozen=True)
class PythonSpec:
    minimum: tuple[int, int] = (3, 10)
    recommended: Optional[tuple[int, int]] = (3, 12)
    maximum_exclusive: Optional[tuple[int, int]] = None


@dataclass(frozen=True)
class ConfigurationValueSpec:
    name: str
    value: Any
    required: bool = True
    secret: bool = False
    placeholders: tuple[str, ...] = ("TODO", "CHANGEME", "<required>")


@dataclass(frozen=True)
class WritablePathSpec:
    name: str
    path: os.PathLike[str] | str
    required: bool = True


@dataclass(frozen=True)
class CommandSpec:
    name: str
    executable: str
    args: Optional[tuple[str, ...]] = None
    required: bool = True
    timeout_seconds: float = 5.0
    accepted_return_codes: tuple[int, ...] = (0,)


@dataclass(frozen=True)
class EndpointSpec:
    name: str
    url: str
    required: bool = True
    timeout_seconds: float = 3.0
    headers: Mapping[str, str] = field(default_factory=dict)
    accepted_statuses: tuple[int, ...] = tuple(range(200, 400))


@dataclass(frozen=True)
class ModelEndpointSpec:
    name: str
    base_url: str
    model_name: str
    backend: str = "openai"
    required: bool = True
    timeout_seconds: float = 3.0
    headers: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class GpuSpec:
    required: bool = False
    minimum_total_memory_mib: int = 0
    minimum_free_memory_mib: int = 0
    timeout_seconds: float = 5.0


@dataclass(frozen=True)
class ParserWorkerDiagnosticSpec:
    """One explicitly configured parser worker to inspect during ``doctor``.

    ``RuntimeSettings`` deliberately does not require workers for the legacy
    compatibility profile.  New composition roots may expose a tuple or mapping
    of these specs through a duck-typed ``parser_workers`` attribute.
    """

    name: str
    runtime_id: str
    base_url: str
    required: bool = True
    timeout_seconds: float = 3.0
    bearer_token: Optional[str] = None
    expected_manifest_sha256: Optional[str] = None
    expected_adapter_version: Optional[str] = None


@dataclass(frozen=True)
class _SettingsWithParserWorkers:
    """Dependency-light view that adds routed workers to legacy settings."""

    base: Any
    parser_workers: tuple[ParserWorkerDiagnosticSpec, ...]

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base, name)


@dataclass(frozen=True)
class DiagnosticsOptions:
    python: PythonSpec = field(default_factory=PythonSpec)
    configuration: tuple[ConfigurationValueSpec, ...] = ()
    writable_paths: tuple[WritablePathSpec, ...] = ()
    commands: tuple[CommandSpec, ...] = ()
    endpoints: tuple[EndpointSpec, ...] = ()
    models: tuple[ModelEndpointSpec, ...] = ()
    gpu: Optional[GpuSpec] = None


@dataclass(frozen=True)
class CommandOutput:
    returncode: int
    stdout: str = ""
    stderr: str = ""


@dataclass(frozen=True)
class HttpResponse:
    status: int
    body: bytes = b""
    headers: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class PathProbeResult:
    writable: bool
    probe_path: str
    exists: bool
    error: Optional[str] = None


@dataclass(frozen=True)
class DiagnosticProbes:
    """Injectable operating-system and network probes."""

    python_version: Callable[[], tuple[int, int, int]]
    which: Callable[[str], Optional[str]]
    run_command: Callable[[Sequence[str], float], CommandOutput]
    http_get: Callable[[str, Mapping[str, str], float], HttpResponse]
    writable_path: Callable[[Path], PathProbeResult]
    monotonic: Callable[[], float]
    timestamp: Callable[[], str]

    @classmethod
    def system(cls) -> "DiagnosticProbes":
        return cls(
            python_version=lambda: (
                sys.version_info.major,
                sys.version_info.minor,
                sys.version_info.micro,
            ),
            which=shutil.which,
            run_command=_run_command,
            http_get=_http_get,
            writable_path=_probe_writable_path,
            monotonic=time.monotonic,
            timestamp=lambda: datetime.now(timezone.utc).isoformat(),
        )


def run_diagnostics(
    options: DiagnosticsOptions,
    *,
    probes: Optional[DiagnosticProbes] = None,
) -> DiagnosticsReport:
    """Run all configured checks without importing model frameworks."""

    probes = probes or DiagnosticProbes.system()
    results: list[CheckResult] = []
    results.append(_timed(probes, lambda: _check_python(options.python, probes)))
    for spec in options.configuration:
        results.append(_timed(probes, lambda spec=spec: _check_configuration(spec)))
    for spec in options.writable_paths:
        results.append(_timed(probes, lambda spec=spec: _check_path(spec, probes)))
    for spec in options.commands:
        results.append(_timed(probes, lambda spec=spec: _check_command(spec, probes)))
    for spec in options.endpoints:
        results.append(_timed(probes, lambda spec=spec: _check_endpoint(spec, probes)))
    for spec in options.models:
        results.append(_timed(probes, lambda spec=spec: _check_model(spec, probes)))
    if options.gpu is not None:
        results.append(_timed(probes, lambda: _check_gpu(options.gpu, probes)))
    return DiagnosticsReport(tuple(results), generated_at=probes.timestamp())


def options_from_docarag_config(
    config: Any,
    *,
    writable_paths: Iterable[WritablePathSpec] = (),
    commands: Iterable[CommandSpec] = (),
    gpu: Optional[GpuSpec] = None,
    environment: Optional[Mapping[str, str]] = None,
) -> DiagnosticsOptions:
    """Build diagnostics for endpoint-bearing parts of a DocARAG config.

    ``config`` may be a dictionary, dataclass, or Pydantic model.  Local model
    blocks are configuration-checked but are not imported or loaded.  Remote
    OpenAI-compatible and Ollama blocks additionally receive model inventory
    checks.  MinerU client configurations receive an HTTP health check.
    """

    plain = _plain_mapping(config)
    active_environment = os.environ if environment is None else environment
    configuration: list[ConfigurationValueSpec] = []
    models: list[ModelEndpointSpec] = []
    endpoints: list[EndpointSpec] = []
    paths = list(writable_paths)

    save_path = plain.get("save_path")
    if not _is_missing(save_path, ("TODO", "CHANGEME", "<required>")):
        paths.append(WritablePathSpec("save_path", str(save_path)))

    seen_models: set[tuple[str, str, str]] = set()
    seen_endpoints: set[str] = set()
    for dotted_name, block in _walk_mapping_blocks(plain):
        backend = str(block.get("backend", "")).strip().lower()
        model_name = block.get("model_name")
        api_base = block.get("api_base")
        if model_name is not None:
            configuration.append(
                ConfigurationValueSpec(f"{dotted_name}.model_name", model_name)
            )
        if backend:
            configuration.append(
                ConfigurationValueSpec(f"{dotted_name}.backend", backend)
            )

        if model_name is not None and _is_remote_backend(backend):
            configuration.append(
                ConfigurationValueSpec(f"{dotted_name}.api_base", api_base)
            )
            if not _is_missing(api_base, ("TODO", "CHANGEME", "<required>")) and not _is_missing(
                model_name, ("TODO", "CHANGEME", "<required>")
            ):
                inventory_backend = "ollama" if backend == "ollama" else "openai"
                key = (str(api_base), str(model_name), inventory_backend)
                if key not in seen_models:
                    seen_models.add(key)
                    headers = _authorization_headers(block)
                    models.append(
                        ModelEndpointSpec(
                            dotted_name,
                            str(api_base),
                            str(model_name),
                            backend=inventory_backend,
                            headers=headers,
                        )
                    )

        qdrant_url = block.get("qdrant_url")
        if qdrant_url is not None:
            qdrant_url_env = block.get(
                "qdrant_url_env", "DOCARAG_QDRANT_URL"
            )
            if not isinstance(qdrant_url_env, str) or not qdrant_url_env:
                qdrant_url_env = "DOCARAG_QDRANT_URL"
            effective_qdrant_url = active_environment.get(
                qdrant_url_env, str(qdrant_url)
            )
            raw_qdrant_url = str(effective_qdrant_url)
            configuration.append(
                ConfigurationValueSpec(
                    f"{dotted_name}.qdrant_url",
                    _safe_http_url_for_report(raw_qdrant_url),
                )
            )
            configuration.append(
                ConfigurationValueSpec(
                    f"{dotted_name}.qdrant_url_env", qdrant_url_env
                )
            )
            if not _is_missing(qdrant_url, ("TODO", "CHANGEME", "<required>")):
                health_url = (
                    raw_qdrant_url
                    if _validate_http_url(raw_qdrant_url)
                    else _append_url_path(raw_qdrant_url, "healthz")
                )
                if health_url not in seen_endpoints:
                    seen_endpoints.add(health_url)
                    headers: dict[str, str] = {}
                    api_key_env = block.get("api_key_env", "QDRANT_API_KEY")
                    if isinstance(api_key_env, str) and api_key_env:
                        api_key = active_environment.get(api_key_env)
                        if api_key:
                            headers["api-key"] = api_key
                    configured_timeout = block.get("timeout_seconds", 3.0)
                    timeout_seconds = (
                        min(float(configured_timeout), 10.0)
                        if isinstance(configured_timeout, (int, float))
                        and not isinstance(configured_timeout, bool)
                        and float(configured_timeout) > 0
                        else 3.0
                    )
                    endpoints.append(
                        EndpointSpec(
                            f"{dotted_name}.qdrant",
                            health_url,
                            timeout_seconds=timeout_seconds,
                            headers=headers,
                        )
                    )

        server_url = block.get("server_url")
        if server_url is not None and "client" in backend:
            configuration.append(
                ConfigurationValueSpec(f"{dotted_name}.server_url", server_url)
            )
            if not _is_missing(server_url, ("TODO", "CHANGEME", "<required>")):
                health_url = _append_url_path(str(server_url), "health")
                if health_url not in seen_endpoints:
                    seen_endpoints.add(health_url)
                    endpoints.append(
                        EndpointSpec(f"{dotted_name}.service", health_url)
                    )

    return DiagnosticsOptions(
        configuration=tuple(_deduplicate_configuration(configuration)),
        writable_paths=tuple(_deduplicate_paths(paths)),
        commands=tuple(commands),
        endpoints=tuple(endpoints),
        models=tuple(models),
        gpu=gpu,
    )


def doctor_for_settings(
    settings: Any,
    check_endpoints: bool = True,
    *,
    probes: Optional[DiagnosticProbes] = None,
) -> DiagnosticsReport:
    """Run the supported controller diagnostics for ``RuntimeSettings``.

    The settings object is intentionally duck-typed to keep this module
    independent from :mod:`document_services.runtime.settings` and importable in partial setups.
    The legacy YAML file is read with a small mapping-only parser; no PyYAML or
    model/runtime dependency is needed by ``docarag doctor``.
    """

    probes = probes or DiagnosticProbes.system()
    project_root = Path(getattr(settings, "project_root")).expanduser()
    system_config = Path(getattr(settings, "system_config")).expanduser()
    runs_dir = Path(getattr(settings, "runs_dir")).expanduser()
    conda_environment = str(getattr(settings, "conda_environment", "gbc-rag"))
    python_executable = getattr(settings, "python_executable", None)

    compatibility_manifest_path = _compatibility_manifest_path(
        settings, project_root=project_root
    )
    worker_specs, worker_config_results = _configured_parser_workers(settings)
    compatibility_results = _runtime_compatibility_results(
        compatibility_manifest_path,
        worker_specs,
        check_endpoints=check_endpoints,
        probes=probes,
    )

    direct_results = [
        _directory_configuration_result("project_root", project_root),
        _project_layout_result(project_root),
        _file_configuration_result("system_config", system_config),
    ]

    config: Mapping[str, Any] = {}
    if system_config.is_file():
        try:
            config = _load_mapping_config(system_config)
        except (OSError, ValueError) as exc:
            direct_results.append(
                CheckResult(
                    "config.system_config.syntax",
                    CheckType.CONFIGURATION,
                    Status.RED,
                    "System configuration could not be parsed",
                    {"path": str(system_config), "error": str(exc)},
                    "Correct the mapping-only YAML/JSON/TOML configuration and rerun docarag doctor.",
                )
            )

    commands = [
        CommandSpec("bash", "bash", args=("--version",)),
        CommandSpec("python3", "python3", args=("--version",)),
        CommandSpec("flock", "flock", args=("--version",)),
        CommandSpec("setsid", "setsid", args=("--version",)),
        CommandSpec("nohup", "nohup", args=("--version",)),
    ]
    if python_executable:
        runtime_python = (str(python_executable),)
        commands.append(
            CommandSpec(
                "runtime_python", str(python_executable), args=("--version",)
            )
        )
    elif os.environ.get("CONDA_DEFAULT_ENV") != conda_environment:
        runtime_python = (
            "conda",
            "run",
            "--no-capture-output",
            "-n",
            conda_environment,
            "python",
        )
        commands.append(
            CommandSpec(
                "runtime_environment",
                runtime_python[0],
                args=(*runtime_python[1:], "--version"),
                timeout_seconds=20.0,
            )
        )
    else:
        runtime_python = (sys.executable,)
        commands.append(
            CommandSpec(
                "runtime_python", sys.executable, args=("--version",)
            )
        )

    runtime_version_probe = (
        "import sys; "
        "version=sys.version_info[:2]; "
        "print(f'runtime_python={sys.version.split()[0]} required=3.12.x'); "
        "raise SystemExit(0 if version == (3, 12) else 1)"
    )
    commands.append(
        CommandSpec(
            "runtime_python_version",
            runtime_python[0],
            args=(*runtime_python[1:], "-c", runtime_version_probe),
            timeout_seconds=20.0,
        )
    )

    dependency_probe = (
        "import importlib.util,sys; "
        "required=("
        "'bs4','doclayout_yolo','en_core_web_sm','json_repair',"
        "'loguru','mineru','modelscope','networkx','nltk','numpy','ollama',"
        "'openai','pandas','PIL','pydantic','PyPDF2','pyclipper','yaml','zmq',"
        "'psycopg','psycopg_pool','qdrant_client','qwen_vl_utils','rapid_table',"
        "'regex','requests','rich','sklearn',"
        "'scipy','shapely','spacy','sympy','textacy','timm','tiktoken','torch',"
        "'torchvision','tqdm','transformers','ultralytics','umap','word2number'"
        "); "
        "missing=[name for name in required if importlib.util.find_spec(name) is None]; "
        "print('missing=' + ','.join(missing) if missing else 'dependencies=available'); "
        "raise SystemExit(bool(missing))"
    )
    commands.append(
        CommandSpec(
            "runtime_dependencies",
            runtime_python[0],
            args=(*runtime_python[1:], "-c", dependency_probe),
            timeout_seconds=30.0,
        )
    )
    commands.append(
        CommandSpec(
            "runtime_package_consistency",
            runtime_python[0],
            args=(*runtime_python[1:], "-m", "pip", "check"),
            timeout_seconds=30.0,
        )
    )
    if check_endpoints:
        database_url_env = str(
            getattr(settings, "database_url_env", "DOCARAG_DATABASE_URL")
        )
        postgres_probe = (
            "import json; "
            "from document_services.persistence import PostgresLedger; "
            f"health=PostgresLedger(env_name={database_url_env!r}).health(); "
            "print(json.dumps(health.to_dict(),sort_keys=True)); "
            "raise SystemExit(0 if health.ok else 1)"
        )
        commands.append(
            CommandSpec(
                "postgresql",
                runtime_python[0],
                args=(*runtime_python[1:], "-c", postgres_probe),
                timeout_seconds=15.0,
            )
        )

    gpu_required = _configuration_requests_cuda(config)
    options = options_from_docarag_config(
        config,
        writable_paths=(WritablePathSpec("runs_dir", runs_dir),),
        commands=commands,
        # An explicit CUDA device makes GPU availability a hard prerequisite.
        # CPU/remote-only configurations still receive an optional inventory
        # check.  In either case flag less than 6 GiB request headroom.
        gpu=GpuSpec(
            required=gpu_required,
            minimum_free_memory_mib=6 * 1024,
        ),
    )
    options = DiagnosticsOptions(
        python=PythonSpec(
            minimum=(3, 12),
            recommended=(3, 12),
            maximum_exclusive=(3, 13),
        ),
        configuration=options.configuration,
        writable_paths=options.writable_paths,
        commands=options.commands,
        endpoints=options.endpoints if check_endpoints else (),
        models=options.models if check_endpoints else (),
        gpu=options.gpu,
    )
    report = run_diagnostics(options, probes=probes)
    checks = (
        report.checks[0],
        *direct_results,
        *worker_config_results,
        *compatibility_results,
        *report.checks[1:],
    )
    return DiagnosticsReport(tuple(checks), generated_at=report.generated_at)


def _compatibility_manifest_path(settings: Any, *, project_root: Path) -> Path:
    configured = getattr(settings, "runtime_compatibility_manifest", None)
    if configured:
        path = Path(configured).expanduser()
        return path if path.is_absolute() else project_root / path
    # The manifest is package data. Resolving it beside this module keeps
    # `doctor` working from an editable checkout and an installed wheel.
    return DEFAULT_MANIFEST_PATH


def doctor_for_routed_profile(
    settings: Any,
    profile: Any,
    check_endpoints: bool = True,
    *,
    probes: Optional[DiagnosticProbes] = None,
    environment: Optional[Mapping[str, str]] = None,
) -> DiagnosticsReport:
    """Diagnose a loaded routed profile before selecting it for indexing.

    The profile loader has already established syntax and contract validity.
    This layer separates static pin readiness, environment readiness, runtime
    compatibility, and live worker readiness. Bearer values are passed only to
    the bounded HTTP probe and are never added to report details.
    """

    probes = probes or DiagnosticProbes.system()
    active_environment = os.environ if environment is None else environment
    project_root = Path(getattr(settings, "project_root")).expanduser()
    manifest_path = _compatibility_manifest_path(
        settings, project_root=project_root
    )
    profile_id = getattr(profile, "profile_id", None)
    source_sha256 = getattr(profile, "source_sha256", None)
    configuration_sha256 = getattr(profile, "configuration_sha256", None)
    config = getattr(profile, "config", None)
    endpoints = getattr(config, "endpoints", None)
    if (
        not isinstance(profile_id, str)
        or not isinstance(source_sha256, str)
        or not isinstance(configuration_sha256, str)
        or not isinstance(endpoints, tuple)
        or not callable(getattr(profile, "configuration_issues", None))
    ):
        raise TypeError("profile must be a loaded RoutedRuntimeProfile")

    profile_checks: list[CheckResult] = [
        CheckResult(
            "routed_profile.identity",
            CheckType.CONFIGURATION,
            Status.GREEN,
            "Routed profile is structurally valid and source-bound",
            {
                "profile_id": profile_id,
                "source_path": str(getattr(profile, "source_path")),
                "source_sha256": source_sha256,
                "configuration_sha256": configuration_sha256,
                "worker_count": len(endpoints),
            },
        ),
        CheckResult(
            "routed_profile.indexing_integration",
            CheckType.CONFIGURATION,
            Status.GREEN,
            "Routed profile can be selected by the supported index command",
            {
                "routed_indexing_enabled": True,
                "selection_flag": "--routed-profile",
                "revision_binding": "exact profile bytes plus normalized configuration",
            },
            None,
        ),
    ]
    static_issues: dict[str, list[Any]] = {}
    for issue in profile.configuration_issues():
        static_issues.setdefault(issue.runtime_id, []).append(issue)

    compatibility = None
    try:
        compatibility = load_runtime_compatibility_manifest(manifest_path)
    except CompatibilityManifestError:
        # doctor_for_settings below reports the manifest failure authoritatively.
        pass

    specs: list[ParserWorkerDiagnosticSpec] = []
    for endpoint in endpoints:
        runtime_id = endpoint.runtime_id
        issues = list(static_issues.get(runtime_id, ()))
        expectation = (
            compatibility.by_adapter_name.get(endpoint.adapter_name)
            if compatibility is not None
            else None
        )
        reasons = [issue.message for issue in issues]
        issue_codes = [issue.code for issue in issues]
        if compatibility is not None and expectation is None:
            reasons.append(
                "adapter_name has no parser-worker entry in the runtime compatibility manifest"
            )
            issue_codes.append("adapter_not_declared")
        elif (
            expectation is not None
            and "unqualified_adapter_version" not in issue_codes
            and endpoint.expected_adapter_version not in expectation.adapter_versions
        ):
            reasons.append(
                "expected_adapter_version is not allowed by the runtime compatibility manifest"
            )
            issue_codes.append("adapter_version_not_allowed")

        token: str | None = None
        token_env = endpoint.bearer_token_env
        token_present = False
        if token_env is not None:
            candidate = active_environment.get(token_env)
            token_present = isinstance(candidate, str) and bool(candidate)
            # Offline mode skips network contact; it does not make a missing
            # required credential usable.  Keep the value out of the report in
            # both modes, and only hand it to a live bounded probe when needed.
            if not token_present:
                reasons.append(
                    "configured bearer-token environment variable is missing or empty"
                )
                issue_codes.append("bearer_token_environment_missing")
            elif check_endpoints:
                token = candidate

        safe_id = re.sub(r"[^a-zA-Z0-9_.-]+", "-", runtime_id)
        details: dict[str, Any] = {
            "runtime_id": runtime_id,
            "adapter_name": endpoint.adapter_name,
            "base_url": endpoint.base_url,
            "expected_adapter_version": endpoint.expected_adapter_version,
            "expected_manifest_sha256": endpoint.expected_manifest_sha256,
            "bearer_token_env": token_env,
            "bearer_token_present": token_present,
            "compatibility_runtime_id": (
                expectation.runtime_id if expectation is not None else None
            ),
            "issue_codes": issue_codes,
        }
        if reasons:
            profile_checks.append(
                CheckResult(
                    f"routed_profile.worker_config.{safe_id}",
                    CheckType.CONFIGURATION,
                    Status.RED,
                    "Routed worker configuration is not ready",
                    {**details, "reasons": reasons},
                    "Replace every template pin with a qualified exact value, use an "
                    "allowed adapter version, and configure required secret environments.",
                )
            )
            continue
        if expectation is None:
            # A missing/invalid manifest is reported by doctor_for_settings. Keep
            # a spec so that a missing manifest is required rather than optional.
            compatibility_runtime_id = runtime_id
        else:
            compatibility_runtime_id = expectation.runtime_id
        specs.append(
            ParserWorkerDiagnosticSpec(
                name=runtime_id,
                runtime_id=compatibility_runtime_id,
                base_url=endpoint.base_url,
                timeout_seconds=min(
                    float(endpoint.connect_timeout_seconds),
                    MAX_WORKER_DIAGNOSTIC_TIMEOUT_SECONDS,
                ),
                bearer_token=token,
                expected_manifest_sha256=endpoint.expected_manifest_sha256,
                expected_adapter_version=endpoint.expected_adapter_version,
            )
        )
        profile_checks.append(
            CheckResult(
                f"routed_profile.worker_config.{safe_id}",
                CheckType.CONFIGURATION,
                Status.GREEN,
                "Routed worker has exact static pins",
                details,
            )
        )

    base_report = doctor_for_settings(
        _SettingsWithParserWorkers(settings, tuple(specs)),
        check_endpoints=check_endpoints,
        probes=probes,
    )
    return DiagnosticsReport(
        (*profile_checks, *base_report.checks),
        generated_at=base_report.generated_at,
    )


def _configured_parser_workers(
    settings: Any,
) -> tuple[tuple[ParserWorkerDiagnosticSpec, ...], tuple[CheckResult, ...]]:
    """Coerce an optional duck-typed worker collection without exposing tokens."""

    raw = getattr(settings, "parser_workers", ())
    if raw in (None, ()):
        return (), ()
    entries: list[tuple[str | None, object]] = []
    if isinstance(raw, Mapping):
        entries.extend((str(name), value) for name, value in raw.items())
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        entries.extend((None, value) for value in raw)
    else:
        entries.append((None, raw))

    specs: list[ParserWorkerDiagnosticSpec] = []
    failures: list[CheckResult] = []
    for index, (mapping_name, value) in enumerate(entries):
        try:
            spec = _coerce_parser_worker_spec(value, mapping_name=mapping_name)
            _validate_parser_worker_spec(spec)
        except (TypeError, ValueError) as exc:
            failures.append(
                CheckResult(
                    f"runtime_compatibility.worker_config.{index}",
                    CheckType.CONFIGURATION,
                    Status.RED,
                    "Configured parser worker is invalid",
                    {"index": index, "error": str(exc)},
                    "Set name, runtime_id, and an http(s) base_url; keep timeout_seconds "
                    f"between 0 and {MAX_WORKER_DIAGNOSTIC_TIMEOUT_SECONDS:g} seconds.",
                )
            )
            continue
        specs.append(spec)

    names = [item.name for item in specs]
    duplicate_names = sorted({name for name in names if names.count(name) > 1})
    if duplicate_names:
        failures.append(
            CheckResult(
                "runtime_compatibility.worker_config.duplicates",
                CheckType.CONFIGURATION,
                Status.RED,
                "Configured parser worker names are not unique",
                {"duplicate_names": duplicate_names},
                "Give every configured parser worker a unique name.",
            )
        )
        duplicate_set = set(duplicate_names)
        specs = [item for item in specs if item.name not in duplicate_set]
    return tuple(specs), tuple(failures)


def _coerce_parser_worker_spec(
    value: object, *, mapping_name: str | None
) -> ParserWorkerDiagnosticSpec:
    if isinstance(value, ParserWorkerDiagnosticSpec):
        return value
    if isinstance(value, str):
        if mapping_name is None:
            raise ValueError("a string worker endpoint requires a mapping key/runtime_id")
        return ParserWorkerDiagnosticSpec(mapping_name, mapping_name, value)
    if isinstance(value, Mapping):
        name = value.get("name", mapping_name)
        runtime_id = value.get("runtime_id", mapping_name)
        return ParserWorkerDiagnosticSpec(
            name=name,  # type: ignore[arg-type]
            runtime_id=runtime_id,  # type: ignore[arg-type]
            base_url=value.get("base_url"),  # type: ignore[arg-type]
            required=value.get("required", True),  # type: ignore[arg-type]
            timeout_seconds=value.get("timeout_seconds", 3.0),  # type: ignore[arg-type]
            bearer_token=value.get("bearer_token"),  # type: ignore[arg-type]
            expected_manifest_sha256=value.get("expected_manifest_sha256"),  # type: ignore[arg-type]
            expected_adapter_version=value.get("expected_adapter_version"),  # type: ignore[arg-type]
        )
    try:
        return ParserWorkerDiagnosticSpec(
            name=getattr(value, "name"),
            runtime_id=getattr(value, "runtime_id"),
            base_url=getattr(value, "base_url"),
            required=getattr(value, "required", True),
            timeout_seconds=getattr(value, "timeout_seconds", 3.0),
            bearer_token=getattr(value, "bearer_token", None),
            expected_manifest_sha256=getattr(
                value, "expected_manifest_sha256", None
            ),
            expected_adapter_version=getattr(
                value, "expected_adapter_version", None
            ),
        )
    except AttributeError as exc:
        raise TypeError("worker must be a spec, mapping, or endpoint object") from exc


def _validate_parser_worker_spec(spec: ParserWorkerDiagnosticSpec) -> None:
    for field_name in ("name", "runtime_id", "base_url"):
        value = getattr(spec, field_name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field_name} must be a non-empty string")
    if not isinstance(spec.required, bool):
        raise ValueError("required must be a boolean")
    if (
        isinstance(spec.timeout_seconds, bool)
        or not isinstance(spec.timeout_seconds, (int, float))
        or not 0 < float(spec.timeout_seconds) <= MAX_WORKER_DIAGNOSTIC_TIMEOUT_SECONDS
    ):
        raise ValueError(
            "timeout_seconds must be positive and no greater than "
            f"{MAX_WORKER_DIAGNOSTIC_TIMEOUT_SECONDS:g}"
        )
    if spec.bearer_token is not None and (
        not isinstance(spec.bearer_token, str) or not spec.bearer_token
    ):
        raise ValueError("bearer_token must be a non-empty string when set")
    if spec.expected_manifest_sha256 is not None and (
        not isinstance(spec.expected_manifest_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", spec.expected_manifest_sha256) is None
    ):
        raise ValueError(
            "expected_manifest_sha256 must be a lowercase SHA-256 digest"
        )
    if spec.expected_adapter_version is not None and (
        not isinstance(spec.expected_adapter_version, str)
        or not spec.expected_adapter_version.strip()
    ):
        raise ValueError(
            "expected_adapter_version must be a non-empty string when set"
        )
    # Validate the origin once here. Runtime-declared paths are resolved later.
    worker_endpoint_url(spec.base_url, "/v1/health")


def _runtime_compatibility_results(
    manifest_path: Path,
    worker_specs: Sequence[ParserWorkerDiagnosticSpec],
    *,
    check_endpoints: bool,
    probes: DiagnosticProbes,
) -> tuple[CheckResult, ...]:
    results: list[CheckResult] = []
    if not manifest_path.is_file():
        status = Status.RED if worker_specs else Status.AMBER
        results.append(
            CheckResult(
                "runtime_compatibility.manifest",
                CheckType.RUNTIME_COMPATIBILITY,
                status,
                "Runtime compatibility manifest is missing",
                {
                    "path": str(manifest_path),
                    "configured_worker_count": len(worker_specs),
                },
                "Restore document_services/runtime/runtime_compatibility.toml from the same DocARAG "
                "release, or configure runtime_compatibility_manifest explicitly.",
            )
        )
        return tuple(results)
    try:
        manifest = load_runtime_compatibility_manifest(manifest_path)
    except CompatibilityManifestError as exc:
        results.append(
            CheckResult(
                "runtime_compatibility.manifest",
                CheckType.RUNTIME_COMPATIBILITY,
                Status.RED,
                "Runtime compatibility manifest is invalid",
                {"path": str(manifest_path), "error": str(exc)},
                "Replace the manifest with the version shipped by this controller or "
                "correct it against docarag.runtime-compatibility.v1.",
            )
        )
        return tuple(results)

    results.append(
        CheckResult(
            "runtime_compatibility.manifest",
            CheckType.RUNTIME_COMPATIBILITY,
            Status.GREEN,
            "Runtime compatibility manifest is valid",
            {
                "path": str(manifest_path),
                "schema_version": manifest.schema_version,
                "manifest_revision": manifest.manifest_revision,
                "source_sha256": manifest.source_sha256,
                "runtime_count": len(manifest.runtimes),
            },
        )
    )
    for runtime in manifest.runtimes:
        results.append(_runtime_verification_result(runtime))
    for worker in worker_specs:
        results.append(
            _configured_worker_result(
                worker,
                manifest,
                check_endpoints=check_endpoints,
                probes=probes,
            )
        )
    return tuple(results)


def _runtime_verification_result(runtime: RuntimeCompatibility) -> CheckResult:
    details = {
        "runtime_id": runtime.runtime_id,
        "runtime_kind": runtime.runtime_kind.value,
        "adapter_name": runtime.adapter_name,
        "adapter_versions": list(runtime.adapter_versions),
        "contract_schema": runtime.contract_schema,
        "worker_protocol": runtime.worker_protocol,
        "accelerator": runtime.accelerator.value,
        "minimum_vram_mib": runtime.minimum_vram_mib,
        "minimum_vram_certified": runtime.minimum_vram_certified,
        "health_location": runtime.health_location,
        "capability_location": runtime.capability_location,
        "verification_level": runtime.verification_level.value,
        "production_verified": runtime.is_production_verified,
        "uncertified_artifacts": list(runtime.uncertified_artifacts),
        "verification_notes": runtime.verification_notes,
    }
    if runtime.is_production_verified:
        return CheckResult(
            f"runtime_compatibility.runtime.{runtime.runtime_id}",
            CheckType.RUNTIME_COMPATIBILITY,
            Status.GREEN,
            f"Runtime matrix is real-model tested: {runtime.runtime_id}",
            details,
        )
    return CheckResult(
        f"runtime_compatibility.runtime.{runtime.runtime_id}",
        CheckType.RUNTIME_COMPATIBILITY,
        Status.AMBER,
        f"Runtime matrix is not production certified: {runtime.runtime_id}",
        details,
        "Run the real-model qualification suite, pin exact package/model/image "
        "versions (and immutable image digests), record a measured VRAM minimum, "
        "then promote this matrix in a reviewed manifest revision.",
    )


def _configured_worker_result(
    worker: ParserWorkerDiagnosticSpec,
    manifest: RuntimeCompatibilityManifest,
    *,
    check_endpoints: bool,
    probes: DiagnosticProbes,
) -> CheckResult:
    check_id = "runtime_compatibility.worker." + re.sub(
        r"[^a-zA-Z0-9_.-]+", "-", worker.name
    )
    expectation = manifest.by_runtime_id.get(worker.runtime_id)
    base_details: dict[str, Any] = {
        "name": worker.name,
        "runtime_id": worker.runtime_id,
        "base_url": worker.base_url,
        "required": worker.required,
        "configured_adapter_version": worker.expected_adapter_version,
    }
    failure_status = Status.RED if worker.required else Status.AMBER
    if expectation is None:
        base_details["available_runtime_ids"] = sorted(manifest.by_runtime_id)
        return CheckResult(
            check_id,
            CheckType.WORKER_CAPABILITY,
            failure_status,
            "Configured parser worker has no compatibility declaration",
            base_details,
            "Set runtime_id to a parser-worker entry in the compatibility manifest, "
            "or add a reviewed entry before enabling this worker.",
        )
    if expectation.adapter_name is None:
        return CheckResult(
            check_id,
            CheckType.WORKER_CAPABILITY,
            failure_status,
            "Configured endpoint refers to a controller runtime, not a parser worker",
            base_details,
            "Select a parser-worker runtime_id from the compatibility manifest.",
        )
    health_url = worker_endpoint_url(worker.base_url, expectation.health_location or "")
    capability_url = worker_endpoint_url(
        worker.base_url, expectation.capability_location or ""
    )
    base_details.update(
        {
            "health_url": health_url,
            "capability_url": capability_url,
            "expected_adapter": expectation.adapter_name,
            "expected_adapter_versions": list(expectation.adapter_versions),
            "verification_level": expectation.verification_level.value,
        }
    )
    if not check_endpoints:
        return CheckResult(
            check_id,
            CheckType.WORKER_CAPABILITY,
            Status.AMBER,
            "Parser worker capability check was skipped in offline mode",
            base_details,
            "Run `docarag doctor` without --offline while the parser worker is running.",
        )

    headers = {}
    if worker.bearer_token:
        headers["Authorization"] = f"Bearer {worker.bearer_token}"
    transport_secrets = _transport_secret_values(headers)
    timeout = float(worker.timeout_seconds)
    health_payload, health_error, health_meta = _fetch_bounded_json(
        health_url,
        headers,
        timeout,
        MAX_WORKER_HEALTH_RESPONSE_BYTES,
        probes,
    )
    base_details["health"] = health_meta
    if health_error:
        return CheckResult(
            check_id,
            CheckType.WORKER_CAPABILITY,
            failure_status,
            "Parser worker health check failed",
            {**base_details, "error": health_error},
            "Start the configured worker, verify its base URL/credentials, and ensure "
            "GET /v1/health returns the parser-worker protocol payload.",
        )
    health = validate_worker_health_payload(expectation, health_payload)

    capability_payload, capability_error, capability_meta = _fetch_bounded_json(
        capability_url,
        headers,
        timeout,
        MAX_WORKER_CAPABILITY_RESPONSE_BYTES,
        probes,
    )
    base_details["capability"] = capability_meta
    if capability_error:
        return CheckResult(
            check_id,
            CheckType.WORKER_CAPABILITY,
            failure_status,
            "Parser worker capability check failed",
            {**base_details, "error": capability_error},
            "Verify GET /v1/capabilities returns the protocol wrapper and a complete "
            "ParserCapabilityManifest JSON object within the response-size limit.",
        )
    capability = validate_worker_capability_payload(
        expectation,
        capability_payload,
        expected_manifest_sha256=worker.expected_manifest_sha256,
    )
    errors = [
        _redact_sensitive_text(item, transport_secrets)
        for item in (*health.errors, *capability.errors)
    ]
    advertised_adapter_version = capability.details.get("adapter_version")
    if (
        worker.expected_adapter_version is not None
        and advertised_adapter_version != worker.expected_adapter_version
    ):
        errors.append(
            "worker adapter version differs from the configured pinned identity"
        )
    health_sha = health.details.get("manifest_sha256")
    capability_sha = capability.details.get("manifest_sha256")
    if health_sha != capability_sha:
        errors.append(
            "health and capability responses advertise different manifest identities"
        )
    base_details["advertised"] = _redact_sensitive_value(
        dict(capability.details), transport_secrets
    )
    base_details["errors"] = errors
    base_details["warnings"] = [
        _redact_sensitive_text(item, transport_secrets)
        for item in capability.warnings
    ]
    if errors:
        return CheckResult(
            check_id,
            CheckType.WORKER_CAPABILITY,
            failure_status,
            "Parser worker capability is incompatible with the controller",
            base_details,
            "Deploy a worker using the declared protocol, parser schema, adapter identity, "
            "and allowed adapter version; do not bypass the manifest check.",
        )
    if capability.warnings or not expectation.is_production_verified:
        return CheckResult(
            check_id,
            CheckType.WORKER_CAPABILITY,
            Status.AMBER,
            "Parser worker contract is compatible but its runtime matrix is uncertified",
            base_details,
            "Keep this worker out of production routing until its exact package/model/image "
            "matrix and VRAM requirement pass real-model qualification.",
        )
    return CheckResult(
        check_id,
        CheckType.WORKER_CAPABILITY,
        Status.GREEN,
        "Parser worker is ready and production compatible",
        base_details,
    )


def _fetch_bounded_json(
    url: str,
    headers: Mapping[str, str],
    timeout_seconds: float,
    maximum_bytes: int,
    probes: DiagnosticProbes,
) -> tuple[object | None, str | None, Mapping[str, object]]:
    meta: dict[str, object] = {
        "url": url,
        "maximum_bytes": maximum_bytes,
        "timeout_seconds": timeout_seconds,
    }
    try:
        response = probes.http_get(url, headers, timeout_seconds)
    except Exception as exc:
        error = _redact_sensitive_text(str(exc), _transport_secret_values(headers))
        return None, f"request failed: {error}", meta
    meta["http_status"] = response.status
    meta["response_bytes"] = len(response.body)
    if response.status < 200 or response.status >= 300:
        return None, f"endpoint returned HTTP {response.status}", meta
    if len(response.body) > maximum_bytes:
        return (
            None,
            f"response exceeds the {maximum_bytes}-byte diagnostic limit",
            meta,
        )
    try:
        return json.loads(response.body), None, meta
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return None, f"response is not valid UTF-8 JSON: {exc}", meta


def _transport_secret_values(headers: Mapping[str, str]) -> tuple[str, ...]:
    values: set[str] = set()
    for key, value in headers.items():
        if key.casefold() not in {
            "authorization",
            "proxy-authorization",
            "api-key",
            "x-api-key",
        }:
            continue
        if value:
            values.add(value)
            if value.casefold().startswith("bearer ") and value[7:]:
                values.add(value[7:])
    return tuple(sorted(values, key=len, reverse=True))


def _redact_sensitive_text(value: str, secrets: Sequence[str]) -> str:
    result = value
    for secret in secrets:
        result = result.replace(secret, "<redacted>")
    return result


def _redact_sensitive_value(value: Any, secrets: Sequence[str]) -> Any:
    if isinstance(value, str):
        return _redact_sensitive_text(value, secrets)
    if isinstance(value, Mapping):
        return {
            str(key): _redact_sensitive_value(item, secrets)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_redact_sensitive_value(item, secrets) for item in value]
    return value


def _configuration_requests_cuda(value: Any) -> bool:
    """Return whether a parsed configuration explicitly selects CUDA."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized_key = str(key).strip().casefold()
            if normalized_key in {"device", "devices"}:
                if isinstance(item, str) and "cuda" in item.casefold():
                    return True
                if isinstance(item, (list, tuple)) and any(
                    isinstance(device, str) and "cuda" in device.casefold()
                    for device in item
                ):
                    return True
            if _configuration_requests_cuda(item):
                return True
        return False
    if isinstance(value, (list, tuple)):
        return any(_configuration_requests_cuda(item) for item in value)
    return False


def render_report(
    report: DiagnosticsReport,
    *,
    color: bool = False,
    include_details: bool = False,
) -> str:
    """Public rendering helper used by the CLI."""

    return report.render(color=color, include_details=include_details)


def _check_python(spec: PythonSpec, probes: DiagnosticProbes) -> CheckResult:
    version = probes.python_version()
    short = version[:2]
    version_text = ".".join(str(part) for part in version)
    minimum_text = ".".join(str(part) for part in spec.minimum)
    if short < spec.minimum:
        return CheckResult(
            "python",
            CheckType.PYTHON,
            Status.RED,
            f"Python {version_text} is below the supported minimum {minimum_text}",
            {"version": version_text, "minimum": minimum_text},
            f"Install Python {minimum_text} or newer and recreate the environment.",
        )
    if spec.maximum_exclusive and short >= spec.maximum_exclusive:
        maximum_text = ".".join(str(part) for part in spec.maximum_exclusive)
        return CheckResult(
            "python",
            CheckType.PYTHON,
            Status.RED,
            f"Python {version_text} is outside the supported range (<{maximum_text})",
            {
                "version": version_text,
                "minimum": minimum_text,
                "maximum_exclusive": maximum_text,
            },
            f"Use Python {minimum_text} through the release immediately before {maximum_text}.",
        )
    if spec.recommended and short < spec.recommended:
        recommended_text = ".".join(str(part) for part in spec.recommended)
        return CheckResult(
            "python",
            CheckType.PYTHON,
            Status.AMBER,
            f"Python {version_text} is supported; {recommended_text}+ is recommended",
            {
                "version": version_text,
                "minimum": minimum_text,
                "recommended": recommended_text,
            },
            f"Upgrade to Python {recommended_text} when practical.",
        )
    return CheckResult(
        "python",
        CheckType.PYTHON,
        Status.GREEN,
        f"Python {version_text} is supported",
        {"version": version_text, "minimum": minimum_text},
    )


def _check_configuration(spec: ConfigurationValueSpec) -> CheckResult:
    missing = _is_missing(spec.value, spec.placeholders)
    shown_value = "<redacted>" if spec.secret and not missing else spec.value
    details = {"required": spec.required, "value": _json_safe(shown_value)}
    if missing:
        status = Status.RED if spec.required else Status.AMBER
        return CheckResult(
            f"config.{spec.name}",
            CheckType.CONFIGURATION,
            status,
            f"Configuration value {spec.name} is missing or still a placeholder",
            details,
            f"Set {spec.name} in the selected DocARAG configuration.",
        )
    return CheckResult(
        f"config.{spec.name}",
        CheckType.CONFIGURATION,
        Status.GREEN,
        f"Configuration value {spec.name} is set",
        details,
    )


def _check_path(spec: WritablePathSpec, probes: DiagnosticProbes) -> CheckResult:
    path = Path(spec.path).expanduser()
    try:
        probe = probes.writable_path(path)
    except Exception as exc:  # Diagnostics must report failures, never crash setup.
        probe = PathProbeResult(False, str(path), path.exists(), str(exc))
    details = {
        "path": str(path),
        "probe_path": probe.probe_path,
        "exists": probe.exists,
        "required": spec.required,
    }
    if probe.error:
        details["error"] = probe.error
    if not probe.writable:
        status = Status.RED if spec.required else Status.AMBER
        return CheckResult(
            f"path.{spec.name}",
            CheckType.PATH,
            status,
            f"Path is not writable: {path}",
            details,
            f"Create {path} or grant the current user write access to {probe.probe_path}.",
        )
    return CheckResult(
        f"path.{spec.name}",
        CheckType.PATH,
        Status.GREEN,
        f"Path is writable: {path}",
        details,
    )


def _check_command(spec: CommandSpec, probes: DiagnosticProbes) -> CheckResult:
    executable_path = probes.which(spec.executable)
    details: dict[str, Any] = {
        "executable": spec.executable,
        "required": spec.required,
    }
    if not executable_path:
        status = Status.RED if spec.required else Status.AMBER
        return CheckResult(
            f"command.{spec.name}",
            CheckType.COMMAND,
            status,
            f"Command not found: {spec.executable}",
            details,
            f"Install {spec.executable} and ensure it is available on PATH.",
        )
    details["path"] = executable_path
    if spec.args is None:
        return CheckResult(
            f"command.{spec.name}",
            CheckType.COMMAND,
            Status.GREEN,
            f"Command is available: {spec.executable}",
            details,
        )
    try:
        output = probes.run_command(
            (executable_path, *spec.args), spec.timeout_seconds
        )
    except Exception as exc:
        status = Status.RED if spec.required else Status.AMBER
        details["error"] = str(exc)
        return CheckResult(
            f"command.{spec.name}",
            CheckType.COMMAND,
            status,
            f"Command probe failed: {spec.executable}",
            details,
            f"Run `{spec.executable} {' '.join(spec.args)}` and resolve the reported error.",
        )
    details["returncode"] = output.returncode
    version_output = (output.stdout or output.stderr).strip().splitlines()
    if version_output:
        details["output"] = version_output[0][:500]
    if output.returncode not in spec.accepted_return_codes:
        status = Status.RED if spec.required else Status.AMBER
        return CheckResult(
            f"command.{spec.name}",
            CheckType.COMMAND,
            status,
            f"Command returned exit code {output.returncode}: {spec.executable}",
            details,
            f"Run `{spec.executable} {' '.join(spec.args)}` and resolve the reported error.",
        )
    return CheckResult(
        f"command.{spec.name}",
        CheckType.COMMAND,
        Status.GREEN,
        f"Command probe succeeded: {spec.executable}",
        details,
    )


def _check_endpoint(spec: EndpointSpec, probes: DiagnosticProbes) -> CheckResult:
    safe_url = _safe_http_url_for_report(spec.url)
    secrets = _transport_secret_values(spec.headers)
    details: dict[str, Any] = {"url": safe_url, "required": spec.required}
    invalid = _validate_http_url(spec.url)
    if invalid:
        status = Status.RED if spec.required else Status.AMBER
        details["error"] = invalid
        return CheckResult(
            f"endpoint.{spec.name}",
            CheckType.HTTP_ENDPOINT,
            status,
            f"Endpoint URL is invalid: {safe_url}",
            details,
            f"Set {spec.name} to an http:// or https:// URL.",
        )
    try:
        response = probes.http_get(spec.url, spec.headers, spec.timeout_seconds)
    except Exception as exc:
        status = Status.RED if spec.required else Status.AMBER
        details["error"] = _redact_sensitive_text(str(exc), secrets)
        return CheckResult(
            f"endpoint.{spec.name}",
            CheckType.HTTP_ENDPOINT,
            status,
            f"Endpoint is unreachable: {safe_url}",
            details,
            "Start the configured service and verify its host, port, and network access.",
        )
    details["http_status"] = response.status
    if response.status not in spec.accepted_statuses:
        status = Status.RED if spec.required else Status.AMBER
        return CheckResult(
            f"endpoint.{spec.name}",
            CheckType.HTTP_ENDPOINT,
            status,
            f"Endpoint returned HTTP {response.status}: {safe_url}",
            details,
            "Check the configured health URL and inspect the service logs.",
        )
    return CheckResult(
        f"endpoint.{spec.name}",
        CheckType.HTTP_ENDPOINT,
        Status.GREEN,
        f"Endpoint is reachable: {safe_url}",
        details,
    )


def _check_model(spec: ModelEndpointSpec, probes: DiagnosticProbes) -> CheckResult:
    safe_base_url = _safe_http_url_for_report(spec.base_url)
    invalid_base = _validate_http_url(spec.base_url)
    inventory_url = (
        spec.base_url
        if invalid_base
        else _model_inventory_url(spec.base_url, spec.backend)
    )
    safe_inventory_url = _safe_http_url_for_report(inventory_url)
    secrets = _transport_secret_values(spec.headers)
    details: dict[str, Any] = {
        "base_url": safe_base_url,
        "inventory_url": safe_inventory_url,
        "backend": spec.backend,
        "model": spec.model_name,
        "required": spec.required,
    }
    invalid = invalid_base or _validate_http_url(inventory_url)
    if invalid:
        status = Status.RED if spec.required else Status.AMBER
        details["error"] = invalid
        return CheckResult(
            f"model.{spec.name}",
            CheckType.MODEL_ENDPOINT,
            status,
            f"Model endpoint URL is invalid: {safe_inventory_url}",
            details,
            f"Correct the API base configured for {spec.name}.",
        )
    try:
        response = probes.http_get(inventory_url, spec.headers, spec.timeout_seconds)
    except Exception as exc:
        status = Status.RED if spec.required else Status.AMBER
        details["error"] = _redact_sensitive_text(str(exc), secrets)
        return CheckResult(
            f"model.{spec.name}",
            CheckType.MODEL_ENDPOINT,
            status,
            f"Model service is unreachable for {spec.name}",
            details,
            "Start the model service and verify its API base, port, and credentials.",
        )
    details["http_status"] = response.status
    if not 200 <= response.status < 300:
        status = Status.RED if spec.required else Status.AMBER
        return CheckResult(
            f"model.{spec.name}",
            CheckType.MODEL_ENDPOINT,
            status,
            f"Model inventory returned HTTP {response.status}",
            details,
            "Verify the model service API base and credentials.",
        )
    try:
        payload = json.loads(response.body.decode("utf-8"))
        available = _extract_model_names(payload)
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        status = Status.RED if spec.required else Status.AMBER
        details["error"] = f"Invalid model inventory response: {exc}"
        return CheckResult(
            f"model.{spec.name}",
            CheckType.MODEL_ENDPOINT,
            status,
            "Model service responded, but its inventory could not be read",
            details,
            "Verify that the configured endpoint exposes an OpenAI /models or Ollama /api/tags response.",
        )
    details["available_model_count"] = len(available)
    details["available_models"] = sorted(available)[:50]
    if not any(_model_names_match(spec.model_name, candidate) for candidate in available):
        status = Status.RED if spec.required else Status.AMBER
        return CheckResult(
            f"model.{spec.name}",
            CheckType.MODEL_ENDPOINT,
            status,
            f"Configured model is not available: {spec.model_name}",
            details,
            f"Load/pull {spec.model_name} or select one of the advertised models.",
        )
    return CheckResult(
        f"model.{spec.name}",
        CheckType.MODEL_ENDPOINT,
        Status.GREEN,
        f"Configured model is available: {spec.model_name}",
        details,
    )


def _check_gpu(spec: GpuSpec, probes: DiagnosticProbes) -> CheckResult:
    executable = probes.which("nvidia-smi")
    details: dict[str, Any] = {
        "required": spec.required,
        "minimum_total_memory_mib": spec.minimum_total_memory_mib,
        "minimum_free_memory_mib": spec.minimum_free_memory_mib,
    }
    if not executable:
        status = Status.RED if spec.required else Status.AMBER
        return CheckResult(
            "gpu.nvidia",
            CheckType.GPU,
            status,
            "nvidia-smi is not available",
            details,
            "Install a supported NVIDIA driver, or select a CPU/remote-model profile.",
        )
    command = (
        executable,
        "--query-gpu=index,name,memory.total,memory.used,memory.free",
        "--format=csv,noheader,nounits",
    )
    try:
        output = probes.run_command(command, spec.timeout_seconds)
    except Exception as exc:
        status = Status.RED if spec.required else Status.AMBER
        details["error"] = str(exc)
        return CheckResult(
            "gpu.nvidia",
            CheckType.GPU,
            status,
            "nvidia-smi failed",
            details,
            "Run nvidia-smi directly and resolve the driver or container GPU error.",
        )
    details["returncode"] = output.returncode
    if output.returncode != 0:
        status = Status.RED if spec.required else Status.AMBER
        details["error"] = output.stderr.strip()[:500]
        return CheckResult(
            "gpu.nvidia",
            CheckType.GPU,
            status,
            f"nvidia-smi returned exit code {output.returncode}",
            details,
            "Inspect nvidia-smi output and the NVIDIA driver/container runtime.",
        )
    try:
        gpus = _parse_nvidia_smi(output.stdout)
    except ValueError as exc:
        status = Status.RED if spec.required else Status.AMBER
        details["error"] = str(exc)
        return CheckResult(
            "gpu.nvidia",
            CheckType.GPU,
            status,
            "nvidia-smi returned an unexpected response",
            details,
            "Run nvidia-smi with the query arguments and inspect its CSV output.",
        )
    details["gpus"] = gpus
    suitable = [
        gpu
        for gpu in gpus
        if gpu["memory_total_mib"] >= spec.minimum_total_memory_mib
        and gpu["memory_free_mib"] >= spec.minimum_free_memory_mib
    ]
    if not suitable:
        status = Status.RED if spec.required else Status.AMBER
        return CheckResult(
            "gpu.nvidia",
            CheckType.GPU,
            status,
            "No GPU currently satisfies the configured memory requirements",
            details,
            "Stop unused GPU workloads, lower the profile memory requirement, or use remote model workers.",
        )
    best = max(suitable, key=lambda item: item["memory_free_mib"])
    return CheckResult(
        "gpu.nvidia",
        CheckType.GPU,
        Status.GREEN,
        f"GPU {best['index']} is ready with {best['memory_free_mib']} MiB free",
        details,
    )


def _timed(
    probes: DiagnosticProbes, function: Callable[[], CheckResult]
) -> CheckResult:
    start = probes.monotonic()
    result = function()
    elapsed_ms = max(0.0, (probes.monotonic() - start) * 1000.0)
    return CheckResult(
        check_id=result.check_id,
        check_type=result.check_type,
        status=result.status,
        message=result.message,
        details=result.details,
        remediation=result.remediation,
        duration_ms=elapsed_ms,
    )


def _run_command(command: Sequence[str], timeout: float) -> CommandOutput:
    completed = subprocess.run(
        list(command),
        capture_output=True,
        check=False,
        text=True,
        timeout=timeout,
    )
    return CommandOutput(completed.returncode, completed.stdout, completed.stderr)


def _http_get(url: str, headers: Mapping[str, str], timeout: float) -> HttpResponse:
    request_headers = {"Accept": "application/json", **dict(headers)}
    request = Request(url, headers=request_headers, method="GET")
    try:
        with urlopen(request, timeout=timeout) as response:  # nosec B310: configured URL
            return HttpResponse(
                status=int(response.status),
                # Read one byte beyond the limit so callers can distinguish an
                # exact-limit payload from a truncated/oversized response.
                body=response.read(MAX_DIAGNOSTIC_RESPONSE_BYTES + 1),
                headers=dict(response.headers.items()),
            )
    except HTTPError as exc:
        return HttpResponse(
            status=int(exc.code),
            body=exc.read(MAX_DIAGNOSTIC_RESPONSE_BYTES + 1),
            headers=dict(exc.headers.items()) if exc.headers else {},
        )


def _probe_writable_path(path: Path) -> PathProbeResult:
    expanded = path.expanduser()
    exists = expanded.exists()
    candidate = expanded if exists and expanded.is_dir() else expanded.parent
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    if not candidate.exists() or not candidate.is_dir():
        return PathProbeResult(False, str(candidate), exists, "No existing parent directory")
    try:
        with tempfile.NamedTemporaryFile(
            prefix=".docarag-doctor-", dir=candidate, delete=True
        ) as handle:
            handle.write(b"ok")
            handle.flush()
        return PathProbeResult(True, str(candidate), exists)
    except OSError as exc:
        return PathProbeResult(False, str(candidate), exists, str(exc))


def _parse_nvidia_smi(output: str) -> list[dict[str, Any]]:
    gpus: list[dict[str, Any]] = []
    for line_number, line in enumerate(output.splitlines(), start=1):
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 5:
            raise ValueError(f"Expected 5 CSV columns on line {line_number}")
        try:
            index = int(parts[0])
            total = int(float(parts[2]))
            used = int(float(parts[3]))
            free = int(float(parts[4]))
        except ValueError as exc:
            raise ValueError(f"Invalid numeric GPU value on line {line_number}") from exc
        gpus.append(
            {
                "index": index,
                "name": parts[1],
                "memory_total_mib": total,
                "memory_used_mib": used,
                "memory_free_mib": free,
            }
        )
    if not gpus:
        raise ValueError("No GPUs were reported")
    return gpus


def _model_inventory_url(base_url: str, backend: str) -> str:
    backend = backend.strip().lower()
    parsed = urlsplit(base_url.rstrip("/"))
    path = parsed.path.rstrip("/")
    if backend == "ollama":
        if path.endswith("/api/tags"):
            target_path = path
        else:
            target_path = f"{path}/api/tags"
    else:
        if path.endswith("/models"):
            target_path = path
        elif path.endswith("/v1"):
            target_path = f"{path}/models"
        else:
            target_path = f"{path}/v1/models"
    return urlunsplit((parsed.scheme, parsed.netloc, target_path, "", ""))


def _extract_model_names(payload: Any) -> set[str]:
    if not isinstance(payload, Mapping):
        raise TypeError("Model inventory is not an object")
    entries = payload.get("data", payload.get("models"))
    if not isinstance(entries, list):
        raise TypeError("Model inventory has no data/models list")
    names: set[str] = set()
    for entry in entries:
        if isinstance(entry, str):
            names.add(entry)
        elif isinstance(entry, Mapping):
            for key in ("id", "name", "model"):
                value = entry.get(key)
                if isinstance(value, str) and value:
                    names.add(value)
                    break
    return names


def _model_names_match(configured: str, advertised: str) -> bool:
    configured = configured.strip()
    advertised = advertised.strip()
    if configured == advertised:
        return True
    return configured.removesuffix(":latest") == advertised.removesuffix(":latest")


def _validate_http_url(url: str) -> Optional[str]:
    try:
        parsed = urlsplit(url)
    except ValueError as exc:
        return str(exc)
    if parsed.scheme not in {"http", "https"}:
        return "URL scheme must be http or https"
    if not parsed.netloc:
        return "URL must include a host"
    if parsed.username is not None or parsed.password is not None:
        return "URL must not contain credentials; configure credentials separately"
    return None


def _safe_http_url_for_report(url: str) -> str:
    """Return an endpoint label that cannot serialize URL credentials.

    Query strings and fragments are omitted as well because operators sometimes
    put access tokens there even though DocARAG transports credentials in
    headers or named environment variables.
    """

    try:
        parsed = urlsplit(url)
        hostname = parsed.hostname
        port = parsed.port
    except (TypeError, ValueError):
        return "<invalid-url>"
    if not parsed.scheme or hostname is None:
        return url if "@" not in url else "<invalid-url>"
    safe_host = f"[{hostname}]" if ":" in hostname else hostname
    netloc = f"{safe_host}:{port}" if port is not None else safe_host
    return urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))


def _append_url_path(base_url: str, suffix: str) -> str:
    parsed = urlsplit(base_url.rstrip("/"))
    path = f"{parsed.path.rstrip('/')}/{suffix.lstrip('/')}"
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def _authorization_headers(block: Mapping[str, Any]) -> Mapping[str, str]:
    api_key = block.get("api_key")
    if _is_missing(api_key, ("TODO", "CHANGEME", "<required>", "None", "none")):
        return {}
    if str(api_key).lower() in {"ollama", "openai"}:
        return {}
    return {"Authorization": f"Bearer {api_key}"}


def _is_remote_backend(backend: str) -> bool:
    return backend in {
        "openai",
        "ollama",
        "vllm",
        "vllm-server",
        "sglang",
        "sglang-client",
    }


def _is_missing(value: Any, placeholders: Sequence[str]) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return True
        return stripped.casefold() in {item.casefold() for item in placeholders}
    return False


def _plain_mapping(config: Any) -> Mapping[str, Any]:
    if isinstance(config, Mapping):
        return {str(key): _plain_value(value) for key, value in config.items()}
    if is_dataclass(config) and not isinstance(config, type):
        value = asdict(config)
        return {str(key): _plain_value(item) for key, item in value.items()}
    model_dump = getattr(config, "model_dump", None)
    if callable(model_dump):
        value = model_dump(mode="python")
        if isinstance(value, Mapping):
            return {str(key): _plain_value(item) for key, item in value.items()}
    attributes = getattr(config, "__dict__", None)
    if isinstance(attributes, Mapping):
        return {
            str(key): _plain_value(value)
            for key, value in attributes.items()
            if not str(key).startswith("_")
        }
    raise TypeError("config must be a mapping, dataclass, or model-like object")


def _plain_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain_value(item) for key, item in value.items()}
    if is_dataclass(value) and not isinstance(value, type):
        return {str(key): _plain_value(item) for key, item in asdict(value).items()}
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _plain_value(model_dump(mode="python"))
    if isinstance(value, (list, tuple)):
        return [_plain_value(item) for item in value]
    return value


def _walk_mapping_blocks(
    mapping: Mapping[str, Any], prefix: str = ""
) -> Iterable[tuple[str, Mapping[str, Any]]]:
    for key, value in mapping.items():
        dotted = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            yield dotted, value
            yield from _walk_mapping_blocks(value, dotted)


def _deduplicate_configuration(
    specs: Iterable[ConfigurationValueSpec],
) -> Iterable[ConfigurationValueSpec]:
    seen: set[str] = set()
    for spec in specs:
        if spec.name not in seen:
            seen.add(spec.name)
            yield spec


def _deduplicate_paths(
    specs: Iterable[WritablePathSpec],
) -> Iterable[WritablePathSpec]:
    seen: set[str] = set()
    for spec in specs:
        key = str(Path(spec.path).expanduser())
        if key not in seen:
            seen.add(key)
            yield spec


def _directory_configuration_result(name: str, path: Path) -> CheckResult:
    details = {"path": str(path)}
    if not path.is_dir():
        return CheckResult(
            f"config.{name}",
            CheckType.CONFIGURATION,
            Status.RED,
            f"Configured directory does not exist: {path}",
            details,
            f"Correct {name} or create the directory.",
        )
    return CheckResult(
        f"config.{name}",
        CheckType.CONFIGURATION,
        Status.GREEN,
        f"Configured directory exists: {path}",
        details,
    )


def _project_layout_result(path: Path) -> CheckResult:
    required = ("main.py", "Scripts/index_pdf.sh", "config")
    missing = [name for name in required if not (path / name).exists()]
    details = {"path": str(path), "required": list(required), "missing": missing}
    if missing:
        return CheckResult(
            "config.project_layout",
            CheckType.CONFIGURATION,
            Status.RED,
            "Project root is not a usable DocARAG checkout",
            details,
            "Set runtime.project_root to the DocARAG repository containing main.py, Scripts/, and config/.",
        )
    return CheckResult(
        "config.project_layout",
        CheckType.CONFIGURATION,
        Status.GREEN,
        "Project root contains the required compatibility runtime",
        details,
    )


def _file_configuration_result(name: str, path: Path) -> CheckResult:
    details = {"path": str(path)}
    if not path.is_file():
        return CheckResult(
            f"config.{name}",
            CheckType.CONFIGURATION,
            Status.RED,
            f"Configured file does not exist: {path}",
            details,
            f"Correct {name} or create the file.",
        )
    try:
        with path.open("rb") as handle:
            handle.read(1)
    except OSError as exc:
        details["error"] = str(exc)
        return CheckResult(
            f"config.{name}",
            CheckType.CONFIGURATION,
            Status.RED,
            f"Configured file is not readable: {path}",
            details,
            f"Grant the current user read access to {path}.",
        )
    return CheckResult(
        f"config.{name}",
        CheckType.CONFIGURATION,
        Status.GREEN,
        f"Configured file is readable: {path}",
        details,
    )


def _load_mapping_config(path: Path) -> Mapping[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    elif suffix == ".toml":
        with path.open("rb") as handle:
            value = tomllib.load(handle)
    else:
        value = _parse_simple_yaml(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("Top-level configuration must be a mapping")
    return {str(key): item for key, item in value.items()}


def _parse_simple_yaml(text: str) -> Mapping[str, Any]:
    """Parse the mapping/scalar YAML subset used by DocARAG configs.

    This is intentionally not a general YAML implementation.  Lists, anchors,
    tags, block scalars, and flow collections are rejected with an actionable
    error instead of being interpreted ambiguously by the setup command.
    """

    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for line_number, original in enumerate(text.splitlines(), start=1):
        if "\t" in original[: len(original) - len(original.lstrip())]:
            raise ValueError(f"Tabs are not supported for indentation (line {line_number})")
        line = _strip_yaml_comment(original).rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        content = line.strip()
        if content.startswith("-"):
            raise ValueError(f"YAML lists are not supported (line {line_number})")
        if ":" not in content:
            raise ValueError(f"Expected key: value mapping (line {line_number})")
        key_text, raw_value = content.split(":", 1)
        key = key_text.strip().strip("'\"")
        if not key:
            raise ValueError(f"Empty mapping key (line {line_number})")
        while stack and indent <= stack[-1][0]:
            stack.pop()
        if not stack:
            raise ValueError(f"Invalid indentation (line {line_number})")
        parent = stack[-1][1]
        if key in parent:
            raise ValueError(f"Duplicate key {key!r} (line {line_number})")
        raw_value = raw_value.strip()
        if not raw_value:
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
            continue
        if raw_value.startswith(("[", "{", "|", ">", "&", "*", "!")):
            raise ValueError(
                f"Unsupported YAML construct for {key!r} (line {line_number})"
            )
        parent[key] = _parse_yaml_scalar(raw_value)
    return root


def _strip_yaml_comment(line: str) -> str:
    quote: Optional[str] = None
    escaped = False
    for index, character in enumerate(line):
        if escaped:
            escaped = False
            continue
        if character == "\\" and quote == '"':
            escaped = True
            continue
        if character in {"'", '"'}:
            if quote is None:
                quote = character
            elif quote == character:
                quote = None
            continue
        if character == "#" and quote is None and (
            index == 0 or line[index - 1].isspace()
        ):
            return line[:index]
    if quote is not None:
        raise ValueError("Unterminated quoted YAML scalar")
    return line


def _parse_yaml_scalar(raw: str) -> Any:
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in {"'", '"'}:
        if raw[0] == '"':
            try:
                return json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid quoted scalar: {exc}") from exc
        return raw[1:-1].replace("''", "'")
    lowered = raw.casefold()
    if lowered in {"null", "~"}:
        return None
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        return raw


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item) for item in value]
    if is_dataclass(value) and not isinstance(value, type):
        return _json_safe(asdict(value))
    return str(value)


__all__ = [
    "CheckResult",
    "CheckType",
    "CommandOutput",
    "CommandSpec",
    "ConfigurationValueSpec",
    "DiagnosticProbes",
    "DiagnosticsOptions",
    "DiagnosticsReport",
    "EndpointSpec",
    "GpuSpec",
    "HttpResponse",
    "ModelEndpointSpec",
    "PathProbeResult",
    "PythonSpec",
    "Status",
    "WritablePathSpec",
    "doctor_for_settings",
    "options_from_docarag_config",
    "render_report",
    "run_diagnostics",
]
