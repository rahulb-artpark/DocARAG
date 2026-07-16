"""Read-only, dependency-light setup planning for DocARAG operators.

The setup plan is deliberately not an installer.  It inspects only local files,
the current process, and executable discovery; it never starts a subprocess,
contacts an endpoint, writes a file, starts a model, or changes GPU state.
Commands are represented as argument vectors so callers do not need to parse or
interpolate shell text.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any

from docarag.runtime_compatibility import (
    DEFAULT_MANIFEST_PATH,
    RuntimeCompatibilityManifest,
    load_runtime_compatibility_manifest,
)
from docarag.settings import RuntimeSettings


SETUP_PLAN_SCHEMA_VERSION = "docarag.setup-plan.v1"
_MAX_ENVIRONMENT_SPEC_BYTES = 64 * 1024


def _command(
    command_id: str,
    argv: Sequence[str],
    *,
    purpose: str,
    state_behavior: str,
    network_behavior: str = "none",
    condition: str | None = None,
    positional_arguments: Sequence[Mapping[str, object]] = (),
) -> dict[str, object]:
    result: dict[str, object] = {
        "command_id": command_id,
        "argv": list(argv),
        "purpose": purpose,
        "state_behavior": state_behavior,
        "network_behavior": network_behavior,
    }
    if condition is not None:
        result["condition"] = condition
    if positional_arguments:
        result["positional_arguments"] = [dict(item) for item in positional_arguments]
    return result


def _prerequisite(
    check_id: str,
    status: str,
    message: str,
    *,
    required: bool,
    remediation: str | None = None,
    details: Mapping[str, object] | None = None,
) -> dict[str, object]:
    result: dict[str, object] = {
        "check_id": check_id,
        "status": status,
        "required": required,
        "message": message,
    }
    if remediation:
        result["remediation"] = remediation
    if details:
        result["details"] = dict(details)
    return result


def _conda_environment_name(path: Path) -> str:
    """Read the one top-level ``name`` scalar from a bounded Conda YAML file."""

    size = path.stat().st_size
    if size > _MAX_ENVIRONMENT_SPEC_BYTES:
        raise ValueError(
            f"environment specification exceeds {_MAX_ENVIRONMENT_SPEC_BYTES} bytes"
        )
    text = path.read_text(encoding="utf-8")
    for raw_line in text.splitlines():
        if not raw_line or raw_line[0].isspace() or raw_line.lstrip().startswith("#"):
            continue
        key, separator, raw_value = raw_line.partition(":")
        if separator and key.strip() == "name":
            value = raw_value.split("#", 1)[0].strip().strip("'\"")
            if not value:
                break
            return value
    raise ValueError("environment specification has no non-empty top-level name")


def _cli_prefix(settings: RuntimeSettings) -> list[str]:
    prefix = ["docarag"]
    if settings.settings_path is not None:
        prefix.extend(("--settings", str(settings.settings_path)))
    return prefix


def _load_manifest_for_plan(
    path: Path,
) -> tuple[RuntimeCompatibilityManifest | None, dict[str, object]]:
    try:
        manifest = load_runtime_compatibility_manifest(path)
    except (OSError, ValueError) as exc:
        return None, _prerequisite(
            "runtime_compatibility_manifest",
            "blocked",
            "Runtime compatibility manifest is missing or invalid",
            required=True,
            remediation="Restore a valid packaged runtime_compatibility.toml manifest.",
            details={"path": str(path), "error": str(exc)},
        )
    return manifest, _prerequisite(
        "runtime_compatibility_manifest",
        "ready",
        "Runtime compatibility declarations are readable and structurally valid",
        required=True,
        details={
            "path": str(path),
            "manifest_revision": manifest.manifest_revision,
            "source_sha256": manifest.source_sha256,
        },
    )


def _manifest_components(
    manifest: RuntimeCompatibilityManifest | None,
) -> list[dict[str, object]]:
    if manifest is None:
        return []
    components: list[dict[str, object]] = []
    for runtime in manifest.parser_workers:
        components.append(
            {
                "component_id": runtime.runtime_id,
                "role": "routed-parser-worker",
                "parser_name": runtime.parser_name,
                "adapter_name": runtime.adapter_name,
                "adapter_versions": list(runtime.adapter_versions),
                "ownership": "operator-managed external process",
                "accelerator": runtime.accelerator.value,
                "available_to_supported_index_command": True,
                "verification_level": runtime.verification_level.value,
                "production_verified": runtime.is_production_verified,
                "minimum_vram_certified": runtime.minimum_vram_certified,
                "uncertified_artifacts": list(runtime.uncertified_artifacts),
                "verification_notes": runtime.verification_notes,
            }
        )
    return components


def _routed_section(profile: Any | None, environment: Mapping[str, str]) -> dict[str, object]:
    if profile is None:
        return {
            "profile_supplied": False,
            "indexing_enabled": True,
            "selected": False,
            "reason": (
                "Routed indexing is available; supply --routed-profile PATH to "
                "select and revision-bind an explicit worker profile."
            ),
        }

    issues = [issue.to_dict() for issue in profile.configuration_issues()]
    workers: list[dict[str, object]] = []
    for endpoint in profile.config.endpoints:
        token_name = endpoint.bearer_token_env
        workers.append(
            {
                "runtime_id": endpoint.runtime_id,
                "adapter_name": endpoint.adapter_name,
                "base_url": endpoint.base_url,
                "expected_adapter_version": endpoint.expected_adapter_version,
                "expected_manifest_sha256": endpoint.expected_manifest_sha256,
                "bearer_token_env": token_name,
                "bearer_token_configured": (
                    bool(environment.get(token_name, "")) if token_name else True
                ),
                "ownership": "operator-managed external process",
                "live_readiness_checked": False,
            }
        )
    return {
        "profile_supplied": True,
        "indexing_enabled": True,
        "selected": True,
        "reason": (
            "The supported docarag index command binds this profile's exact bytes "
            "and normalized configuration to the index revision."
        ),
        "profile_id": profile.profile_id,
        "source_path": str(profile.source_path),
        "source_sha256": profile.source_sha256,
        "configuration_sha256": profile.configuration_sha256,
        "configuration_ready": not issues,
        "runtime_compatibility_checked": False,
        "live_readiness_checked": False,
        "configuration_issues": issues,
        "workers": workers,
    }


def build_setup_plan(
    settings: RuntimeSettings,
    *,
    routed_profile: Any | None = None,
    environment: Mapping[str, str] | None = None,
    executable_lookup: Callable[[str], str | None] = shutil.which,
    compatibility_manifest_path: Path | None = None,
) -> dict[str, object]:
    """Build a stable, secret-free setup plan without probing live services.

    ``executable_lookup`` and ``environment`` are injectable to keep policy
    tests deterministic.  Environment values are never serialized; routed
    profiles expose only a configured/not-configured boolean for named bearer
    variables.
    """

    if not isinstance(settings, RuntimeSettings):
        raise TypeError("settings must be RuntimeSettings")
    settings = settings.resolved()
    active_environment = os.environ if environment is None else environment
    root = settings.project_root
    environment_spec = root / "environment.yml"
    requirements = root / "requirements.txt"
    pyproject = root / "pyproject.toml"
    compose_file = root / "compose.yaml"
    manifest_path = (compatibility_manifest_path or DEFAULT_MANIFEST_PATH).resolve()
    cli_prefix = _cli_prefix(settings)

    commands: list[dict[str, object]] = [
        _command(
            "install_controller",
            (
                sys.executable,
                "-m",
                "pip",
                "install",
                str(root),
            ),
            purpose=(
                "Install the controller plus its lightweight Qdrant/PostgreSQL clients."
            ),
            state_behavior="persistent",
            network_behavior="configuration-dependent-build-isolation",
            condition="Use when the docarag command is not installed in this interpreter.",
        ),
        _command(
            "start_default_storage",
            ("docker", "compose", "--file", str(compose_file), "up", "--detach", "--wait"),
            purpose="Start the default PostgreSQL ledger and Qdrant vector database.",
            state_behavior="persistent",
            network_behavior="required-on-first-image-pull",
            condition=(
                "Use for the loopback-only local profile; externally managed "
                "PostgreSQL and Qdrant services may be configured instead."
            ),
        ),
        _command(
            "initialize_default_database",
            ("docarag", "storage", "init"),
            purpose="Create the PostgreSQL schema if it is not present.",
            state_behavior="persistent",
            network_behavior="configured-postgresql",
        ),
        _command(
            "create_compatibility_environment",
            ("conda", "env", "create", "--file", str(environment_spec)),
            purpose=(
                "Create the version-pinned research compatibility environment; "
                "this input is not a hashed lock or clean-host qualification."
            ),
            state_behavior="persistent",
            network_behavior="required",
            condition="Use once for a fresh environment; this is not an update command.",
        ),
    ]
    if settings.settings_path is None:
        commands.append(
            _command(
                "initialize_settings",
                (
                    "docarag",
                    "init",
                    "--path",
                    str(Path.cwd().resolve() / "docarag.toml"),
                    "--project-root",
                    str(root),
                ),
                purpose="Create an explicit local settings source of truth.",
                state_behavior="persistent",
                condition="Recommended because this invocation is using implicit defaults.",
            )
        )
    commands.extend(
        (
            _command(
                "offline_doctor",
                (*cli_prefix, "doctor", "--offline", "--strict", "--json"),
                purpose=(
                    "Validate the interpreter, dependencies, config, writable paths, "
                    "package consistency, and GPU headroom without contacting services."
                ),
                state_behavior="transient-writability-probe",
            ),
            _command(
                "live_doctor",
                (*cli_prefix, "doctor", "--strict", "--json"),
                purpose="Validate configured model/service inventories after offline checks pass.",
                state_behavior="transient-writability-probe",
                network_behavior="configuration-dependent",
            ),
            _command(
                "index_pdf",
                (*cli_prefix, "index"),
                purpose="Index one PDF through the supported compatibility BookRAG path.",
                state_behavior="persistent",
                positional_arguments=(
                    {
                        "name": "document",
                        "type": "absolute PDF path",
                        "required": True,
                    },
                ),
                network_behavior="configuration-dependent",
            ),
        )
    )
    if routed_profile is not None:
        routed_path = str(routed_profile.source_path)
        commands.extend(
            (
                _command(
                    "validate_routed_profile",
                    ("docarag", "routed-profile", "validate", routed_path, "--json"),
                    purpose="Validate exact static worker pins without contacting workers.",
                    state_behavior="none",
                ),
                _command(
                    "offline_routed_doctor",
                    (
                        *cli_prefix,
                        "doctor",
                        "--routed-profile",
                        routed_path,
                        "--offline",
                        "--strict",
                        "--json",
                    ),
                    purpose=(
                        "Check routed profile pins, secret-variable presence, and "
                        "compatibility declarations without contacting workers."
                    ),
                    state_behavior="transient-writability-probe",
                ),
                _command(
                    "live_routed_doctor",
                    (
                        *cli_prefix,
                        "doctor",
                        "--routed-profile",
                        routed_path,
                        "--strict",
                        "--json",
                    ),
                    purpose="Check live worker health and exact capability identity.",
                    state_behavior="transient-writability-probe",
                    network_behavior="required",
                ),
            )
        )

    prerequisites: list[dict[str, object]] = []
    current_python_ready = (3, 11) <= sys.version_info[:2] < (3, 14)
    prerequisites.append(
        _prerequisite(
            "controller_python",
            "ready" if current_python_ready else "blocked",
            (
                "Controller is running on supported Python 3.11-3.13"
                if current_python_ready
                else "Controller requires Python 3.11, 3.12, or 3.13"
            ),
            required=True,
            remediation=(
                None
                if current_python_ready
                else "Run the controller CLI from a Python 3.11-3.13 interpreter."
            ),
            details={"executable": sys.executable, "version": list(sys.version_info[:3])},
        )
    )
    required_layout = ("main.py", "Scripts/index_pdf.sh", "config")
    missing_layout = [name for name in required_layout if not (root / name).exists()]
    prerequisites.append(
        _prerequisite(
            "project_layout",
            "blocked" if missing_layout else "ready",
            (
                "Project root contains the compatibility runtime"
                if not missing_layout
                else "Project root is missing compatibility runtime files"
            ),
            required=True,
            remediation=(
                None
                if not missing_layout
                else "Point runtime.project_root at a complete DocARAG checkout."
            ),
            details={"path": str(root), "missing": missing_layout},
        )
    )
    for check_id, path, label in (
        ("controller_package_manifest", pyproject, "pyproject.toml"),
        ("compatibility_environment_spec", environment_spec, "environment.yml"),
        ("compatibility_requirements", requirements, "requirements.txt"),
        ("system_config", settings.system_config, "system configuration"),
        ("default_storage_compose", compose_file, "default storage Compose file"),
    ):
        exists = path.is_file()
        prerequisites.append(
            _prerequisite(
                check_id,
                "ready" if exists else "blocked",
                f"{label} is {'available' if exists else 'missing'}",
                required=True,
                remediation=None if exists else f"Restore or configure {label}.",
                details={"path": str(path)},
            )
        )

    declared_environment: str | None = None
    if environment_spec.is_file():
        try:
            declared_environment = _conda_environment_name(environment_spec)
        except (OSError, UnicodeError, ValueError) as exc:
            prerequisites.append(
                _prerequisite(
                    "compatibility_environment_name",
                    "blocked",
                    "Conda environment name could not be read deterministically",
                    required=settings.python_executable is None,
                    remediation="Declare one top-level name in environment.yml.",
                    details={"error": str(exc)},
                )
            )
        else:
            name_matches = declared_environment == settings.conda_environment
            prerequisites.append(
                _prerequisite(
                    "compatibility_environment_name",
                    (
                        "ready"
                        if name_matches or settings.python_executable is not None
                        else "blocked"
                    ),
                    (
                        "Conda environment name matches runtime settings"
                        if name_matches
                        else (
                            "Explicit runtime Python makes the Conda name informational"
                            if settings.python_executable is not None
                            else "Conda environment name differs from runtime settings"
                        )
                    ),
                    required=settings.python_executable is None,
                    remediation=(
                        None
                        if name_matches or settings.python_executable is not None
                        else "Align runtime.conda_environment with environment.yml."
                    ),
                    details={
                        "declared": declared_environment,
                        "configured": settings.conda_environment,
                    },
                )
            )

    if settings.python_executable is not None:
        runtime_path = Path(settings.python_executable).expanduser()
        executable_ready = runtime_path.is_file() and os.access(runtime_path, os.X_OK)
        prerequisites.append(
            _prerequisite(
                "compatibility_runtime_selector",
                "ready" if executable_ready else "blocked",
                (
                    "Explicit compatibility interpreter is executable"
                    if executable_ready
                    else "Explicit compatibility interpreter is unavailable"
                ),
                required=True,
                remediation=(
                    None
                    if executable_ready
                    else "Correct runtime.python_executable in docarag.toml."
                ),
                details={"python_command": settings.python_command()},
            )
        )
    elif active_environment.get("CONDA_DEFAULT_ENV") == settings.conda_environment:
        prerequisites.append(
            _prerequisite(
                "compatibility_runtime_selector",
                "ready",
                "Configured Conda environment is active in this process",
                required=True,
                details={"python_command": settings.python_command()},
            )
        )
    else:
        conda_path = executable_lookup("conda")
        prerequisites.append(
            _prerequisite(
                "compatibility_runtime_selector",
                "deferred" if conda_path else "blocked",
                (
                    "Conda is available; environment existence is delegated to offline doctor"
                    if conda_path
                    else "Conda is unavailable and no explicit runtime Python is configured"
                ),
                required=True,
                remediation=(
                    None
                    if conda_path
                    else "Install Conda or set runtime.python_executable explicitly."
                ),
                details={
                    "python_command": settings.python_command(),
                    "conda_executable": conda_path,
                },
            )
        )

    manifest, manifest_check = _load_manifest_for_plan(manifest_path)
    prerequisites.append(manifest_check)
    if settings.settings_path is None:
        prerequisites.append(
            _prerequisite(
                "settings_source_of_truth",
                "recommended",
                "Implicit defaults are active; create docarag.toml for reproducibility",
                required=False,
                remediation="Run the initialize_settings argv from this plan.",
            )
        )
    else:
        prerequisites.append(
            _prerequisite(
                "settings_source_of_truth",
                "ready",
                "An explicit settings file is selected",
                required=False,
                details={"path": str(settings.settings_path)},
            )
        )

    routed = _routed_section(routed_profile, active_environment)
    if routed_profile is not None:
        routed_issues = routed["configuration_issues"]
        if not isinstance(routed_issues, list):
            raise TypeError("routed configuration issues must be a list")
        prerequisites.append(
            _prerequisite(
                "routed_profile_static_configuration",
                "blocked" if routed_issues else "ready",
                (
                    "Routed profile has exact, non-placeholder static worker pins"
                    if not routed_issues
                    else "Routed profile contains placeholder or mutable worker pins"
                ),
                required=True,
                remediation=(
                    None
                    if not routed_issues
                    else "Replace every reported adapter version and manifest digest with values from a qualified worker."
                ),
                details={
                    "issue_count": len(routed_issues),
                    "issue_codes": [item["code"] for item in routed_issues],
                },
            )
        )
        routed_workers = routed["workers"]
        if not isinstance(routed_workers, list):
            raise TypeError("routed workers must be a list")
        for worker in routed_workers:
            token_name = worker["bearer_token_env"]
            if token_name is None:
                continue
            configured = bool(worker["bearer_token_configured"])
            runtime_id = str(worker["runtime_id"])
            prerequisites.append(
                _prerequisite(
                    f"routed_worker.{runtime_id}.bearer_token_env",
                    "ready" if configured else "blocked",
                    (
                        "Configured routed-worker bearer variable is present"
                        if configured
                        else "Configured routed-worker bearer variable is absent"
                    ),
                    required=True,
                    remediation=(
                        None
                        if configured
                        else f"Set {token_name} without placing its value in the profile."
                    ),
                    details={"environment_variable": token_name},
                )
            )
    controller_runtime = next(
        (
            runtime
            for runtime in manifest.runtimes
            if runtime.runtime_kind.value == "controller"
        ),
        None,
    ) if manifest is not None else None
    components: list[dict[str, object]] = [
        {
            "component_id": "controller-cli",
            "role": "dependency-light orchestration and diagnostics",
            "ownership": "current docarag process",
            "dependency_source": str(pyproject),
            "accelerator": "none",
            "available_to_supported_index_command": True,
            "verification_level": (
                controller_runtime.verification_level.value
                if controller_runtime is not None
                else "unknown"
            ),
            "production_verified": bool(
                controller_runtime is not None
                and controller_runtime.is_production_verified
            ),
        },
        {
            "component_id": "postgresql-ledger",
            "role": "default metadata, work, trace, and index-adoption system of record",
            "ownership": "operator-managed service (local Compose profile available)",
            "dependency_source": str(compose_file),
            "accelerator": "none",
            "available_to_supported_index_command": True,
            "production_verified": False,
        },
        {
            "component_id": "qdrant-vector-database",
            "role": "sole dense/multimodal vector index and filtered similarity search",
            "ownership": "operator-managed service (local Compose profile available)",
            "dependency_source": str(compose_file),
            "accelerator": "none",
            "available_to_supported_index_command": True,
            "production_verified": False,
        },
        {
            "component_id": "bookrag-compatibility-runtime",
            "role": "primary indexing and retrieval runtime",
            "ownership": "operator-managed Conda environment or explicit interpreter",
            "dependency_source": str(requirements),
            "dependency_lock": False,
            "clean_host_qualified": False,
            "python_command": settings.python_command(),
            "accelerator": "configuration-dependent",
            "available_to_supported_index_command": True,
            "production_verified": False,
            "qualification_note": (
                "This is the supported compatibility path, but its local/remote models "
                "and hardware still require doctor and corpus qualification. The "
                "version pins are not a hashed dependency lock."
            ),
        },
        *_manifest_components(manifest),
    ]

    blockers = [item for item in prerequisites if item["status"] == "blocked"]
    recommendations = [
        item for item in prerequisites if item["status"] == "recommended"
    ]
    command_by_id = {item["command_id"]: item for item in commands}
    if blockers:
        first = blockers[0]
        next_action: dict[str, object] = {
            "kind": "remediate_static_blocker",
            "check_id": first["check_id"],
            "message": first["message"],
            "remediation": first.get("remediation"),
        }
    elif recommendations:
        first = recommendations[0]
        next_action = {
            "kind": "recommended_reproducibility_step",
            "check_id": first["check_id"],
            "message": first["message"],
            "command": command_by_id.get("initialize_settings"),
        }
    else:
        next_action = {
            "kind": "run_offline_doctor",
            "message": "Static prerequisites are ready; run bounded offline diagnostics.",
            "command": command_by_id["offline_doctor"],
        }

    counts = {
        status: sum(item["status"] == status for item in prerequisites)
        for status in ("ready", "deferred", "recommended", "blocked")
    }
    return {
        "schema_version": SETUP_PLAN_SCHEMA_VERSION,
        "report_behavior": {
            "writes_files": False,
            "starts_subprocesses": False,
            "contacts_network": False,
            "starts_or_stops_services": False,
            "changes_gpu_state": False,
            "resolves_secret_values": False,
        },
        "supported_runtime": {
            "profile": "BookRAG with optional routed canonical ingestion",
            "core": "BookRAG",
            "indexing_enabled": True,
            "routed_indexing_enabled": True,
        },
        "settings": settings.as_dict(),
        "static_readiness": {
            "ready_for_offline_doctor": not blockers,
            "counts": counts,
            "prerequisites": prerequisites,
        },
        "next_action": next_action,
        "components": components,
        "routed_runtime": routed,
        "resource_ownership": {
            "model_and_parser_services": "operator-managed external processes",
            "docarag_starts_or_stops_services": False,
            "gpu_scheduler_available": False,
            "automatic_gpu_memory_arbitration": False,
            "conflict_risk": "shared-GPU contention is possible",
            "operator_rule": (
                "Do not co-locate local BookRAG models, Ollama, vLLM, MinerU, or "
                "PaddleOCR-VL workers until measured peak VRAM plus headroom fits."
            ),
            "live_gpu_inventory_checked": False,
        },
        "deferred_validation": {
            "offline_doctor": [
                "compatibility interpreter is exactly Python 3.12",
                "required runtime imports are available",
                "pip package consistency",
                "controller build requirements when installation is requested",
                "configuration syntax and placeholders",
                "runs directory writability",
                "GPU inventory and at least 6144 MiB request headroom",
            ],
            "live_doctor": [
                "PostgreSQL connectivity and schema state",
                "Qdrant health, collection access, and scoped readiness",
                "configured model inventories",
                "configured service endpoints",
                "routed worker health and exact capability identity when requested",
            ],
        },
        "commands": commands,
    }


def render_setup_plan(plan: Mapping[str, object]) -> str:
    """Render a concise operator view while keeping argv boundaries explicit."""

    readiness = plan["static_readiness"]
    if not isinstance(readiness, Mapping):
        raise TypeError("plan.static_readiness must be a mapping")
    counts = readiness["counts"]
    if not isinstance(counts, Mapping):
        raise TypeError("plan.static_readiness.counts must be a mapping")
    state = "READY FOR OFFLINE DOCTOR" if readiness["ready_for_offline_doctor"] else "BLOCKED"
    lines = [
        f"DocARAG setup plan: {state}",
        "Supported index runtime: BookRAG with optional routed canonical ingestion",
        "Routed parser profiles: select with docarag index --routed-profile PATH",
        (
            "Static prerequisites: "
            f"{counts['ready']} ready, {counts['deferred']} deferred, "
            f"{counts['recommended']} recommended, {counts['blocked']} blocked"
        ),
    ]
    next_action = plan["next_action"]
    if not isinstance(next_action, Mapping):
        raise TypeError("plan.next_action must be a mapping")
    lines.append(f"Next: {next_action['message']}")
    command = next_action.get("command")
    if isinstance(command, Mapping):
        lines.append("argv: " + json.dumps(command["argv"], ensure_ascii=False))
    elif next_action.get("remediation"):
        lines.append(f"Fix: {next_action['remediation']}")
    lines.append(
        "This report made no writes, network calls, subprocess calls, or GPU/service changes."
    )
    lines.append("Use --json for the full command, ownership, and qualification plan.")
    return "\n".join(lines)


__all__ = [
    "SETUP_PLAN_SCHEMA_VERSION",
    "build_setup_plan",
    "render_setup_plan",
]
