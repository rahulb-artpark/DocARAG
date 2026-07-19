"""Supported DocARAG command line.

The top-level module remains standard-library only. Heavy BookRAG dependencies
are loaded in a configured compatibility interpreter and only for commands that
actually need them.
"""

from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import shlex
import stat
import subprocess
import sys
import time
from typing import Iterable, Sequence
import uuid
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen

from docarag import __version__
from docarag.settings import (
    DEFAULT_FILENAME,
    RuntimeSettings,
    initialize_settings,
    load_settings,
    repository_root,
)


class CLIError(RuntimeError):
    pass


_PROJECT_ENV_KEYS = frozenset(
    {
        "DOCARAG_DATABASE_URL",
        "DOCARAG_POSTGRES_DB",
        "DOCARAG_POSTGRES_USER",
        "DOCARAG_POSTGRES_PASSWORD",
        "DOCARAG_POSTGRES_PORT",
        "DOCARAG_QDRANT_HTTP_PORT",
        "DOCARAG_QDRANT_GRPC_PORT",
        "DOCARAG_QDRANT_URL",
        "QDRANT_API_KEY",
    }
)
_MAX_PROJECT_ENV_BYTES = 64 * 1024


def _load_project_environment(project_root: Path) -> None:
    """Load the allowlisted local storage environment without executing it.

    Real process environment values always win.  This mirrors Docker Compose's
    project ``.env`` behavior so the CLI, BookRAG subprocess, PostgreSQL client,
    and Qdrant client use the same ports and endpoints.
    """

    path = project_root / ".env"
    if not path.exists():
        return
    if path.is_symlink() or not path.is_file():
        raise CLIError(f"Project environment file is unsafe: {path}")
    if path.stat().st_size > _MAX_PROJECT_ENV_BYTES:
        raise CLIError(f"Project environment file is too large: {path}")
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise CLIError(f"Could not read project environment file: {path}") from exc
    parsed: dict[str, str] = {}
    for line_number, raw_line in enumerate(lines, start=1):
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("export "):
            stripped = stripped[7:].lstrip()
        if "=" not in stripped:
            raise CLIError(f"Invalid .env assignment at {path}:{line_number}")
        key, raw_value = stripped.split("=", 1)
        key = key.strip()
        if key not in _PROJECT_ENV_KEYS:
            continue
        value = raw_value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        if not value:
            raise CLIError(f"Empty .env value for {key} at {path}:{line_number}")
        parsed[key] = value
    for key, value in parsed.items():
        os.environ.setdefault(key, value)


def _run(
    command: Sequence[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
) -> int:
    try:
        completed = subprocess.run(list(command), cwd=cwd, env=env, check=False)
    except FileNotFoundError as exc:
        raise CLIError(
            f"Required command was not found: {command[0]}. "
            "Run `docarag doctor` for remediation."
        ) from exc
    return completed.returncode


def _load(args: argparse.Namespace) -> RuntimeSettings:
    settings = load_settings(args.settings)
    _load_project_environment(settings.project_root)
    return settings


def command_init(args: argparse.Namespace) -> int:
    destination = Path(args.path or DEFAULT_FILENAME)
    target = initialize_settings(
        destination,
        project_root=(
            Path(args.project_root).expanduser()
            if args.project_root
            else Path.cwd()
        ),
        force=args.force,
    )
    print(f"Created {target}")
    print(
        "Next: docarag --settings "
        + shlex.quote(str(target))
        + " doctor --offline"
    )
    return 0


def command_settings(args: argparse.Namespace) -> int:
    settings = _load(args)
    print(json.dumps(settings.as_dict(), indent=2, sort_keys=True))
    return 0


def command_setup_plan(args: argparse.Namespace) -> int:
    """Explain a setup without installing, probing services, or changing state."""

    try:
        from docarag.setup_plan import build_setup_plan, render_setup_plan
    except ImportError as exc:
        raise CLIError(f"Setup planning is unavailable: {exc}") from exc

    settings = _load(args)
    profile = (
        _load_routed_profile(args.routed_profile)
        if args.routed_profile
        else None
    )
    plan = build_setup_plan(settings, routed_profile=profile)
    if args.json:
        print(json.dumps(plan, indent=2, ensure_ascii=False, sort_keys=True))
    else:
        print(render_setup_plan(plan))
    return 0 if plan["static_readiness"]["ready_for_offline_doctor"] else 1


def _postgres_ledger(settings: RuntimeSettings):
    try:
        from docarag.persistence import PostgresLedger
    except ImportError as exc:
        raise CLIError(
            "PostgreSQL support is unavailable; install the project dependencies"
        ) from exc
    try:
        return PostgresLedger(settings.database_url)
    except (OSError, RuntimeError, ValueError) as exc:
        raise CLIError(
            "PostgreSQL configuration is invalid "
            f"({type(exc).__name__}); check {settings.database_url_env}"
        ) from exc


def _require_postgres(settings: RuntimeSettings):
    ledger = _postgres_ledger(settings)
    health = ledger.health()
    if not health.reachable:
        raise CLIError(
            f"PostgreSQL is unreachable at {health.endpoint}; run "
            "`docarag storage up` or configure DOCARAG_DATABASE_URL"
        )
    if not health.schema.current:
        detail = health.schema.errors[0] if health.schema.errors else (
            "run `docarag storage init`"
        )
        raise CLIError(f"PostgreSQL schema is not current; {detail}")
    return ledger


def _compose_command(settings: RuntimeSettings, *arguments: str) -> int:
    compose_file = settings.project_root / "compose.yaml"
    if not compose_file.is_file():
        raise CLIError(f"Default storage definition not found: {compose_file}")
    return _run(
        ("docker", "compose", "--file", str(compose_file), *arguments),
        cwd=settings.project_root,
        env=os.environ.copy(),
    )


def command_storage_up(args: argparse.Namespace) -> int:
    settings = _load(args)
    return_code = _compose_command(settings, "up", "--detach", "--wait")
    if return_code != 0 or args.no_init:
        return return_code
    ledger = _postgres_ledger(settings)
    try:
        status = ledger.ensure_schema()
    except Exception as exc:
        raise CLIError(
            f"PostgreSQL schema initialization failed at {ledger.endpoint} "
            f"({type(exc).__name__})"
        ) from exc
    qdrant = _qdrant_health()
    for _attempt in range(19):
        if qdrant["ok"]:
            break
        time.sleep(0.25)
        qdrant = _qdrant_health()
    if not qdrant["ok"]:
        raise CLIError(
            f"Qdrant did not become ready at {qdrant['endpoint']} "
            f"({qdrant.get('error_code', 'health_check_failed')})"
        )
    print(
        "Default storage is running; PostgreSQL schema is "
        + ("ready" if status.current else "not current")
    )
    return 0 if status.current else 1


def command_storage_down(args: argparse.Namespace) -> int:
    settings = _load(args)
    return _compose_command(settings, "down")


def _print_schema_status(ledger, status) -> int:
    print(
        f"PostgreSQL schema at {ledger.endpoint}: "
        + ("ready" if status.current else "not current")
    )
    if not status.current and status.errors:
        print("  " + status.errors[0])
    return 0 if status.current else 1


def command_storage_init(args: argparse.Namespace) -> int:
    settings = _load(args)
    ledger = _postgres_ledger(settings)
    try:
        status = ledger.ensure_schema()
    except Exception as exc:
        raise CLIError(
            f"PostgreSQL schema initialization failed at {ledger.endpoint} "
            f"({type(exc).__name__})"
        ) from exc
    if args.json:
        print(json.dumps({"endpoint": ledger.endpoint, **status.as_dict()}, indent=2, sort_keys=True))
        return 0 if status.current else 1
    return _print_schema_status(ledger, status)


def command_storage_reset(args: argparse.Namespace) -> int:
    settings = _load(args)
    if not args.yes:
        raise CLIError(
            "`storage reset` drops the entire docarag schema and all its data; "
            "pass --yes to confirm"
        )
    ledger = _postgres_ledger(settings)
    try:
        status = ledger.reset_schema()
    except Exception as exc:
        raise CLIError(
            f"PostgreSQL schema reset failed at {ledger.endpoint} "
            f"({type(exc).__name__})"
        ) from exc
    if args.json:
        print(json.dumps({"endpoint": ledger.endpoint, **status.as_dict()}, indent=2, sort_keys=True))
        return 0 if status.current else 1
    return _print_schema_status(ledger, status)


def _qdrant_health() -> dict[str, object]:
    configured = os.environ.get(
        "DOCARAG_QDRANT_URL", "http://127.0.0.1:6333"
    )
    parsed = urlsplit(configured)
    try:
        parsed_hostname = parsed.hostname
        parsed_port = parsed.port
    except ValueError:
        parsed_hostname = None
        parsed_port = None
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed_hostname
        or parsed.username is not None
        or parsed.password is not None
    ):
        return {
            "ok": False,
            "endpoint": "<invalid>",
            "error_code": "invalid_qdrant_url",
        }
    hostname = parsed_hostname
    if ":" in hostname and not hostname.startswith("["):
        hostname = f"[{hostname}]"
    netloc = hostname + (f":{parsed_port}" if parsed_port is not None else "")
    endpoint = urlunsplit((parsed.scheme, netloc, parsed.path.rstrip("/"), "", ""))
    health_url = endpoint.rstrip("/") + "/healthz"
    headers = {"Accept": "text/plain"}
    api_key = os.environ.get("QDRANT_API_KEY")
    if api_key:
        headers["api-key"] = api_key
    try:
        request = Request(health_url, headers=headers, method="GET")
        with urlopen(request, timeout=5.0) as response:  # nosec B310: validated URL
            response.read(4097)
            status = int(response.status)
        return {
            "ok": 200 <= status < 300,
            "endpoint": endpoint,
            "http_status": status,
        }
    except (HTTPError, URLError, OSError, ValueError) as exc:
        return {
            "ok": False,
            "endpoint": endpoint,
            "error_code": type(exc).__name__,
        }


def command_storage_status(args: argparse.Namespace) -> int:
    settings = _load(args)
    ledger = _postgres_ledger(settings)
    health = ledger.health()
    qdrant = _qdrant_health()
    payload = {"postgres": health.as_dict(), "qdrant": qdrant}
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        state = "ready" if health.ok else "not ready"
        print(f"PostgreSQL at {health.endpoint}: {state}")
        if health.reachable:
            schema_state = "ready" if health.schema.current else "not current"
            print(f"Schema: {schema_state}")
            if not health.schema.current and health.schema.errors:
                print("  " + health.schema.errors[0])
        elif health.error_code:
            print(f"Failure class: {health.error_code}")
        qdrant_state = "ready" if qdrant["ok"] else "not ready"
        print(f"Qdrant at {qdrant['endpoint']}: {qdrant_state}")
        if not qdrant["ok"] and qdrant.get("error_code"):
            print(f"Failure class: {qdrant['error_code']}")
    return 0 if health.ok and qdrant["ok"] else 1


def _stable_file_sha256(path: Path) -> str:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise CLIError(f"Expected a regular file: {path}")
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        after = os.fstat(descriptor)
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
        if identity_before != identity_after:
            raise CLIError(f"File changed while it was being identified: {path}")
        return digest.hexdigest()
    finally:
        os.close(descriptor)


def _load_routed_profile(path: str):
    # Kept lazy so the compatibility-only CLI can still explain partial installs.
    try:
        from docarag.routed_profile import load_routed_runtime_profile
    except ImportError as exc:
        raise CLIError(f"Routed profile support is unavailable: {exc}") from exc
    return load_routed_runtime_profile(path)


def _routed_profile_inspection(profile, *, include_configuration: bool) -> dict[str, object]:
    issues = [item.to_dict() for item in profile.configuration_issues()]
    payload: dict[str, object] = {
        "schema_version": "docarag.routed-profile-inspection.v1",
        "valid": True,
        "configuration_ready": not issues,
        "runtime_compatibility_checked": False,
        "live_readiness_checked": False,
        "routed_indexing": {
            "enabled": True,
            "reason": (
                "Pass this profile to `docarag index --routed-profile PATH`; "
                "the exact profile bytes and normalized configuration digest are "
                "bound to the immutable index revision."
            ),
        },
        "revision_binding": dict(profile.revision_binding()),
        "source_path": str(profile.source_path),
        "issues": issues,
    }
    if include_configuration:
        payload["profile"] = profile.to_dict()
    return payload


def command_routed_profile_validate(args: argparse.Namespace) -> int:
    profile = _load_routed_profile(args.profile_path)
    payload = _routed_profile_inspection(profile, include_configuration=False)
    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))
    else:
        if payload["configuration_ready"]:
            print(
                f"Routed profile {profile.profile_id!r} is structurally valid "
                "and statically configuration-ready."
            )
        else:
            print(
                f"Routed profile {profile.profile_id!r} is structurally valid "
                "but not configuration-ready."
            )
            for issue in profile.configuration_issues():
                print(f"  - {issue.runtime_id}.{issue.field}: {issue.message}")
        print("Runtime compatibility and live worker readiness were not checked.")
        print(
            "Use `docarag doctor --routed-profile PATH --strict` for live readiness, "
            "then select it with `docarag index DOCUMENT --routed-profile PATH`."
        )
    return 0 if payload["configuration_ready"] else 1


def command_routed_profile_show(args: argparse.Namespace) -> int:
    profile = _load_routed_profile(args.profile_path)
    payload = _routed_profile_inspection(profile, include_configuration=True)
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


def command_doctor(args: argparse.Namespace) -> int:
    # Imported lazily so the CLI can still explain failures in a partial install.
    try:
        from docarag.diagnostics import (
            doctor_for_routed_profile,
            doctor_for_settings,
            render_report,
        )
    except ImportError as exc:
        raise CLIError(f"Diagnostics module is unavailable: {exc}") from exc

    settings = _load(args)
    if args.routed_profile:
        profile = _load_routed_profile(args.routed_profile)
        report = doctor_for_routed_profile(
            settings,
            profile,
            check_endpoints=not args.offline,
        )
    else:
        report = doctor_for_settings(settings, check_endpoints=not args.offline)
    if args.json:
        print(json.dumps(report.as_dict(), indent=2, ensure_ascii=False))
    else:
        print(render_report(report))
    if args.strict and not report.ok:
        return 1
    return 0


def command_edit(args: argparse.Namespace) -> int:
    _load(args)
    from docarag.edit import main as edit_main

    return edit_main(
        [args.edit_what, "--run", args.run, "--node", str(args.node), "--text", args.text]
    )


def command_stage(args: argparse.Namespace) -> int:
    _load(args)
    from docarag.pipeline import main as pipeline_main

    return pipeline_main([args.stage_name, "--run", args.run])


def command_pipeline(args: argparse.Namespace) -> int:
    _load(args)
    from docarag.pipeline import main as pipeline_main

    action = "overview" if args.pipeline_action == "status" else "watch"
    return pipeline_main([action])


def command_web(args: argparse.Namespace) -> int:
    _load(args)
    from docarag.web_service import main as web_main

    argv = [args.web_action, "--port", str(args.port)]
    if args.run:
        argv += ["--run", args.run]
    return web_main(argv)


def command_index(args: argparse.Namespace) -> int:
    settings = _load(args)
    pdf = Path(args.document).expanduser().resolve()
    if not pdf.is_file():
        raise CLIError(f"Document not found: {pdf}")
    if pdf.suffix.lower() != ".pdf":
        raise CLIError(
            "The current compatibility indexer accepts PDF files only. "
            "Additional ingestors are part of the canonical-ingestion phase."
        )
    if not settings.system_config.is_file():
        raise CLIError(f"System configuration not found: {settings.system_config}")

    routed_profile = None
    routed_binding: dict[str, str] | None = None
    routed_profile_path = getattr(args, "routed_profile", None)
    if routed_profile_path:
        try:
            routed_profile = _load_routed_profile(routed_profile_path)
        except (OSError, RuntimeError, ValueError) as exc:
            raise CLIError(
                f"Routed parser profile is invalid ({type(exc).__name__}): "
                f"{routed_profile_path}"
            ) from exc
        issues = routed_profile.configuration_issues()
        if issues:
            details = "; ".join(
                f"{item.runtime_id}.{item.field}: {item.message}" for item in issues
            )
            raise CLIError(
                "Routed parser profile is not configuration-ready: " + details
            )
        routed_binding = dict(routed_profile.revision_binding())

    script = settings.project_root / "Scripts" / "index_pdf.sh"
    if not script.is_file():
        raise CLIError(f"Compatibility index driver not found: {script}")

    ledger = _require_postgres(settings)
    try:
        from docarag.domain import DocumentRef
        from docarag.persistence import (
            CorpusScope,
            StageEvent,
            StageEventStatus,
        )
    except ImportError as exc:
        raise CLIError("PostgreSQL index tracing contracts are unavailable") from exc

    source_sha256 = _stable_file_sha256(pdf)
    configuration_sha256 = _stable_file_sha256(settings.system_config)
    document = DocumentRef.identified(
        content_sha256=source_sha256,
        mime_type="application/pdf",
        source_uri=pdf.as_uri(),
        display_name=pdf.name,
    )
    try:
        scope = ledger.record_document(
            CorpusScope(settings.tenant_id, settings.corpus_id), document
        )
    except Exception as exc:
        raise CLIError(
            f"Could not admit the document version to PostgreSQL at "
            f"{ledger.endpoint} ({type(exc).__name__})"
        ) from exc
    trace_id = f"trace_{uuid.uuid4().hex}"
    stage_run_id = f"stage_{uuid.uuid4().hex}"
    dispatch_stage = (
        "routed-index-dispatch"
        if routed_profile is not None
        else "compatibility-index-dispatch"
    )
    dispatch_payload: dict[str, object] = {
        "configuration_sha256": configuration_sha256,
        "ingestion_mode": "routed" if routed_profile is not None else "legacy",
    }
    if routed_binding is not None:
        dispatch_payload["routed_profile"] = routed_binding
    try:
        ledger.append_stage_event(
            StageEvent(
                scope=scope,
                trace_id=trace_id,
                stage_run_id=stage_run_id,
                sequence_no=0,
                stage_name=dispatch_stage,
                event_type="driver_started",
                status=StageEventStatus.STARTED,
                payload=dispatch_payload,
            )
        )
    except Exception as exc:
        raise CLIError(
            f"Could not append the PostgreSQL dispatch trace "
            f"({type(exc).__name__})"
        ) from exc

    env = os.environ.copy()
    env["DOCARAG_ENV"] = settings.conda_environment
    env["DOCARAG_CONFIG"] = str(settings.system_config)
    env["DOCARAG_RUNS_DIR"] = str(settings.runs_dir)
    env["DOCARAG_DATABASE_URL"] = settings.database_url
    env["DOCARAG_TENANT_ID"] = scope.tenant_id
    env["DOCARAG_CORPUS_ID"] = scope.corpus_id
    env["DOCARAG_DOCUMENT_ID"] = scope.document_id
    env["DOCARAG_DOCUMENT_VERSION_ID"] = scope.document_version_id
    env["DOCARAG_SOURCE_SHA256"] = source_sha256
    env["DOCARAG_SYSTEM_CONFIG_SHA256"] = configuration_sha256
    env["DOCARAG_TRACE_ID"] = trace_id
    if routed_profile is not None:
        assert routed_binding is not None
        env["DOCARAG_ROUTED_PROFILE"] = str(routed_profile.source_path)
        env["DOCARAG_ROUTED_PROFILE_ID"] = routed_profile.profile_id
        env["DOCARAG_ROUTED_PROFILE_SOURCE_SHA256"] = routed_profile.source_sha256
        env["DOCARAG_ROUTED_PROFILE_CONFIGURATION_SHA256"] = (
            routed_profile.configuration_sha256
        )
    if settings.python_executable:
        env["DOCARAG_PYTHON"] = settings.python_executable
    return_code = _run(
        ["bash", str(script), str(pdf)], cwd=settings.project_root, env=env
    )
    try:
        ledger.append_stage_event(
            StageEvent(
                scope=scope,
                trace_id=trace_id,
                stage_run_id=stage_run_id,
                sequence_no=1,
                stage_name=dispatch_stage,
                event_type=(
                    "driver_dispatched" if return_code == 0 else "driver_rejected"
                ),
                status=(
                    StageEventStatus.SUCCEEDED
                    if return_code == 0
                    else StageEventStatus.FAILED
                ),
                payload={
                    "return_code": return_code,
                    "ingestion_mode": (
                        "routed" if routed_profile is not None else "legacy"
                    ),
                },
            )
        )
    except Exception as exc:
        raise CLIError(
            "The driver returned but its PostgreSQL dispatch trace could not be "
            f"completed ({type(exc).__name__})"
        ) from exc
    return return_code


def command_status(args: argparse.Namespace) -> int:
    settings = _load(args)
    script = settings.project_root / "Scripts" / "progress.sh"
    if not script.is_file():
        raise CLIError(f"Progress command not found: {script}")
    command = ["bash", str(script)]
    if args.watch:
        command.append("--watch")
    if args.run:
        try:
            from docarag.compat import resolve_run
        except ImportError as exc:
            raise CLIError(
                "Compatibility run selection is unavailable; use a complete "
                "BookRAG runtime for status on a named run"
            ) from exc
        command.append(
            str(resolve_run(settings.runs_dir, args.run, purpose="status"))
        )
    env = os.environ.copy()
    env["DOCARAG_RUNS_DIR"] = str(settings.runs_dir)
    return _run(command, cwd=settings.project_root, env=env)


def command_verify(args: argparse.Namespace) -> int:
    """Verify source, runtime, and published artifact provenance for one run."""

    try:
        from docarag.compat import resolve_run
    except ImportError as exc:
        raise CLIError(
            "Compatibility run selection is unavailable; use a complete "
            "BookRAG runtime to select a run for verification"
        ) from exc
    from docarag.provenance import verify_artifact_manifest

    settings = _load(args)
    run = resolve_run(settings.runs_dir, args.run, purpose="status")
    result = verify_artifact_manifest(run, deep=not args.quick)
    payload = {
        "valid": result.valid,
        "mode": "quick" if args.quick else "deep",
        "run_dir": str(run),
        "manifest": str(result.manifest_path),
        "artifact_set_sha256": result.artifact_set_sha256,
        "file_count": result.file_count,
        "total_bytes": result.total_bytes,
        "errors": list(result.errors),
    }
    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    elif result.valid:
        print(
            f"Verified {result.file_count} files ({result.total_bytes} bytes) "
            f"for {run} [{payload['mode']}]"
        )
        print(f"Artifact set SHA-256: {result.artifact_set_sha256}")
    else:
        print(f"Verification failed for {run}:", file=sys.stderr)
        for error in result.errors:
            print(f"  - {error}", file=sys.stderr)
    return 0 if result.valid else 1


def command_ask(args: argparse.Namespace) -> int:
    if not args.question.strip():
        raise CLIError("Question must not be empty")
    settings = _load(args)
    command = [
        *settings.python_command(),
        "-m",
        "docarag.compat",
        "ask",
        "--runs-dir",
        str(settings.runs_dir),
        "--fallback-system-config",
        str(settings.system_config),
    ]
    if args.run:
        command.extend(("--run", args.run))
    if args.allow_legacy_config:
        command.append("--allow-legacy-config")
    if args.json:
        command.append("--json")
    command.append(args.question)
    return _run(command, cwd=settings.project_root)


def _load_parser_worker_profile(path: str):
    # Parser-worker composition stays lazy so existing BookRAG commands do not
    # import any vendor service adapter unless this command family is selected.
    try:
        from docarag.parser_worker import load_parser_worker_profile
    except ImportError as exc:
        raise CLIError(
            f"Parser-worker profile support is unavailable ({type(exc).__name__})"
        ) from exc
    return load_parser_worker_profile(path)


def command_parser_worker_validate(args: argparse.Namespace) -> int:
    """Validate and inspect a profile without resolving secrets or creating state."""

    profile = _load_parser_worker_profile(args.profile_path)
    payload = profile.safe_dict()
    payload["validation"] = {
        "valid": True,
        "secrets_resolved": False,
        "filesystem_roots_created": False,
        "service_contacted": False,
        "qualification": "uncertified",
    }
    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))
    else:
        print(
            f"Parser-worker profile {profile.profile_id!r} is structurally valid "
            f"for {profile.vendor.value}; qualification remains uncertified."
        )
        print("Secrets, filesystem roots, optional renderers, and services were not resolved.")
    return 0


def command_parser_worker_serve(args: argparse.Namespace) -> int:
    """Construct exactly one configured parser adapter and serve it in foreground."""

    profile = _load_parser_worker_profile(args.profile_path)
    try:
        from docarag.parser_worker import (
            ParserWorkerProfileError,
            build_parser_worker_runtime,
        )
    except ImportError as exc:
        raise CLIError(
            f"Parser-worker runtime support is unavailable ({type(exc).__name__})"
        ) from exc
    runtime = None
    try:
        runtime = build_parser_worker_runtime(profile)
        print(
            json.dumps(
                dict(runtime.summary()),
                ensure_ascii=False,
                sort_keys=True,
            ),
            flush=True,
        )
        runtime.serve_forever()
        return 0
    except ParserWorkerProfileError:
        raise
    except Exception as exc:
        # Unexpected dependency/runtime exceptions are reduced to their type;
        # vendor exception text is not trusted to be secret-free.
        raise CLIError(
            f"Parser worker could not start or serve ({type(exc).__name__})"
        ) from exc
    finally:
        if runtime is not None:
            runtime.close()


def _is_loopback_host(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="docarag",
        description="Index, inspect, and query documents with DocARAG/BookRAG.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "--settings",
        help=(
            f"Settings file; selection order is this flag, $DOCARAG_SETTINGS, "
            f"./{DEFAULT_FILENAME}, then the repository-root {DEFAULT_FILENAME}"
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="Create a local DocARAG settings file")
    init_parser.add_argument("--path", default=DEFAULT_FILENAME)
    init_parser.add_argument("--project-root")
    init_parser.add_argument("--force", action="store_true")
    init_parser.set_defaults(handler=command_init)

    settings_parser = subparsers.add_parser("settings", help="Show fully resolved settings")
    settings_parser.set_defaults(handler=command_settings)

    setup_plan_parser = subparsers.add_parser(
        "setup-plan",
        help="Explain exact setup and ownership without changing the machine",
        description=(
            "Build a dependency-light, read-only setup plan. The command does not "
            "install packages, contact endpoints, start services, or change GPU state."
        ),
    )
    setup_plan_parser.add_argument("--json", action="store_true")
    setup_plan_parser.add_argument(
        "--routed-profile",
        metavar="PATH",
        help=(
            "Include static routed-worker requirements; this does not enable "
            "routed indexing or contact workers"
        ),
    )
    setup_plan_parser.set_defaults(handler=command_setup_plan)

    storage_parser = subparsers.add_parser(
        "storage",
        help="Start, stop, initialize, reset, or inspect PostgreSQL and Qdrant storage",
    )
    storage_subparsers = storage_parser.add_subparsers(
        dest="storage_command", required=True
    )
    storage_up = storage_subparsers.add_parser(
        "up", help="Start the loopback-only default PostgreSQL and Qdrant services"
    )
    storage_up.add_argument(
        "--no-init",
        action="store_true",
        help="Start services without creating the PostgreSQL schema",
    )
    storage_up.set_defaults(handler=command_storage_up)
    storage_down = storage_subparsers.add_parser(
        "down", help="Stop default storage containers without deleting named volumes"
    )
    storage_down.set_defaults(handler=command_storage_down)
    storage_init = storage_subparsers.add_parser(
        "init", help="Create the PostgreSQL schema if it is not present"
    )
    storage_init.add_argument("--json", action="store_true")
    storage_init.set_defaults(handler=command_storage_init)
    storage_reset = storage_subparsers.add_parser(
        "reset", help="Drop and recreate the PostgreSQL schema (destroys all data)"
    )
    storage_reset.add_argument("--json", action="store_true")
    storage_reset.add_argument(
        "--yes", action="store_true", help="Confirm dropping the entire schema"
    )
    storage_reset.set_defaults(handler=command_storage_reset)
    storage_status = storage_subparsers.add_parser(
        "status", help="Check PostgreSQL reachability and schema state"
    )
    storage_status.add_argument("--json", action="store_true")
    storage_status.set_defaults(handler=command_storage_status)

    routed_profile_parser = subparsers.add_parser(
        "routed-profile",
        help="Inspect an explicit routed parser profile used by routed indexing",
        description=(
            "Validate or show a strict routed parser profile. Select a ready "
            "profile with `docarag index --routed-profile PATH`."
        ),
    )
    routed_profile_subparsers = routed_profile_parser.add_subparsers(
        dest="routed_profile_command", required=True
    )
    routed_validate_parser = routed_profile_subparsers.add_parser(
        "validate",
        help="Validate structure and exact static pins without contacting workers",
    )
    routed_validate_parser.add_argument("profile_path")
    routed_validate_parser.add_argument("--json", action="store_true")
    routed_validate_parser.set_defaults(handler=command_routed_profile_validate)
    routed_show_parser = routed_profile_subparsers.add_parser(
        "show",
        help="Show normalized secret-free configuration and readiness state",
    )
    routed_show_parser.add_argument("profile_path")
    routed_show_parser.set_defaults(handler=command_routed_profile_show)

    parser_worker_parser = subparsers.add_parser(
        "parser-worker",
        help="Validate or serve one explicit isolated parser-worker profile",
        description=(
            "Operate one strict Docling, MinerU, or PaddleOCR-VL worker profile. "
            "No vendor/model auto-discovery is performed."
        ),
    )
    parser_worker_subparsers = parser_worker_parser.add_subparsers(
        dest="parser_worker_command", required=True
    )
    parser_worker_validate = parser_worker_subparsers.add_parser(
        "validate",
        help="Validate configuration without resolving secrets or contacting services",
    )
    parser_worker_validate.add_argument("profile_path")
    parser_worker_validate.add_argument("--json", action="store_true")
    parser_worker_validate.set_defaults(handler=command_parser_worker_validate)
    parser_worker_serve = parser_worker_subparsers.add_parser(
        "serve",
        help="Serve the configured single parser adapter in the foreground",
    )
    parser_worker_serve.add_argument("profile_path")
    parser_worker_serve.set_defaults(handler=command_parser_worker_serve)

    doctor_parser = subparsers.add_parser("doctor", help="Validate setup and print remediation")
    doctor_parser.add_argument("--json", action="store_true")
    doctor_parser.add_argument(
        "--offline", action="store_true", help="Do not contact configured endpoints"
    )
    doctor_parser.add_argument(
        "--strict",
        action="store_true",
        help="Return nonzero for failed required checks",
    )
    doctor_parser.add_argument(
        "--routed-profile",
        metavar="PATH",
        help=(
            "Also validate static pins and parser-worker readiness for this explicit "
            "profile before selecting it for routed indexing"
        ),
    )
    doctor_parser.set_defaults(handler=command_doctor)

    index_parser = subparsers.add_parser("index", help="Start indexing one PDF")
    index_parser.add_argument("document")
    index_parser.add_argument(
        "--routed-profile",
        metavar="PATH",
        help=(
            "Use the ready Docling/MinerU/PaddleOCR-VL page router and bind the "
            "exact profile to this immutable index revision"
        ),
    )
    index_parser.set_defaults(handler=command_index)

    stage_parser = subparsers.add_parser(
        "stage",
        help="Run or inspect one pipeline stage for an existing run",
        description=(
            "A stage is a store transformation and can be re-run alone -- "
            "after an edit, a model change or a failure -- without running "
            "the pipeline around it. `status` reads stage state from "
            "PostgreSQL."
        ),
    )
    stage_parser.add_argument(
        "stage_name",
        choices=("tree", "graph", "vdb", "persist", "status", "all"),
    )
    stage_parser.add_argument("--run", required=True)
    stage_parser.set_defaults(handler=command_stage)

    pipeline_parser = subparsers.add_parser(
        "pipeline",
        help="Monitor every document's pipeline from the store",
    )
    pipeline_parser.add_argument("pipeline_action", choices=("status", "watch"))
    pipeline_parser.set_defaults(handler=command_pipeline)

    webui_parser = subparsers.add_parser(
        "web", help="Start, stop or inspect the web UI service"
    )
    webui_parser.add_argument(
        "web_action", choices=("start", "stop", "status", "restart")
    )
    webui_parser.add_argument("--port", type=int, default=8765)
    webui_parser.add_argument("--run", help="run directory to serve initially")
    webui_parser.set_defaults(handler=command_web)

    edit_parser = subparsers.add_parser(
        "edit",
        help="Edit stored content; downstream staleness is recorded",
    )
    edit_parser.add_argument("edit_what", choices=("summary",))
    edit_parser.add_argument("--run", required=True)
    edit_parser.add_argument("--node", type=int, required=True)
    edit_parser.add_argument("--text", required=True)
    edit_parser.set_defaults(handler=command_edit)

    status_parser = subparsers.add_parser("status", help="Show indexing status")
    status_parser.add_argument("run", nargs="?")
    status_parser.add_argument("-w", "--watch", action="store_true")
    status_parser.set_defaults(handler=command_status)

    verify_parser = subparsers.add_parser(
        "verify", help="Verify source and index artifact integrity"
    )
    verify_parser.add_argument(
        "--run",
        help="Run name or directory beneath configured runs_dir; defaults to latest",
    )
    verify_parser.add_argument(
        "--quick",
        action="store_true",
        help="Check identity, containment, and sizes without hashing every artifact",
    )
    verify_parser.add_argument("--json", action="store_true")
    verify_parser.set_defaults(handler=command_verify)

    ask_parser = subparsers.add_parser("ask", help="Ask one question against an index")
    ask_parser.add_argument("question")
    ask_parser.add_argument(
        "--run",
        help=(
            "Run name or directory beneath configured runs_dir; defaults to "
            "the latest ask-ready run"
        ),
    )
    ask_parser.add_argument(
        "--allow-legacy-config",
        "--legacy-config-fallback",
        action="store_true",
        help=(
            "For an old run without run_config.yaml, explicitly use the current "
            "configured system_config (not reproducible)"
        ),
    )
    ask_parser.add_argument("--json", action="store_true")
    ask_parser.set_defaults(handler=command_ask)

    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    # Project-local storage settings apply to one command only. Restoring the
    # caller's environment keeps embedded CLI invocations from leaking one
    # project's endpoints into a later invocation in the same process.
    previous_environment = {
        key: os.environ[key] if key in os.environ else None
        for key in _PROJECT_ENV_KEYS
    }
    try:
        return int(args.handler(args) or 0)
    except (CLIError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    finally:
        for key, previous_value in previous_environment.items():
            if previous_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous_value


if __name__ == "__main__":
    raise SystemExit(main())
