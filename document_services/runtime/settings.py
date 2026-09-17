"""Dependency-light settings for the supported DocARAG command line.

This module intentionally uses only the Python standard library. Commands such as
``docarag doctor`` must remain usable when ML dependencies are missing or broken.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
import re
import sys
import tomllib
from typing import Any, Mapping
from urllib.parse import urlsplit


SETTINGS_VERSION = 1
DEFAULT_FILENAME = "docarag.toml"
DEFAULT_DATABASE_URL_ENV = "DOCARAG_DATABASE_URL"
DEFAULT_DATABASE_URL = (
    "postgresql://docarag:docarag-local@127.0.0.1:5432/docarag"
)
SUPPORTED_PROFILES = frozenset({"compatibility"})
_ROOT_KEYS = frozenset({"schema_version", "profile", "runtime", "storage"})
_RUNTIME_KEYS = frozenset(
    {
        "project_root",
        "system_config",
        "runs_dir",
        "conda_environment",
        "python_executable",
        "database_url_env",
    }
)
_STORAGE_KEYS = frozenset({"tenant_id", "corpus_id"})

_ENVIRONMENT_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def repository_root() -> Path:
    """Locate a DocARAG checkout without assuming an editable installation."""

    configured = os.environ.get("DOCARAG_PROJECT_ROOT")
    candidates = [
        Path(configured).expanduser() if configured else None,
        Path.cwd(),
        Path(__file__).resolve().parents[2],
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        resolved = candidate.resolve()
        if (resolved / "main.py").is_file() and (resolved / "Scripts").is_dir():
            return resolved
    return Path.cwd().resolve()


@dataclass(frozen=True)
class RuntimeSettings:
    """Resolved compatibility-runtime settings.

    Relative paths are resolved against ``project_root``. This keeps commands
    independent from the caller's current working directory.
    """

    project_root: Path
    profile: str = "compatibility"
    system_config: Path = Path("config/ollama-gbc.yaml")
    runs_dir: Path = Path("runs")
    conda_environment: str = "gbc-rag"
    python_executable: str | None = None
    database_url_env: str = DEFAULT_DATABASE_URL_ENV
    tenant_id: str = "default"
    corpus_id: str = "default"
    settings_path: Path | None = None

    def resolved(self) -> "RuntimeSettings":
        if (
            not isinstance(self.database_url_env, str)
            or _ENVIRONMENT_NAME.fullmatch(self.database_url_env) is None
        ):
            raise ValueError(
                "database_url_env must be an environment-variable name"
            )
        _validate_storage_id(self.tenant_id, "tenant_id")
        _validate_storage_id(self.corpus_id, "corpus_id")
        root = self.project_root.expanduser().resolve()
        config = self.system_config.expanduser()
        runs = self.runs_dir.expanduser()
        return replace(
            self,
            project_root=root,
            system_config=config if config.is_absolute() else root / config,
            runs_dir=runs if runs.is_absolute() else root / runs,
        )

    def python_command(self) -> list[str]:
        """Return the interpreter command for compatibility workloads."""

        if self.python_executable:
            return [self.python_executable]
        if os.environ.get("CONDA_DEFAULT_ENV") == self.conda_environment:
            return [sys.executable]
        return [
            "conda",
            "run",
            "--no-capture-output",
            "-n",
            self.conda_environment,
            "python",
        ]

    def resolve_database_url(
        self,
        environment: Mapping[str, str] | None = None,
    ) -> str:
        """Resolve the PostgreSQL URL without copying it into settings output.

        The TOML file stores only an environment-variable *name*.  This keeps
        credentials out of configuration snapshots, run manifests, CLI
        ``settings`` output, and dataclass representations.
        """

        env = os.environ if environment is None else environment
        value = (
            env[self.database_url_env]
            if self.database_url_env in env
            else DEFAULT_DATABASE_URL
        )
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"{self.database_url_env} must contain a non-empty PostgreSQL URL"
            )
        parsed = urlsplit(value)
        if parsed.scheme not in {"postgres", "postgresql"}:
            raise ValueError(
                f"{self.database_url_env} must use postgres:// or postgresql://"
            )
        try:
            host = parsed.hostname
            port = parsed.port
        except ValueError as exc:
            raise ValueError(
                f"{self.database_url_env} has an invalid host or port"
            ) from exc
        if host is None or not parsed.path or parsed.path == "/":
            raise ValueError(
                f"{self.database_url_env} must include a host and database name"
            )
        if port is not None and not 1 <= port <= 65535:
            raise ValueError(f"{self.database_url_env} has an invalid port")
        if parsed.fragment:
            raise ValueError(f"{self.database_url_env} must not contain a fragment")
        return value

    @property
    def database_url(self) -> str:
        """The effective secret-bearing URL; never included by :meth:`as_dict`."""

        return self.resolve_database_url()

    def redacted_database_url(
        self,
        environment: Mapping[str, str] | None = None,
    ) -> str:
        parsed = urlsplit(self.resolve_database_url(environment))
        hostname = parsed.hostname or ""
        if ":" in hostname and not hostname.startswith("["):
            hostname = f"[{hostname}]"
        netloc = hostname
        if parsed.port is not None:
            netloc = f"{netloc}:{parsed.port}"
        return f"postgresql://{netloc}{parsed.path}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SETTINGS_VERSION,
            "profile": self.profile,
            "project_root": str(self.project_root),
            "system_config": str(self.system_config),
            "runs_dir": str(self.runs_dir),
            "conda_environment": self.conda_environment,
            "python_executable": self.python_executable,
            "database_url_env": self.database_url_env,
            "database_url": self.redacted_database_url(),
            "tenant_id": self.tenant_id,
            "corpus_id": self.corpus_id,
            "settings_path": str(self.settings_path) if self.settings_path else None,
        }


def discover_settings_path(
    explicit: str | os.PathLike[str] | None = None,
    *,
    cwd: Path | None = None,
) -> Path | None:
    """Return the selected settings file.

    Precedence is explicit ``--settings``, ``DOCARAG_SETTINGS``, the current
    directory, then the repository root.  This function selects a file; values
    inside a selected file are not silently overridden by other environment
    variables.
    """

    if explicit:
        return Path(explicit).expanduser().resolve()

    env_path = os.environ.get("DOCARAG_SETTINGS")
    if env_path:
        return Path(env_path).expanduser().resolve()

    here = (cwd or Path.cwd()).resolve()
    candidate = here / DEFAULT_FILENAME
    if candidate.is_file():
        return candidate

    repo_candidate = repository_root() / DEFAULT_FILENAME
    return repo_candidate if repo_candidate.is_file() else None


def _table(data: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = data.get(name, {})
    if not isinstance(value, Mapping):
        raise ValueError(f"[{name}] must be a TOML table")
    return value


def _reject_unknown_keys(
    data: Mapping[str, Any], allowed: frozenset[str], *, scope: str
) -> None:
    unknown = sorted(str(key) for key in data if key not in allowed)
    if unknown:
        rendered = ", ".join(unknown)
        raise ValueError(f"Unknown DocARAG setting(s) in {scope}: {rendered}")


def _nonempty_string(
    data: Mapping[str, Any], key: str, default: str, *, optional: bool = False
) -> str | None:
    value = data.get(key, default)
    if optional and value in (None, ""):
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"[runtime].{key} must be a non-empty string")
    return value


def _validate_storage_id(value: object, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > 512
        or "\x00" in value
    ):
        raise ValueError(
            f"[storage].{field_name} must contain 1 to 512 characters"
        )
    return value


def load_settings(
    path: str | os.PathLike[str] | None = None,
    *,
    cwd: Path | None = None,
) -> RuntimeSettings:
    settings_path = discover_settings_path(path, cwd=cwd)
    if path is not None and (settings_path is None or not settings_path.is_file()):
        raise FileNotFoundError(f"DocARAG settings file not found: {path}")

    if settings_path is None:
        return RuntimeSettings(project_root=repository_root()).resolved()

    with settings_path.open("rb") as handle:
        data = tomllib.load(handle)

    _reject_unknown_keys(data, _ROOT_KEYS, scope="the root table")

    version = data.get("schema_version", SETTINGS_VERSION)
    if version != SETTINGS_VERSION:
        raise ValueError(
            f"Unsupported DocARAG settings schema {version!r}; "
            f"expected {SETTINGS_VERSION}"
        )

    profile = data.get("profile", "compatibility")
    if not isinstance(profile, str) or profile not in SUPPORTED_PROFILES:
        supported = ", ".join(sorted(SUPPORTED_PROFILES))
        raise ValueError(
            f"Unsupported DocARAG profile {profile!r}; supported: {supported}"
        )

    runtime = _table(data, "runtime")
    _reject_unknown_keys(runtime, _RUNTIME_KEYS, scope="[runtime]")
    root_value = _nonempty_string(
        runtime, "project_root", str(settings_path.parent)
    )
    root = Path(root_value).expanduser()
    if not root.is_absolute():
        root = settings_path.parent / root

    system_config = _nonempty_string(
        runtime, "system_config", "config/ollama-gbc.yaml"
    )
    runs_dir = _nonempty_string(runtime, "runs_dir", "runs")
    conda_environment = _nonempty_string(
        runtime, "conda_environment", "gbc-rag"
    )
    python_executable = _nonempty_string(
        runtime, "python_executable", "", optional=True
    )
    database_url_env = _nonempty_string(
        runtime, "database_url_env", DEFAULT_DATABASE_URL_ENV
    )
    if _ENVIRONMENT_NAME.fullmatch(database_url_env) is None:
        raise ValueError(
            "[runtime].database_url_env must be an environment-variable name"
        )

    storage = _table(data, "storage")
    _reject_unknown_keys(storage, _STORAGE_KEYS, scope="[storage]")
    tenant_id = _validate_storage_id(
        storage.get("tenant_id", "default"), "tenant_id"
    )
    corpus_id = _validate_storage_id(
        storage.get("corpus_id", "default"), "corpus_id"
    )

    result = RuntimeSettings(
        project_root=root,
        profile=profile,
        system_config=Path(system_config),
        runs_dir=Path(runs_dir),
        conda_environment=conda_environment,
        python_executable=python_executable,
        database_url_env=database_url_env,
        tenant_id=tenant_id,
        corpus_id=corpus_id,
        settings_path=settings_path,
    )
    return result.resolved()


def default_toml(*, project_root: Path | None = None) -> str:
    root = (project_root or repository_root()).expanduser().resolve()
    return (
        f"schema_version = {SETTINGS_VERSION}\n"
        'profile = "compatibility"\n\n'
        "[runtime]\n"
        f"project_root = {_toml_string(str(root))}\n"
        'system_config = "config/ollama-gbc.yaml"\n'
        'runs_dir = "runs"\n'
        'conda_environment = "gbc-rag"\n'
        "# python_executable = \"/absolute/path/to/python\"\n"
        f'database_url_env = "{DEFAULT_DATABASE_URL_ENV}"\n'
        "\n[storage]\n"
        'tenant_id = "default"\n'
        'corpus_id = "default"\n'
    )


def _toml_string(value: str) -> str:
    """Encode a TOML basic string without adding a TOML dependency.

    JSON and TOML basic strings share the escaping required here.  In
    particular, this protects generated settings from quotes, backslashes and
    control characters in a project path.
    """

    return json.dumps(value, ensure_ascii=False)


def initialize_settings(
    destination: Path,
    *,
    project_root: Path | None = None,
    force: bool = False,
) -> Path:
    target = destination.expanduser().resolve()
    if target.exists() and not force:
        raise FileExistsError(
            f"Settings already exist: {target}. Use --force to replace them."
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(default_toml(project_root=project_root), encoding="utf-8")
    return target
