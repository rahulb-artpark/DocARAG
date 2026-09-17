"""Record a dependency-light execution-environment fingerprint for one attempt."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import tempfile


def distribution_versions() -> dict[str, str]:
    """Return every discoverable distribution with a canonical package name."""

    versions: dict[str, str] = {}
    for distribution in metadata.distributions():
        try:
            raw_name = distribution.metadata.get("Name")
            version = distribution.version
        except (KeyError, metadata.PackageNotFoundError):
            continue
        if not raw_name or not version:
            continue
        name = re.sub(r"[-_.]+", "-", raw_name).lower()
        versions[name] = version
    return dict(sorted(versions.items()))


def gpu_identity() -> list[dict[str, str]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,uuid,driver_version,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return []

    devices = []
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",", 4)]
        if len(fields) != 5:
            continue
        index, name, uuid, driver_version, memory_mib = fields
        devices.append(
            {
                "index": index,
                "name": name,
                "uuid": uuid,
                "driver_version": driver_version,
                "memory_total_mib": memory_mib,
            }
        )
    return devices


def atomic_private_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.chmod(temporary, 0o400)
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def build_payload(*, revision_id: str, attempt_id: str) -> dict[str, object]:
    identity = {
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "executable": str(Path(sys.executable).resolve()),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "packages": distribution_versions(),
        "gpus": gpu_identity(),
        "environment": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
    }
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return {
        "schema_version": 1,
        "revision_id": revision_id,
        "attempt_id": attempt_id,
        "recorded_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "environment_fingerprint_sha256": hashlib.sha256(canonical).hexdigest(),
        **identity,
        "limitations": [
            "Remote model weight and service image digests are not discoverable "
            "from the current configuration API.",
            "The environment fingerprint is attempt provenance; it is not yet "
            "part of revision identity.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision-id", required=True)
    parser.add_argument("--attempt-id", required=True)
    args = parser.parse_args()

    payload = build_payload(
        revision_id=args.revision_id,
        attempt_id=args.attempt_id,
    )
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    atomic_private_write(args.output.expanduser().resolve(), encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
