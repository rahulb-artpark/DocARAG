from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from document_services.operations.gpu_admission import (
    GpuAdmissionController,
    GpuAdmissionError,
    GpuDevice,
)
from document_services.operations.metrics import MetricsRegistry
from document_services.operations.object_storage import (
    LocalObjectStore,
    ObjectStoreError,
    S3CompatibleObjectStore,
)
from document_services.operations.quotas import (
    InMemoryQuotaStore,
    QuotaDelta,
    QuotaExceededError,
    QuotaLimits,
)
from document_services.operations.scheduler import DurableLeaseWorker
from document_services.operations.security import (
    ApiKeyAuthenticator,
    ApiPrincipal,
    AuthenticationError,
    AuthorizationError,
)
from document_services.operations.service_supervisor import (
    ManagedServiceSupervisor,
    ServiceOwnership,
    ServiceSpec,
    ServiceState,
)
from document_services.persistence import CorpusScope, DocumentScope, WorkLease, WorkState


class _Inventory:
    def __init__(self, free: int = 48_000):
        self.free = free

    def devices(self):
        return (GpuDevice(0, "GPU-fixed", 49_000, self.free),)


class _S3:
    def __init__(self):
        self.values = {}

    def put_object(self, **kwargs):
        key = (kwargs["Bucket"], kwargs["Key"])
        if key in self.values:
            raise RuntimeError("precondition")
        self.values[key] = {
            "data": kwargs["Body"],
            "metadata": kwargs["Metadata"],
            "type": kwargs["ContentType"],
        }

    def head_object(self, **kwargs):
        value = self.values[(kwargs["Bucket"], kwargs["Key"])]
        return {"ContentLength": len(value["data"]), "Metadata": value["metadata"]}

    def get_object(self, **kwargs):
        value = self.values[(kwargs["Bucket"], kwargs["Key"])]
        return {"Body": io.BytesIO(value["data"])}


class _Ledger:
    def __init__(self, lease):
        self.lease = lease
        self.completed = []
        self.failed = []

    def lease_work(self, *args, **kwargs):
        value, self.lease = self.lease, None
        return value

    def heartbeat_work(self, lease, *, extension_seconds):
        return lease

    def complete_work(self, lease, *, result=None):
        self.completed.append((lease, result))

    def fail_work(self, lease, *, error, retry_delay_seconds=0):
        self.failed.append((lease, error))
        return WorkState.PENDING


class OperationalFoundationsTests(unittest.TestCase):
    def test_gpu_admission_is_capacity_aware_exclusive_and_reaps_stale(self):
        identities = {321: 99}
        with tempfile.TemporaryDirectory() as directory:
            controller = GpuAdmissionController(
                Path(directory).resolve(),
                inventory=_Inventory(),
                headroom_mib=6000,
                allocation_limit_mib=42000,
                process_identity=lambda pid: identities.get(pid),
            )
            reserved = controller.reserve(owner="paddle", memory_mib=22000)
            with self.assertRaisesRegex(GpuAdmissionError, "managed model-family"):
                controller.reserve(owner="ollama", memory_mib=12000)
            bound = controller.bind(reserved.lease_id, pid=321)
            self.assertEqual(321, bound.pid)
            identities.clear()
            self.assertEqual((), controller.status())

    def test_gpu_admission_accounts_for_observed_external_memory(self):
        with tempfile.TemporaryDirectory() as directory:
            controller = GpuAdmissionController(
                Path(directory).resolve(), inventory=_Inventory(free=20_000)
            )
            with self.assertRaisesRegex(GpuAdmissionError, "observed 20000"):
                controller.reserve(owner="paddle", memory_mib=16_000)

    def test_supervisor_never_signals_borrowed_or_shared_services(self):
        spec = ServiceSpec(
            name="ollama-shared",
            ownership=ServiceOwnership.SHARED,
            runtime_identity="ollama@0.13.4",
            health_url="http://127.0.0.1:11434/api/tags",
        )
        with tempfile.TemporaryDirectory() as directory:
            supervisor = ManagedServiceSupervisor(
                Path(directory).resolve(), health_probe=lambda *args: True
            )
            with patch("os.killpg") as kill:
                status = supervisor.stop(spec)
        kill.assert_not_called()
        self.assertEqual(ServiceState.EXTERNAL_READY, status.state)

    def test_owned_supervisor_binds_exact_pid_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / "service"
            executable.write_text("#!/bin/sh\n", encoding="utf-8")
            executable.chmod(0o700)
            spec = ServiceSpec(
                name="docling-worker",
                ownership=ServiceOwnership.OWNED,
                runtime_identity="docarag-worker@0.1.0",
                health_url="http://127.0.0.1:5001/v1/health",
                argv=(str(executable),),
            )
            identities = {777: 45}
            calls = []
            supervisor = ManagedServiceSupervisor(
                Path(directory).resolve() / "state",
                health_probe=lambda *args: True,
                process_identity=lambda pid: identities.get(pid),
                popen=lambda *args, **kwargs: calls.append((args, kwargs)) or SimpleNamespace(pid=777),
            )
            first = supervisor.start(spec)
            second = supervisor.start(spec)
            self.assertEqual(ServiceState.READY, first.state)
            self.assertEqual(777, second.pid)
            self.assertEqual(1, len(calls))
            with patch.object(supervisor, "_terminate_exact") as terminate:
                stopped = supervisor.stop(spec)
            terminate.assert_called_once_with(777, 45, spec.shutdown_timeout_seconds)
            self.assertEqual(ServiceState.STOPPED, stopped.state)

    def test_local_and_s3_object_stores_verify_content_receipts(self):
        data = b"canonical artifact"
        with tempfile.TemporaryDirectory() as directory:
            local = LocalObjectStore(Path(directory).resolve())
            local_receipt = local.put_bytes(data, media_type="application/json")
            self.assertEqual(data, local.get_bytes(local_receipt, max_bytes=100))
        client = _S3()
        remote = S3CompatibleObjectStore(client, bucket="docarag-test")
        receipt = remote.put_bytes(data, media_type="application/json")
        self.assertEqual(data, remote.get_bytes(receipt, max_bytes=100))
        client.values[("docarag-test", remote._key(receipt.sha256))]["data"] = b"tampered"
        with self.assertRaisesRegex(ObjectStoreError, "immutable receipt"):
            remote.get_bytes(receipt, max_bytes=100)

    def test_auth_quota_and_metrics_fail_closed(self):
        principal = ApiPrincipal("user-1", "tenant-a", frozenset({"query"}))
        key = "a-secure-development-key-12345"
        auth = ApiKeyAuthenticator.from_plaintext_mapping({key: principal})
        self.assertEqual(principal, auth.authenticate(f"Bearer {key}"))
        with self.assertRaises(AuthenticationError):
            auth.authenticate("Bearer wrong-but-long-enough-key-000")
        with self.assertRaises(AuthorizationError):
            auth.authorize(principal, tenant_id="tenant-b", required_scope="query")

        quotas = InMemoryQuotaStore()
        limits = QuotaLimits(2, 100, 200, 1, 3)
        quotas.reserve("tenant-a", QuotaDelta(documents=2, source_bytes=10), limits)
        with self.assertRaisesRegex(QuotaExceededError, "documents"):
            quotas.reserve("tenant-a", QuotaDelta(documents=1), limits)

        metrics = MetricsRegistry(allowed_label_values={"stage": frozenset({"parse"})})
        metrics.increment("docarag_stage_total", labels={"stage": "parse"})
        self.assertIn('docarag_stage_total{stage="parse"} 1', metrics.render_prometheus())
        with self.assertRaises(ValueError):
            metrics.increment("docarag_stage_total", labels={"tenant_id": "tenant-a"})

    def test_durable_worker_uses_fenced_lease_and_commits_result(self):
        lease = WorkLease(
            scope=DocumentScope("tenant-a", "corpus-a", "doc-a", "version-a"),
            work_item_id="work-a",
            kind="parse",
            payload={"revision": "r1"},
            worker_id="worker-a",
            fencing_token=1,
            attempt_count=1,
            max_attempts=3,
            lease_expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
        )
        ledger = _Ledger(lease)
        worker = DurableLeaseWorker(
            ledger,
            CorpusScope("tenant-a", "corpus-a"),
            worker_id="worker-a",
            handlers={"parse": lambda claimed: {"receipt": claimed.work_item_id}},
            lease_seconds=3,
        )
        outcome = worker.run_once()
        self.assertEqual(WorkState.SUCCEEDED, outcome.state)
        self.assertEqual({"receipt": "work-a"}, ledger.completed[0][1])


if __name__ == "__main__":
    unittest.main()
