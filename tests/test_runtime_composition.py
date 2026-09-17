from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import json
import subprocess
import sys
import unittest

from document_services.application import ExtractDocumentUseCase
from document_services.domain import (
    AcceleratorKind,
    BlockType,
    ParseRequest,
    ParseResult,
    ParserCapabilityManifest,
    ParserModality,
)
from document_services.parsing.routing_presets import (
    DOCLING_ADAPTER,
    MINERU_ADAPTER,
    PADDLEOCR_VL_ADAPTER,
)
from document_services.parsing.routed_ingestion import RoutedCanonicalIngestionProvider
from document_services.runtime.parser_composition import (
    BalancedRoutedRuntimeConfig,
    ParserWorkerEndpointConfig,
    RuntimeCompositionClosedError,
    RuntimeCompositionConfigurationError,
    RuntimeCompositionIdentityError,
    build_balanced_routed_runtime,
)
from document_services.worker_transport import WorkerHealth, WorkerNotReadyError, WorkerState


def _manifest(adapter_name: str, adapter_version: str) -> ParserCapabilityManifest:
    return ParserCapabilityManifest(
        adapter_name=adapter_name,
        adapter_version=adapter_version,
        parser_name=adapter_name,
        parser_version="fixture-parser-1",
        model_version="fixture-model-1",
        supported_mime_types=("application/pdf",),
        modalities=tuple(ParserModality),
        block_types=tuple(BlockType),
        accelerator=AcceleratorKind.CPU,
        supports_page_selection=True,
        supports_batch=True,
        deterministic=True,
    )


def _endpoint(
    adapter_name: str,
    port: int,
    *,
    runtime_id: str | None = None,
    bearer_token_env: str | None = None,
) -> tuple[ParserWorkerEndpointConfig, ParserCapabilityManifest]:
    version = "1.2.3"
    manifest = _manifest(adapter_name, version)
    endpoint = ParserWorkerEndpointConfig(
        runtime_id=runtime_id or f"{adapter_name}-runtime",
        adapter_name=adapter_name,
        base_url=f"http://127.0.0.1:{port}",
        expected_adapter_version=version,
        expected_manifest_sha256=manifest.manifest_sha256,
        bearer_token_env=bearer_token_env,
        estimated_latency_ms=port,
        estimated_cost_microusd=0,
    )
    return endpoint, manifest


def _configuration(*, token_env: str | None = "DOCARAG_DOCLING_TOKEN"):
    values = (
        _endpoint(
            DOCLING_ADAPTER,
            8101,
            runtime_id="docling-worker-a",
            bearer_token_env=token_env,
        ),
        _endpoint(
            MINERU_ADAPTER,
            8102,
            runtime_id="mineru-worker-a",
        ),
        _endpoint(
            PADDLEOCR_VL_ADAPTER,
            8103,
            runtime_id="paddle-worker-a",
        ),
    )
    endpoints = tuple(item[0] for item in values)
    manifests = {item[1].adapter_name: item[1] for item in values}
    return BalancedRoutedRuntimeConfig(endpoints=endpoints), manifests


class _FakeBackend:
    backend_name = "fake-pdf-metadata"
    backend_version = "1"

    def open(self, source_path):
        raise AssertionError("factory tests must not probe a PDF")


class _FakeClient:
    def __init__(
        self,
        config,
        manifest,
        *,
        worker_instance_id: str,
        state: WorkerState = WorkerState.READY,
        events: list[tuple[str, str]],
        close_order: list[str],
    ) -> None:
        self.config = config
        self.value = manifest
        self.worker_instance_id = worker_instance_id
        self.state = state
        self.events = events
        self.close_order = close_order
        self.closed = False

    def refresh_manifest(self) -> ParserCapabilityManifest:
        self.events.append((self.value.adapter_name, "capabilities"))
        return self.value

    def manifest(self) -> ParserCapabilityManifest:
        self.events.append((self.value.adapter_name, "manifest-cache"))
        return self.value

    def health(self, *, require_ready: bool = False) -> WorkerHealth:
        self.events.append((self.value.adapter_name, "health"))
        return WorkerHealth(
            worker_instance_id=self.worker_instance_id,
            state=self.state,
            accepting_requests=self.state is WorkerState.READY,
            in_flight=0,
            max_in_flight=1,
            adapter_name=self.value.adapter_name,
            adapter_version=self.value.adapter_version,
            manifest_sha256=self.value.manifest_sha256,
        )

    def parse(self, request: ParseRequest) -> ParseResult:
        raise AssertionError("factory tests must not parse")

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            self.close_order.append(self.value.adapter_name)


def _client_factory(
    manifests,
    *,
    states=None,
    instance_ids=None,
    override_manifests=None,
):
    clients = []
    events = []
    close_order = []
    states = states or {}
    instance_ids = instance_ids or {}
    override_manifests = override_manifests or {}

    def factory(config):
        adapter_name = config.expected_adapter_name
        manifest = override_manifests.get(adapter_name, manifests[adapter_name])
        client = _FakeClient(
            config,
            manifest,
            worker_instance_id=instance_ids.get(
                adapter_name, f"instance-{adapter_name}"
            ),
            state=states.get(adapter_name, WorkerState.READY),
            events=events,
            close_order=close_order,
        )
        clients.append(client)
        return client

    return factory, clients, events, close_order


class BalancedRuntimeConfigurationTests(unittest.TestCase):
    def test_configuration_is_immutable_exact_and_secret_reference_only(self):
        config, _ = _configuration()
        with self.assertRaises(FrozenInstanceError):
            config.endpoints = ()  # type: ignore[misc]
        serialized = json.dumps(config.to_dict(), sort_keys=True)
        self.assertIn("DOCARAG_DOCLING_TOKEN", serialized)
        self.assertNotIn("resolved-secret", serialized)
        self.assertEqual(
            {DOCLING_ADAPTER, MINERU_ADAPTER, PADDLEOCR_VL_ADAPTER},
            {item.adapter_name for item in config.endpoints},
        )

    def test_requires_exact_balanced_identities_and_unique_runtime_and_endpoints(self):
        config, _ = _configuration(token_env=None)
        with self.assertRaisesRegex(
            RuntimeCompositionConfigurationError, "exactly one endpoint"
        ):
            BalancedRoutedRuntimeConfig(endpoints=config.endpoints[:2])

        duplicate_runtime = replace(
            config.endpoints[1], runtime_id=config.endpoints[0].runtime_id
        )
        with self.assertRaisesRegex(
            RuntimeCompositionConfigurationError, "runtime_id"
        ):
            BalancedRoutedRuntimeConfig(
                endpoints=(config.endpoints[0], duplicate_runtime, config.endpoints[2])
            )

        duplicate_endpoint = replace(
            config.endpoints[1], base_url=config.endpoints[0].base_url
        )
        with self.assertRaisesRegex(
            RuntimeCompositionConfigurationError, "URLs must be unique"
        ):
            BalancedRoutedRuntimeConfig(
                endpoints=(config.endpoints[0], duplicate_endpoint, config.endpoints[2])
            )

    def test_endpoint_requires_version_digest_and_valid_token_reference(self):
        config, _ = _configuration(token_env=None)
        endpoint = config.endpoints[0]
        with self.assertRaises(RuntimeCompositionConfigurationError):
            replace(endpoint, expected_manifest_sha256="not-a-digest")
        with self.assertRaises(RuntimeCompositionConfigurationError):
            replace(endpoint, expected_adapter_version="")
        with self.assertRaisesRegex(
            RuntimeCompositionConfigurationError, "environment-variable"
        ):
            replace(endpoint, bearer_token_env="INVALID-NAME!")


class BalancedRuntimeFactoryTests(unittest.TestCase):
    def test_factory_checks_capability_then_health_and_context_closes_in_reverse(self):
        config, manifests = _configuration()
        factory, clients, events, close_order = _client_factory(manifests)
        before_modules = set(sys.modules)

        runtime = build_balanced_routed_runtime(
            config,
            environment={"DOCARAG_DOCLING_TOKEN": "resolved-secret"},
            probe_backend_factory=_FakeBackend,
            client_factory=factory,
        )

        self.assertIsInstance(runtime.provider, RoutedCanonicalIngestionProvider)
        self.assertIsInstance(runtime.document_extractor, ExtractDocumentUseCase)
        self.assertEqual(
            config.ingestion_policy.max_canonical_artifact_bytes,
            runtime.document_extractor.policy.max_canonical_artifact_bytes,
        )
        self.assertEqual(3, len(runtime.worker_identities))
        self.assertEqual(
            set(manifests),
            {item.adapter_name for item in runtime.facade.registry.manifests()},
        )
        for adapter_name in sorted(manifests):
            capability_index = events.index((adapter_name, "capabilities"))
            health_index = events.index((adapter_name, "health"))
            self.assertLess(capability_index, health_index)
        token_client = next(
            client
            for client in clients
            if client.config.expected_adapter_name == DOCLING_ADAPTER
        )
        self.assertEqual("resolved-secret", token_client.config.bearer_token)
        self.assertNotIn("resolved-secret", repr(token_client.config))
        self.assertNotIn("pymupdf", set(sys.modules) - before_modules)
        self.assertNotIn("fitz", set(sys.modules) - before_modules)

        with runtime:
            self.assertFalse(runtime.closed)
        self.assertTrue(runtime.closed)
        self.assertEqual(
            [
                PADDLEOCR_VL_ADAPTER,
                MINERU_ADAPTER,
                DOCLING_ADAPTER,
            ],
            close_order,
        )
        with self.assertRaises(RuntimeCompositionClosedError):
            _ = runtime.provider
        with self.assertRaises(RuntimeCompositionClosedError):
            _ = runtime.document_extractor
        runtime.close()

    def test_missing_token_fails_before_allocating_any_client(self):
        config, manifests = _configuration()
        factory, clients, _, _ = _client_factory(manifests)

        with self.assertRaisesRegex(
            RuntimeCompositionConfigurationError,
            "DOCARAG_DOCLING_TOKEN",
        ):
            build_balanced_routed_runtime(
                config,
                environment={},
                probe_backend_factory=_FakeBackend,
                client_factory=factory,
            )

        self.assertEqual([], clients)

    def test_capability_identity_mismatch_fails_fast_and_closes_created_client(self):
        config, manifests = _configuration(token_env=None)
        wrong = _manifest(DOCLING_ADAPTER, "9.9.9")
        factory, clients, events, close_order = _client_factory(
            manifests,
            override_manifests={DOCLING_ADAPTER: wrong},
        )

        with self.assertRaisesRegex(
            RuntimeCompositionIdentityError, "configured pin"
        ):
            build_balanced_routed_runtime(
                config,
                environment={},
                probe_backend_factory=_FakeBackend,
                client_factory=factory,
            )

        self.assertEqual(1, len(clients))
        self.assertEqual([(DOCLING_ADAPTER, "capabilities")], events)
        self.assertEqual([DOCLING_ADAPTER], close_order)

    def test_nonready_health_fails_fast_and_closes_client(self):
        config, manifests = _configuration(token_env=None)
        factory, clients, _, close_order = _client_factory(
            manifests,
            states={DOCLING_ADAPTER: WorkerState.STARTING},
        )

        with self.assertRaises(WorkerNotReadyError):
            build_balanced_routed_runtime(
                config,
                environment={},
                probe_backend_factory=_FakeBackend,
                client_factory=factory,
            )

        self.assertEqual(1, len(clients))
        self.assertEqual([DOCLING_ADAPTER], close_order)

    def test_duplicate_live_worker_identity_closes_every_client(self):
        config, manifests = _configuration(token_env=None)
        duplicate_ids = {name: "one-shared-worker" for name in manifests}
        factory, clients, _, close_order = _client_factory(
            manifests,
            instance_ids=duplicate_ids,
        )

        with self.assertRaisesRegex(
            RuntimeCompositionIdentityError, "duplicate worker_instance_id"
        ):
            build_balanced_routed_runtime(
                config,
                environment={},
                probe_backend_factory=_FakeBackend,
                client_factory=factory,
            )

        self.assertEqual(3, len(clients))
        self.assertEqual(
            [PADDLEOCR_VL_ADAPTER, MINERU_ADAPTER, DOCLING_ADAPTER],
            close_order,
        )

    def test_module_import_does_not_load_vendor_or_bookrag_runtime(self):
        script = """
import sys
import document_services.runtime.parser_composition
for name in ('pymupdf', 'fitz', 'torch', 'paddle', 'transformers', "retrieval_engine"):
    assert name not in sys.modules, name
"""
        completed = subprocess.run(
            [sys.executable, "-c", script],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(0, completed.returncode, completed.stderr)

    def test_backend_factory_failure_closes_all_ready_clients(self):
        config, manifests = _configuration(token_env=None)
        factory, clients, _, close_order = _client_factory(manifests)

        def fail_backend():
            raise RuntimeError("backend fixture failed")

        with self.assertRaisesRegex(RuntimeError, "backend fixture failed"):
            build_balanced_routed_runtime(
                config,
                environment={},
                probe_backend_factory=fail_backend,
                client_factory=factory,
            )

        self.assertEqual(3, len(clients))
        self.assertEqual(
            [PADDLEOCR_VL_ADAPTER, MINERU_ADAPTER, DOCLING_ADAPTER],
            close_order,
        )


if __name__ == "__main__":
    unittest.main()
