from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import tempfile
import textwrap
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from docarag.compat import inspect_run_readiness
from docarag.provenance import verify_artifact_manifest
from tests.test_cli_settings import qualified_routed_profile_text


REPO = Path(__file__).resolve().parents[1]
INDEX_DRIVER = REPO / "Scripts" / "index_pdf.sh"
PROGRESS = REPO / "Scripts" / "progress.sh"


FAKE_RUNTIME = r'''#!/usr/bin/env python3
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import time
import pickle
from types import SimpleNamespace

args = sys.argv[1:]
if args and args[0].endswith("docarag/runtime_probe.py"):
    output = Path(args[args.index("--output") + 1])
    revision_id = args[args.index("--revision-id") + 1]
    attempt_id = args[args.index("--attempt-id") + 1]
    output.parent.mkdir(parents=True, exist_ok=True)
    (output.parent / ".runtime-started").touch()
    runtime_sleep = float(os.environ.get("FAKE_RUNTIME_SLEEP", "0"))
    if runtime_sleep:
        time.sleep(runtime_sleep)
    identity = {
        "python": {"version": "fake"},
        "platform": {},
        "packages": {},
        "gpus": [],
        "environment": {"CUDA_VISIBLE_DEVICES": None},
    }
    fingerprint = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    output.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "revision_id": revision_id,
                "attempt_id": attempt_id,
                "environment_fingerprint_sha256": fingerprint,
                **identity,
            }
        ),
        encoding="utf-8",
    )
    output.chmod(0o400)
    raise SystemExit(0)
if args[:2] == ["-m", "docarag.tree_integrity"]:
    from docarag.tree_integrity import _main

    raise SystemExit(_main(args[2:]))
if args[:2] == ["-m", "docarag.provenance"]:
    from docarag.provenance import _main

    publish_run = Path(args[args.index("--run") + 1])
    (publish_run / ".fake-publish-started").touch()
    publish_sleep = float(os.environ.get("FAKE_PUBLISH_SLEEP", "0"))
    if publish_sleep:
        time.sleep(publish_sleep)
    raise SystemExit(_main(args[2:]))
if args[:2] == ["-m", "docarag.qdrant_readiness"]:
    attestation = Path(args[args.index("--attestation") + 1])
    revision_id = args[args.index("--revision-id") + 1]
    expected_kind = (
        args[args.index("--expected-index-kind") + 1]
        if "--expected-index-kind" in args
        else None
    )
    payload = json.loads(attestation.read_text(encoding="utf-8"))
    scope = payload.get("scope_filter") or {}
    if scope.get("index_revision_id") != revision_id:
        raise SystemExit("attestation revision mismatch")
    if expected_kind is not None and scope.get("index_kind") != expected_kind:
        raise SystemExit("attestation index-kind mismatch")
    print(payload["point_count"])
    raise SystemExit(0)
if args[:2] == ["-m", "docarag.artifact_roles"]:
    run_dir = Path(args[args.index("--run") + 1])
    output = Path(args[args.index("--output") + 1])
    is_bm25 = (
        "index_type: bm25"
        in Path(args[args.index("--config") + 1]).read_text(encoding="utf-8")
    )
    roles = {
        "index_type": "bm25" if is_bm25 else "gbc",
        "graph": str(run_dir / "graph_data.json"),
        "entity_attestation": str(
            run_dir
            / "entity_attestation_fake"
            / "docarag-qdrant-index.ready.json"
        ),
    }
    if is_bm25:
        roles["lexical_store"] = str(run_dir / "Tree_vdb" / "bm25_index.pkl")
    else:
        roles["retrieval_attestation"] = str(
            run_dir / "Tree_vdb" / "docarag-qdrant-index.ready.json"
        )
        roles["retrieval_scope"] = {"index_kind": "tree"}
    output.write_text(json.dumps(roles), encoding="utf-8")
    output.chmod(0o400)
    raise SystemExit(0)
config_path = Path(args[args.index("-c") + 1])
dataset_config = Path(args[args.index("-d") + 1])
stage = args[args.index("--stage") + 1]

values = {}
for line in dataset_config.read_text(encoding="utf-8").splitlines():
    key, value = line.split(":", 1)
    values[key] = json.loads(value.strip())
record = json.loads(Path(values["dataset_path"]).read_text(encoding="utf-8"))[0]
run_dir = Path(values["working_dir"]) / record["doc_uuid"]
run_dir.mkdir(parents=True, exist_ok=True)
(run_dir / f".fake-stage-{stage}-started").touch()
if os.environ.get("FAKE_IGNORE_TERM") == "1":
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
stage_sleep = float(os.environ.get("FAKE_STAGE_SLEEP", "0"))
if stage_sleep:
    time.sleep(stage_sleep)

run_config = run_dir / "run_config.yaml"
config_text = config_path.read_text(encoding="utf-8")
config_text = config_text.replace(
    "tenant_id: default", f'tenant_id: {json.dumps(record["tenant_id"])}'
).replace(
    "corpus_id: default", f'corpus_id: {json.dumps(record["corpus_id"])}'
)
if "qdrant_url:" not in config_text:
    config_text += (
        "  tenant_id: default\n"
        "  corpus_id: default\n"
        "  qdrant_url: http://qdrant.test:6333\n"
        "  api_key_env: QDRANT_API_KEY\n"
        "  timeout_seconds: 1.0\n"
    )
payload = (
    config_text
    + "source:\n"
    + f'  display_name: {json.dumps(record["source_name"])}\n'
    + f'  original_uri: {json.dumps(record["source_uri"])}\n'
    + f'  sha256: {json.dumps(record["source_sha256"])}\n'
    + f'  document_id: {json.dumps(record["document_id"])}\n'
    + f'  document_version_id: {json.dumps(record["document_version_id"])}\n'
    + f'  index_revision_id: {json.dumps(record["index_revision_id"])}\n'
).encode("utf-8")
if run_config.exists():
    if run_config.read_bytes() != payload:
        raise SystemExit("run config changed")
else:
    run_config.write_bytes(payload)
    run_config.chmod(0o600)

def write_qdrant_attestation(directory, *, index_kind, collection_name):
    directory.mkdir(parents=True, exist_ok=True)
    identity = {
        "schema": "docarag.fake-qdrant-index.v1",
        "source_sha256": record["source_sha256"],
        "document_version_id": record["document_version_id"],
        "index_revision_id": record["index_revision_id"],
        "scope_filter": {
            "tenant_id": record["tenant_id"],
            "corpus_id": record["corpus_id"],
            "document_version_id": record["document_version_id"],
            "index_revision_id": record["index_revision_id"],
            "source_sha256": record["source_sha256"],
            "index_kind": index_kind,
        },
        "point_count": 1,
        "collection_name": collection_name,
        "qdrant_endpoint_sha256": hashlib.sha256(
            b"http://qdrant.test:6333"
        ).hexdigest(),
    }
    payload = {
        **identity,
        "identity_sha256": hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest(),
        "state": "content_attested",
    }
    target = directory / "docarag-qdrant-index.ready.json"
    target.write_text(json.dumps(payload), encoding="utf-8")

if stage == "tree":
    # Mirror BookRAG: an existing tree.pkl is loaded/reused, not overwritten.
    # Recovery tests therefore fail unless the controller invalidates it first.
    if not (run_dir / "tree.pkl").exists():
        count_path = run_dir / ".fake-tree-build-count"
        count = int(count_path.read_text() if count_path.exists() else "0") + 1
        count_path.write_text(str(count), encoding="utf-8")
        metadata = {
            "file_name": record["source_name"],
            "source_uri": record["source_uri"],
            "source_sha256": record["source_sha256"],
            "document_version_id": record["document_version_id"],
            "snapshot_path": record["doc_path"],
        }
        node = SimpleNamespace(
            index_id=0,
            parent=None,
            type="root",
            meta_info=SimpleNamespace(),
            summary="",
        )
        tree = SimpleNamespace(meta_info=SimpleNamespace(**metadata), nodes=[node])
        with (run_dir / "tree.pkl").open("wb") as output:
            pickle.dump(tree, output)
        (run_dir / "tree.json").write_text(
            json.dumps(
                {
                    "meta_info": metadata,
                    "nodes": [
                        {
                            "index_id": 0,
                            "parent_id": None,
                            "type": "root",
                            "meta_info": {},
                            "summary": "",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
elif stage == "graph":
    graph_path = run_dir / "graph_data.json"
    if graph_path.exists():
        # A valid-looking existing graph is reused by BookRAG. Merely touch it
        # to model save_gbc_index() without changing its content.
        json.loads(graph_path.read_text(encoding="utf-8"))
        graph_path.touch()
    else:
        graph_path.write_text(
            json.dumps(
                {
                    "graph": {"nodes": [{"id": "entity"}], "links": []},
                    "tree2kg": {"0": ["entity"]},
                }
            ),
            encoding="utf-8",
        )
    write_qdrant_attestation(
        run_dir / "entity_attestation_fake",
        index_kind="entity",
        collection_name="bookrag-entities",
    )
elif stage == "vdb":
    fail_marker = run_dir / ".fake-vdb-failed-once"
    if os.environ.get("FAKE_FAIL_VDB_ONCE") == "1" and not fail_marker.exists():
        fail_marker.touch()
        (run_dir / "Tree_vdb").mkdir(exist_ok=True)
        raise SystemExit(42)
    if "index_type: bm25" in config_path.read_text(encoding="utf-8"):
        directory = run_dir / "Tree_vdb"
        directory.mkdir(exist_ok=True)
        (directory / "bm25_index.pkl").write_bytes(b"fake-bm25")
    else:
        write_qdrant_attestation(
            run_dir / "Tree_vdb",
            index_kind="tree",
            collection_name="bookrag-tree",
        )
'''


class IndexDriverIntegrationTests(unittest.TestCase):
    def setUp(self):
        qdrant = patch(
            "Core.utils.artifact_paths.qdrant_collection_state",
            return_value=SimpleNamespace(exists=True, point_count=1),
        )
        qdrant.start()
        self.addCleanup(qdrant.stop)
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        self.repo = self.base / "repo with 'quotes'"
        (self.repo / "Scripts").mkdir(parents=True)
        (self.repo / "Core" / "utils").mkdir(parents=True)
        (self.repo / "Core" / "__init__.py").write_text("", encoding="utf-8")
        (self.repo / "Core" / "utils" / "__init__.py").write_text(
            "", encoding="utf-8"
        )
        (self.repo / "Core" / "utils" / "artifact_paths.py").write_text(
            "from dataclasses import dataclass\n"
            "import hashlib\n"
            "class QdrantProtocolError(RuntimeError): pass\n"
            "class QdrantUnavailableError(RuntimeError): pass\n"
            "@dataclass(frozen=True)\n"
            "class QdrantCollectionState:\n"
            "    exists: bool\n"
            "    point_count: int | None\n"
            "def qdrant_endpoint_sha256(value):\n"
            "    return hashlib.sha256(value.rstrip('/').encode('utf-8')).hexdigest()\n"
            "def qdrant_collection_state(**_kwargs):\n"
            "    return QdrantCollectionState(True, 1)\n",
            encoding="utf-8",
        )
        shutil.copy2(INDEX_DRIVER, self.repo / "Scripts/index_pdf.sh")
        shutil.copytree(REPO / "docarag", self.repo / "docarag")
        (self.repo / "main.py").write_text("", encoding="utf-8")
        self.fake_runtime = self.repo / "fake-runtime"
        self.fake_runtime.write_text(FAKE_RUNTIME, encoding="utf-8")
        self.fake_runtime.chmod(0o755)
        self.runs = self.base / "runs with 'quotes'"
        self.pdf = self.base / "Original pitch deck #1.pdf"
        self.pdf.write_bytes(b"%PDF-1.4\nfake immutable document\n")

    def tearDown(self):
        self.temporary.cleanup()

    def launch(
        self,
        *,
        index_type="gbc",
        fail_vdb_once=False,
        tenant_id="default",
        corpus_id="default",
        routed_profile: Path | None = None,
    ):
        config = self.repo / f"config-{index_type}.yaml"
        config.write_text(
            f"index_type: {index_type}\n"
            "vdb:\n"
            "  tenant_id: default\n"
            "  corpus_id: default\n"
            "  qdrant_url: http://qdrant.test:6333\n"
            "  api_key_env: QDRANT_API_KEY\n"
            "  timeout_seconds: 1.0\n"
            "  manifest_dir_name: Tree_vdb\n",
            encoding="utf-8",
        )
        existing_runs = {
            path.parent for path in self.runs.glob("*/*/run-manifest.json")
        }
        environment = os.environ.copy()
        environment.update(
            {
                "DOCARAG_CONFIG": str(config),
                "DOCARAG_RUNS_DIR": str(self.runs),
                "DOCARAG_PYTHON": str(self.fake_runtime),
                "DOCARAG_INDEX_ADMISSION": "none",
                "DOCARAG_TENANT_ID": tenant_id,
                "DOCARAG_CORPUS_ID": corpus_id,
                "FAKE_FAIL_VDB_ONCE": "1" if fail_vdb_once else "0",
            }
        )
        if routed_profile is not None:
            environment["DOCARAG_ROUTED_PROFILE"] = str(routed_profile)
        result = subprocess.run(
            ["bash", str(self.repo / "Scripts/index_pdf.sh"), str(self.pdf)],
            cwd=self.repo,
            env=environment,
            check=False,
            text=True,
            capture_output=True,
        )
        self.assertEqual(0, result.returncode, result.stderr)
        run_dirs = {
            path.parent for path in self.runs.glob("*/*/run-manifest.json")
        }
        new_runs = run_dirs - existing_runs
        if len(new_runs) == 1:
            selected_run = next(iter(new_runs))
        else:
            self.assertEqual(set(), new_runs)
            self.assertEqual(1, len(existing_runs))
            selected_run = next(iter(existing_runs))
        return result, selected_run, environment

    def test_routed_profile_is_immutable_revision_input(self):
        profile = self.base / "balanced routed.toml"
        profile.write_text(qualified_routed_profile_text(), encoding="utf-8")

        _result, run_dir, _environment = self.launch(routed_profile=profile)
        self.wait_for_status(run_dir, "succeeded")

        manifest = json.loads(
            (run_dir / "run-manifest.json").read_text(encoding="utf-8")
        )
        routed = manifest["routed_profile"]
        self.assertEqual("balanced-local-example", routed["profile_id"])
        self.assertEqual(
            hashlib.sha256(profile.read_bytes()).hexdigest(),
            routed["source_sha256"],
        )
        snapshot = Path(routed["snapshot_path"])
        self.assertEqual(profile.read_bytes(), snapshot.read_bytes())
        self.assertEqual(0o400, stat.S_IMODE(snapshot.stat().st_mode))
        self.assertEqual(str(snapshot), manifest["inputs"]["routed_profile"])

    def wait_for_status(self, run_dir, expected, timeout=15):
        deadline = time.monotonic() + timeout
        state_path = run_dir / "index-state.json"
        state = None
        while time.monotonic() < deadline:
            try:
                state = json.loads(state_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                time.sleep(0.05)
                continue
            if state.get("overall_status") == expected:
                return state
            if state.get("overall_status") in {"failed", "succeeded"}:
                break
            time.sleep(0.05)
        driver_log = run_dir / "logs" / "index-driver.log"
        detail = (
            driver_log.read_text(encoding="utf-8", errors="replace")
            if driver_log.is_file()
            else "<missing driver log>"
        )
        self.fail(
            f"run did not reach {expected}; last state: {state}; "
            f"driver log:\n{detail}"
        )

    def wait_for_new_attempt_status(
        self, run_dir, previous_attempt_id, expected, timeout=15
    ):
        deadline = time.monotonic() + timeout
        state_path = run_dir / "index-state.json"
        state = None
        while time.monotonic() < deadline:
            try:
                state = json.loads(state_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                time.sleep(0.05)
                continue
            if (
                state.get("attempt_id") != previous_attempt_id
                and state.get("overall_status") == expected
            ):
                return state
            time.sleep(0.05)
        self.fail(f"new attempt did not reach {expected}; last state: {state}")

    def test_writer_reader_source_permissions_and_exact_success_dedup(self):
        first, run_dir, environment = self.launch()
        state = self.wait_for_status(run_dir, "succeeded")

        self.assertTrue(all(
            state["stages"][stage]["status"] == "succeeded"
            for stage in ("tree", "graph", "vdb")
        ))
        readiness = inspect_run_readiness(run_dir, purpose="ask")
        self.assertTrue(readiness.ready, readiness)
        tree = json.loads((run_dir / "tree.json").read_text(encoding="utf-8"))
        self.assertEqual(self.pdf.name, tree["meta_info"]["file_name"])

        manifest = json.loads(
            (run_dir / "run-manifest.json").read_text(encoding="utf-8")
        )
        snapshot = Path(manifest["source"]["snapshot_path"])
        self.assertEqual(0o400, stat.S_IMODE(snapshot.stat().st_mode))
        self.assertEqual(0o400, stat.S_IMODE((run_dir / "run_config.yaml").stat().st_mode))

        artifact_manifest_path = run_dir / "artifact-manifest.json"
        self.assertEqual(0o400, stat.S_IMODE(artifact_manifest_path.stat().st_mode))
        artifact_manifest = json.loads(
            artifact_manifest_path.read_text(encoding="utf-8")
        )
        self.assertEqual("artifact-manifest.json", state["artifact_manifest"])
        self.assertEqual("tree.json", artifact_manifest["roles"]["tree_json"])
        self.assertEqual("graph_data.json", artifact_manifest["roles"]["graph"])
        self.assertEqual(
            "Tree_vdb/docarag-qdrant-index.ready.json",
            artifact_manifest["roles"]["retrieval_attestation"],
        )
        self.assertTrue(verify_artifact_manifest(run_dir, deep=True).valid)

        attempts = sorted((run_dir / "attempts").iterdir())
        self.assertEqual(1, len(attempts))
        self.assertEqual(state["attempt_id"], attempts[0].name)
        attempt_state = json.loads(
            (attempts[0] / "state.json").read_text(encoding="utf-8")
        )
        self.assertEqual("succeeded", attempt_state["overall_status"])
        self.assertEqual(
            0o400,
            stat.S_IMODE((attempts[0] / "state.json").stat().st_mode),
        )
        self.assertEqual(
            artifact_manifest_path.read_bytes(),
            (attempts[0] / "artifact-manifest.json").read_bytes(),
        )
        runtime_manifest = json.loads(
            (attempts[0] / "runtime.json").read_text(encoding="utf-8")
        )
        self.assertRegex(
            runtime_manifest["environment_fingerprint_sha256"], r"^[0-9a-f]{64}$"
        )
        self.assertEqual(
            runtime_manifest["environment_fingerprint_sha256"],
            artifact_manifest["runtime_manifest"]["environment_fingerprint_sha256"],
        )
        self.assertEqual(0o400, stat.S_IMODE((attempts[0] / "runtime.json").stat().st_mode))
        for stage in ("tree", "graph", "vdb"):
            latest_log = run_dir / "logs" / f"index-{stage}.log"
            self.assertTrue(latest_log.is_symlink())
            self.assertEqual(
                (attempts[0] / f"index-{stage}.log").resolve(),
                latest_log.resolve(),
            )

        progress = subprocess.run(
            ["bash", str(PROGRESS), str(run_dir)],
            cwd=REPO,
            env=environment,
            check=False,
            text=True,
            capture_output=True,
        )
        self.assertEqual(0, progress.returncode, progress.stderr)
        self.assertIn("index complete", progress.stdout)

        started_at = state["started_at"]
        second = subprocess.run(
            ["bash", str(self.repo / "Scripts/index_pdf.sh"), str(self.pdf)],
            cwd=self.repo,
            env=environment,
            check=False,
            text=True,
            capture_output=True,
        )
        self.assertEqual(0, second.returncode, second.stderr)
        self.assertIn("already indexed", second.stdout)
        current = json.loads((run_dir / "index-state.json").read_text(encoding="utf-8"))
        self.assertEqual(started_at, current["started_at"])
        self.assertEqual(1, len(list((run_dir / "attempts").iterdir())))

    def test_tenant_scope_changes_index_revision_but_not_document_version(self):
        _, first_run, _ = self.launch(tenant_id="tenant-a")
        self.wait_for_status(first_run, "succeeded")
        _, second_run, _ = self.launch(tenant_id="tenant-b")
        self.wait_for_status(second_run, "succeeded")

        first = json.loads(
            (first_run / "run-manifest.json").read_text(encoding="utf-8")
        )
        second = json.loads(
            (second_run / "run-manifest.json").read_text(encoding="utf-8")
        )
        self.assertNotEqual(first_run, second_run)
        self.assertNotEqual(first["revision_id"], second["revision_id"])
        self.assertEqual(first["document_id"], second["document_id"])
        self.assertEqual(
            first["document_version_id"], second["document_version_id"]
        )
        self.assertEqual("tenant-a", first["tenant_id"])
        self.assertEqual("tenant-b", second["tenant_id"])
        self.assertTrue(first["source"]["uri"].startswith("file://"))

    def test_failed_vdb_attempt_can_be_repaired(self):
        _, run_dir, _ = self.launch(fail_vdb_once=True)
        failed = self.wait_for_status(run_dir, "failed")
        self.assertEqual("failed", failed["stages"]["vdb"]["status"])
        self.assertEqual(42, failed["stages"]["vdb"]["exit_code"])
        first_attempt = run_dir / "attempts" / failed["attempt_id"]
        self.assertEqual(
            "failed",
            json.loads((first_attempt / "state.json").read_text(encoding="utf-8"))[
                "overall_status"
            ],
        )

        _, same_run, _ = self.launch(fail_vdb_once=True)
        self.assertEqual(run_dir, same_run)
        repaired = self.wait_for_new_attempt_status(
            run_dir, failed["attempt_id"], "succeeded"
        )
        self.assertEqual("succeeded", repaired["stages"]["vdb"]["status"])
        self.assertEqual(2, repaired["attempt_number"])
        self.assertEqual(2, len(list((run_dir / "attempts").iterdir())))
        self.assertEqual(
            "failed",
            json.loads((first_attempt / "state.json").read_text(encoding="utf-8"))[
                "overall_status"
            ],
        )

    def test_same_size_artifact_corruption_forces_a_repair_attempt(self):
        _, run_dir, environment = self.launch()
        first = self.wait_for_status(run_dir, "succeeded")
        tree_pickle = run_dir / "tree.pkl"
        original = tree_pickle.read_bytes()
        tree_pickle.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
        self.assertFalse(verify_artifact_manifest(run_dir, deep=True).valid)

        second = subprocess.run(
            ["bash", str(self.repo / "Scripts/index_pdf.sh"), str(self.pdf)],
            cwd=self.repo,
            env=environment,
            check=False,
            text=True,
            capture_output=True,
        )
        self.assertEqual(0, second.returncode, second.stderr)
        self.assertNotIn("already indexed", second.stdout)
        repaired = self.wait_for_new_attempt_status(
            run_dir, first["attempt_id"], "succeeded"
        )
        self.assertNotEqual(first["attempt_id"], repaired["attempt_id"])
        self.assertEqual(2, repaired["attempt_number"])
        self.assertTrue(verify_artifact_manifest(run_dir, deep=True).valid)
        self.assertEqual("2", (run_dir / ".fake-tree-build-count").read_text())
        recovery = run_dir / "attempts" / repaired["attempt_id"] / "recovery"
        self.assertEqual(
            original[:-1] + bytes([original[-1] ^ 1]),
            (recovery / "tree.pkl").read_bytes(),
        )

    def test_tree_pair_source_mismatch_is_rebuilt_even_without_root_manifest(self):
        _, run_dir, environment = self.launch()
        first = self.wait_for_status(run_dir, "succeeded")
        tree_json = json.loads((run_dir / "tree.json").read_text(encoding="utf-8"))
        tree_json["meta_info"]["source_sha256"] = "0" * 64
        (run_dir / "tree.json").write_text(json.dumps(tree_json), encoding="utf-8")
        (run_dir / "artifact-manifest.json").unlink()

        result = subprocess.run(
            ["bash", str(self.repo / "Scripts/index_pdf.sh"), str(self.pdf)],
            cwd=self.repo,
            env=environment,
            check=False,
            text=True,
            capture_output=True,
        )
        self.assertEqual(0, result.returncode, result.stderr)
        repaired = self.wait_for_new_attempt_status(
            run_dir, first["attempt_id"], "succeeded"
        )
        self.assertEqual("2", (run_dir / ".fake-tree-build-count").read_text())
        rebuilt = json.loads((run_dir / "tree.json").read_text(encoding="utf-8"))
        manifest = json.loads((run_dir / "run-manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(
            manifest["source"]["sha256"], rebuilt["meta_info"]["source_sha256"]
        )
        self.assertTrue(verify_artifact_manifest(run_dir, deep=True).valid)

    def test_graph_digest_mismatch_rebuilds_graph_chain_but_reuses_valid_tree(self):
        _, run_dir, environment = self.launch()
        first = self.wait_for_status(run_dir, "succeeded")
        graph_path = run_dir / "graph_data.json"
        original = graph_path.read_bytes()
        corrupted = original.replace(b'"entity"', b'"Entity"', 1)
        self.assertEqual(len(original), len(corrupted))
        graph_path.write_bytes(corrupted)
        self.assertFalse(verify_artifact_manifest(run_dir, deep=True).valid)

        result = subprocess.run(
            ["bash", str(self.repo / "Scripts/index_pdf.sh"), str(self.pdf)],
            cwd=self.repo,
            env=environment,
            check=False,
            text=True,
            capture_output=True,
        )
        self.assertEqual(0, result.returncode, result.stderr)
        repaired = self.wait_for_new_attempt_status(
            run_dir, first["attempt_id"], "succeeded"
        )
        self.assertEqual("1", (run_dir / ".fake-tree-build-count").read_text())
        recovery = run_dir / "attempts" / repaired["attempt_id"] / "recovery"
        self.assertEqual(corrupted, (recovery / "graph_data.json").read_bytes())
        self.assertEqual(original, graph_path.read_bytes())
        self.assertTrue(verify_artifact_manifest(run_dir, deep=True).valid)

    def test_bm25_is_a_supported_verified_retrieval_artifact(self):
        _, run_dir, _ = self.launch(index_type="bm25")
        self.wait_for_status(run_dir, "succeeded")
        self.assertTrue((run_dir / "Tree_vdb/bm25_index.pkl").is_file())
        self.assertTrue(inspect_run_readiness(run_dir, purpose="ask").ready)

    def test_default_gpu_admission_serializes_distinct_document_runs(self):
        second_pdf = self.base / "Second document.pdf"
        second_pdf.write_bytes(b"%PDF-1.4\nsecond immutable document\n")
        config = self.repo / "config-gpu-lock.yaml"
        config.write_text(
            "index_type: gbc\n"
            "vdb:\n"
            "  manifest_dir_name: Tree_vdb\n",
            encoding="utf-8",
        )
        environment = os.environ.copy()
        environment.update(
            {
                "DOCARAG_CONFIG": str(config),
                "DOCARAG_RUNS_DIR": str(self.runs),
                "DOCARAG_PYTHON": str(self.fake_runtime),
                "DOCARAG_INDEX_ADMISSION": "gpu-lock",
                "FAKE_STAGE_SLEEP": "0.2",
            }
        )

        for document in (self.pdf, second_pdf):
            result = subprocess.run(
                ["bash", str(self.repo / "Scripts/index_pdf.sh"), str(document)],
                cwd=self.repo,
                env=environment,
                check=False,
                text=True,
                capture_output=True,
            )
            self.assertEqual(0, result.returncode, result.stderr)

        deadline = time.monotonic() + 15
        run_dirs = []
        while time.monotonic() < deadline:
            run_dirs = [path.parent for path in self.runs.glob("*/*/run-manifest.json")]
            if len(run_dirs) == 2:
                break
            time.sleep(0.05)
        self.assertEqual(2, len(run_dirs))
        states = [self.wait_for_status(run, "succeeded") for run in run_dirs]
        states.sort(key=lambda state: state["stages"]["tree"]["started_at"])
        first, second = states
        self.assertGreaterEqual(
            second["stages"]["tree"]["started_at"],
            first["stages"]["vdb"]["ended_at"],
        )

    def test_termination_stops_active_stage_and_records_attempt_failure(self):
        config = self.repo / "config-termination.yaml"
        config.write_text(
            "index_type: gbc\n"
            "vdb:\n"
            "  manifest_dir_name: Tree_vdb\n",
            encoding="utf-8",
        )
        environment = os.environ.copy()
        environment.update(
            {
                "DOCARAG_CONFIG": str(config),
                "DOCARAG_RUNS_DIR": str(self.runs),
                "DOCARAG_PYTHON": str(self.fake_runtime),
                "DOCARAG_INDEX_ADMISSION": "none",
                "DOCARAG_TERM_GRACE_SECONDS": "1",
                "FAKE_IGNORE_TERM": "1",
                "FAKE_STAGE_SLEEP": "8",
            }
        )
        launched = subprocess.run(
            ["bash", str(self.repo / "Scripts/index_pdf.sh"), str(self.pdf)],
            cwd=self.repo,
            env=environment,
            check=False,
            text=True,
            capture_output=True,
        )
        self.assertEqual(0, launched.returncode, launched.stderr)
        run_dir = next(path.parent for path in self.runs.glob("*/*/run-manifest.json"))

        deadline = time.monotonic() + 5
        running = None
        while time.monotonic() < deadline:
            try:
                running = json.loads(
                    (run_dir / "index-state.json").read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError):
                time.sleep(0.05)
                continue
            if (
                running.get("current_stage") == "tree"
                and (run_dir / ".fake-stage-tree-started").exists()
            ):
                break
            time.sleep(0.05)
        self.assertIsNotNone(running)
        self.assertEqual("tree", running.get("current_stage"))

        stopped_at = time.monotonic()
        os.kill(running["pid"], 15)
        failed = self.wait_for_status(run_dir, "failed", timeout=5)
        self.assertLess(time.monotonic() - stopped_at, 5)
        self.assertGreaterEqual(time.monotonic() - stopped_at, 0.8)
        self.assertEqual(143, failed["exit_code"])
        self.assertEqual("failed", failed["stages"]["tree"]["status"])
        attempt_state = json.loads(
            (run_dir / "attempts" / failed["attempt_id"] / "state.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual("failed", attempt_state["overall_status"])

    def test_termination_while_waiting_for_gpu_never_acquires_or_publishes(self):
        config = self.repo / "config-admission-termination.yaml"
        config.write_text(
            "index_type: gbc\n"
            "vdb:\n"
            "  manifest_dir_name: Tree_vdb\n",
            encoding="utf-8",
        )
        self.runs.mkdir(parents=True, exist_ok=True)
        admission_lock = self.runs / ".gpu-index.lock"
        holder = subprocess.Popen(
            ["flock", str(admission_lock), "sleep", "8"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            time.sleep(0.15)
            environment = os.environ.copy()
            environment.update(
                {
                    "DOCARAG_CONFIG": str(config),
                    "DOCARAG_RUNS_DIR": str(self.runs),
                    "DOCARAG_PYTHON": str(self.fake_runtime),
                    "DOCARAG_INDEX_ADMISSION": "gpu-lock",
                    "DOCARAG_TERM_GRACE_SECONDS": "1",
                }
            )
            launched = subprocess.run(
                ["bash", str(self.repo / "Scripts/index_pdf.sh"), str(self.pdf)],
                cwd=self.repo,
                env=environment,
                check=False,
                text=True,
                capture_output=True,
            )
            self.assertEqual(0, launched.returncode, launched.stderr)
            run_dir = next(
                path.parent for path in self.runs.glob("*/*/run-manifest.json")
            )
            deadline = time.monotonic() + 5
            waiting = None
            while time.monotonic() < deadline:
                try:
                    waiting = json.loads((run_dir / "index-state.json").read_text())
                except (OSError, json.JSONDecodeError):
                    time.sleep(0.05)
                    continue
                if waiting.get("phase") == "waiting_for_capacity":
                    break
                time.sleep(0.05)
            self.assertEqual("waiting_for_capacity", waiting.get("phase"))
            os.kill(waiting["pid"], 15)
            failed = self.wait_for_status(run_dir, "failed", timeout=5)
            attempt = run_dir / "attempts" / failed["attempt_id"]
            self.assertFalse((attempt / "runtime.json").exists())
            self.assertFalse((run_dir / "artifact-manifest.json").exists())

            holder.terminate()
            holder.wait(timeout=3)
            time.sleep(0.3)
            current = json.loads((run_dir / "index-state.json").read_text())
            self.assertEqual("failed", current["overall_status"])
            self.assertFalse((attempt / "runtime.json").exists())
        finally:
            if holder.poll() is None:
                holder.kill()
                holder.wait(timeout=3)

    def test_termination_during_publication_cannot_publish_after_failure(self):
        config = self.repo / "config-publish-termination.yaml"
        config.write_text(
            "index_type: gbc\n"
            "vdb:\n"
            "  manifest_dir_name: Tree_vdb\n",
            encoding="utf-8",
        )
        environment = os.environ.copy()
        environment.update(
            {
                "DOCARAG_CONFIG": str(config),
                "DOCARAG_RUNS_DIR": str(self.runs),
                "DOCARAG_PYTHON": str(self.fake_runtime),
                "DOCARAG_INDEX_ADMISSION": "none",
                "DOCARAG_TERM_GRACE_SECONDS": "1",
                "FAKE_PUBLISH_SLEEP": "8",
            }
        )
        launched = subprocess.run(
            ["bash", str(self.repo / "Scripts/index_pdf.sh"), str(self.pdf)],
            cwd=self.repo,
            env=environment,
            check=False,
            text=True,
            capture_output=True,
        )
        self.assertEqual(0, launched.returncode, launched.stderr)
        run_dir = next(path.parent for path in self.runs.glob("*/*/run-manifest.json"))
        deadline = time.monotonic() + 8
        state = None
        while time.monotonic() < deadline:
            try:
                state = json.loads((run_dir / "index-state.json").read_text())
            except (OSError, json.JSONDecodeError):
                time.sleep(0.05)
                continue
            if (run_dir / ".fake-publish-started").exists():
                break
            time.sleep(0.05)
        self.assertTrue((run_dir / ".fake-publish-started").exists(), state)
        os.kill(state["pid"], 15)
        failed = self.wait_for_status(run_dir, "failed", timeout=5)
        self.assertEqual(143, failed["exit_code"])
        time.sleep(0.3)
        self.assertFalse((run_dir / "artifact-manifest.json").exists())
        self.assertNotEqual("succeeded", failed["overall_status"])

    def test_stale_running_attempt_is_reconciled_once_revision_lock_is_free(self):
        config = self.repo / "config-stale.yaml"
        config.write_text(
            "index_type: gbc\n"
            "vdb:\n"
            "  manifest_dir_name: Tree_vdb\n",
            encoding="utf-8",
        )
        environment = os.environ.copy()
        environment.update(
            {
                "DOCARAG_CONFIG": str(config),
                "DOCARAG_RUNS_DIR": str(self.runs),
                "DOCARAG_PYTHON": str(self.fake_runtime),
                "DOCARAG_INDEX_ADMISSION": "none",
                "FAKE_STAGE_SLEEP": "1.2",
            }
        )
        launched = subprocess.run(
            ["bash", str(self.repo / "Scripts/index_pdf.sh"), str(self.pdf)],
            cwd=self.repo,
            env=environment,
            check=False,
            text=True,
            capture_output=True,
        )
        self.assertEqual(0, launched.returncode, launched.stderr)
        run_dir = next(path.parent for path in self.runs.glob("*/*/run-manifest.json"))
        deadline = time.monotonic() + 5
        stale = None
        while time.monotonic() < deadline:
            try:
                stale = json.loads((run_dir / "index-state.json").read_text())
            except (OSError, json.JSONDecodeError):
                time.sleep(0.05)
                continue
            if (
                stale.get("current_stage") == "tree"
                and (run_dir / ".fake-stage-tree-started").exists()
            ):
                break
            time.sleep(0.05)
        self.assertEqual("tree", stale.get("current_stage"))
        old_attempt_id = stale["attempt_id"]
        os.kill(stale["pid"], 9)

        # The orphaned stage inherited the lock. Once it exits naturally, the
        # next launcher proves the old driver is gone by acquiring that lock.
        time.sleep(1.5)
        retry_environment = environment.copy()
        retry_environment["FAKE_STAGE_SLEEP"] = "0"
        retried = subprocess.run(
            ["bash", str(self.repo / "Scripts/index_pdf.sh"), str(self.pdf)],
            cwd=self.repo,
            env=retry_environment,
            check=False,
            text=True,
            capture_output=True,
        )
        self.assertEqual(0, retried.returncode, retried.stderr)
        repaired = self.wait_for_new_attempt_status(
            run_dir, old_attempt_id, "succeeded", timeout=15
        )
        prior = json.loads(
            (run_dir / "attempts" / old_attempt_id / "state.json").read_text()
        )
        self.assertEqual("failed", prior["overall_status"])
        self.assertEqual("driver_lock_lost", prior["termination_reason"])
        self.assertIn("abandoned/interrupted", prior["error"])
        self.assertEqual(2, repaired["attempt_number"])


if __name__ == "__main__":
    unittest.main()
