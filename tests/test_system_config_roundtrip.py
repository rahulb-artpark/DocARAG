from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from pydantic import ValidationError
import yaml

from retrieval_engine.configuration.system import SystemConfig, load_system_config


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _minimal_config(*, rag: dict | None = None) -> dict:
    return {
        "mineru": {
            "backend": "pipeline",
            "method": "auto",
            "lang": "en",
        },
        "rag": rag
        or {
            "strategy_config": {
                "strategy": "gbc",
                "variant": "standard",
            }
        },
    }


class SystemConfigRoundTripTests(unittest.TestCase):
    def write_and_load(self, payload: dict) -> SystemConfig:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "system.yaml"
            path.write_text(
                yaml.safe_dump(payload, allow_unicode=True),
                encoding="utf-8",
            )
            return load_system_config(path)

    def assert_yaml_round_trip(self, config: SystemConfig) -> SystemConfig:
        reloaded = self.write_and_load(config.model_dump(mode="python"))
        self.assertEqual(
            config.model_dump(mode="python"),
            reloaded.model_dump(mode="python"),
        )
        return reloaded

    def test_canonical_rag_envelope_is_not_wrapped_twice(self):
        payload = _minimal_config(
            rag={
                "strategy_config": {
                    "strategy": "gbc",
                    "variant": "wo_graph",
                    "sim_threshold_e": 0.47,
                }
            }
        )

        config = self.write_and_load(payload)

        self.assertEqual("wo_graph", config.rag.strategy_config.variant)
        self.assertEqual(0.47, config.rag.strategy_config.sim_threshold_e)
        dumped_rag = config.model_dump()["rag"]
        self.assertEqual({"strategy_config"}, set(dumped_rag))
        self.assertNotIn("strategy_config", dumped_rag["strategy_config"])
        self.assert_yaml_round_trip(config)

    def test_legacy_flat_rag_and_similarity_alias_migrate_to_canonical_shape(self):
        payload = _minimal_config(
            rag={
                "strategy": "graph",
                "topk_docs": 9,
                "sim_threshold": 0.61,
            }
        )

        config = self.write_and_load(payload)

        self.assertEqual("graph", config.rag.strategy_config.strategy)
        self.assertEqual(0.61, config.rag.strategy_config.sim_threshold_e)
        dumped_strategy = config.model_dump()["rag"]["strategy_config"]
        self.assertEqual(0.61, dumped_strategy["sim_threshold_e"])
        self.assertNotIn("sim_threshold", dumped_strategy)
        self.assert_yaml_round_trip(config)

    def test_full_synthetic_run_config_preserves_semantics(self):
        payload = _minimal_config()
        payload.update(
            {
                "index_type": "gbc",
                "pdf_path": "/documents/source.pdf",
                "save_path": "/indexes/document-1",
                "source": {
                    "display_name": "source.pdf",
                    "original_uri": "file:///documents/source.pdf",
                    "sha256": "a" * 64,
                    "snapshot_path": "/indexes/document-1/source.pdf",
                },
                "llm": {
                    "model_name": "example/text-model",
                    "backend": "openai",
                    "api_base": "http://localhost:8003/v1",
                    "api_key": "test",
                    "max_workers": 3,
                    "max_in_flight": 5,
                },
                "vlm": {
                    "model_name": "example/vision-model",
                    "backend": "ollama",
                    "api_base": "http://localhost:11434",
                    "max_workers": 2,
                    "max_in_flight": 2,
                },
                "graph": {
                    "extraction_max_workers": 2,
                    "extraction_max_in_flight": 4,
                    "extraction_batch_size": 16,
                },
            }
        )

        config = self.write_and_load(payload)

        reloaded = self.assert_yaml_round_trip(config)
        self.assertEqual(5, reloaded.llm.max_in_flight)
        self.assertEqual(16, reloaded.graph.extraction_batch_size)
        self.assertEqual("a" * 64, reloaded.source.sha256)

    def test_unknown_root_field_is_rejected(self):
        payload = _minimal_config()
        payload["typo_save_path"] = "/tmp/lost"

        with self.assertRaises(ValidationError) as raised:
            self.write_and_load(payload)

        self.assertIn("typo_save_path", str(raised.exception))
        self.assertIn("Extra inputs are not permitted", str(raised.exception))

    def test_mixed_legacy_and_canonical_rag_shapes_are_rejected(self):
        payload = _minimal_config(
            rag={
                "strategy": "gbc",
                "strategy_config": {"strategy": "gbc"},
            }
        )

        with self.assertRaisesRegex(ValueError, "Ambiguous 'rag' configuration"):
            self.write_and_load(payload)

    def test_unsupported_index_block_has_actionable_error(self):
        payload = _minimal_config()
        payload["index"] = {"chunk_size": 512, "overlap": 50}

        with self.assertRaisesRegex(
            ValueError,
            "index.chunk_size and index.overlap have no runtime consumer",
        ):
            self.write_and_load(payload)

    def test_checked_in_runtime_configs_load_and_round_trip(self):
        config_directory = REPOSITORY_ROOT / "config"
        paths = sorted(config_directory.glob("*.yaml"))
        runtime_paths = [path for path in paths if path.name != "default.yaml"]
        self.assertGreater(len(runtime_paths), 0)

        for path in runtime_paths:
            with self.subTest(filename=path.name):
                config = load_system_config(path)
                self.assert_yaml_round_trip(config)

    def test_checked_in_default_is_explicitly_an_unrendered_template(self):
        path = REPOSITORY_ROOT / "config" / "default.yaml"

        with self.assertRaises(ValidationError) as raised:
            load_system_config(path)

        self.assertIn("max_length", str(raised.exception))
        self.assertIn("valid integer", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
