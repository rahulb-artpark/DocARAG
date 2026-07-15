from __future__ import annotations

import json
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from Core.inference import run_rag
from Core.utils.query_cache import QueryCacheContext, query_prompt_code_sha256


def cache_context(
    *,
    artifact: str = "a",
    config: str = "b",
    model: str = "c",
    prompt: str = "d",
) -> QueryCacheContext:
    return QueryCacheContext(
        artifact_set_sha256=artifact * 64,
        resolved_config_sha256=config * 64,
        model_identity_sha256=model * 64,
        query_prompt_code_sha256=prompt * 64,
    )


class RecordingRAG:
    def __init__(self, *, fail_on_call: int | None = None) -> None:
        self.queries: list[str] = []
        self.fail_on_call = fail_on_call
        self.closed = False

    def generation(self, query: str, output_dir: Path):
        self.queries.append(query)
        if self.fail_on_call == len(self.queries):
            raise RuntimeError("injected query failure")
        return f"answer-{len(self.queries)}-{query}", [f"node-{query}"]

    def close(self) -> None:
        self.closed = True


def run_records(
    rag: RecordingRAG,
    output: Path,
    records,
    *,
    context: QueryCacheContext | None,
    force: bool = False,
) -> None:
    run_rag(
        rag,
        output,
        data_iterable=records,
        cache_context=context,
        force_reprocess=force,
        close_agent_on_exit=False,
    )


class QueryCacheStreamingTests(unittest.TestCase):
    def test_resolved_config_identity_is_secret_free_and_model_specific(self):
        base = {
            "llm": {
                "backend": "openai",
                "model_name": "model-a",
                "api_key": "first-secret",
                "temperature": 0.1,
            },
            "retrieval": {"topk": 5},
        }
        first = QueryCacheContext.from_system_config(
            base,
            artifact_set_sha256="a" * 64,
            prompt_code_sha256="d" * 64,
        )
        changed_secret = QueryCacheContext.from_system_config(
            {**base, "llm": {**base["llm"], "api_key": "second-secret"}},
            artifact_set_sha256="a" * 64,
            prompt_code_sha256="d" * 64,
        )
        changed_config = QueryCacheContext.from_system_config(
            {**base, "retrieval": {"topk": 6}},
            artifact_set_sha256="a" * 64,
            prompt_code_sha256="d" * 64,
        )
        changed_model = QueryCacheContext.from_system_config(
            {**base, "llm": {**base["llm"], "model_name": "model-b"}},
            artifact_set_sha256="a" * 64,
            prompt_code_sha256="d" * 64,
        )

        self.assertEqual(first, changed_secret)
        self.assertNotEqual(
            first.resolved_config_sha256,
            changed_config.resolved_config_sha256,
        )
        self.assertEqual(
            first.model_identity_sha256,
            changed_config.model_identity_sha256,
        )
        self.assertNotEqual(
            first.model_identity_sha256,
            changed_model.model_identity_sha256,
        )
        self.assertNotIn("first-secret", repr(first))

    def test_query_code_identity_changes_with_prompt_source_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "Core" / "prompts").mkdir(parents=True)
            (root / "docarag").mkdir()
            prompt = root / "Core" / "prompts" / "answer.py"
            prompt.write_text('PROMPT = "first"\n', encoding="utf-8")
            (root / "docarag" / "__init__.py").write_text("", encoding="utf-8")
            first = query_prompt_code_sha256(root)
            prompt.write_text('PROMPT = "second"\n', encoding="utf-8")
            second = query_prompt_code_sha256(root)

        self.assertNotEqual(first, second)

    def test_reorder_reuses_by_identity_and_preserves_current_order(self):
        rag = RecordingRAG()
        context = cache_context()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            run_records(
                rag,
                output,
                [
                    {"question": "alpha", "row_id": "a-old"},
                    {"question": "beta", "row_id": "b-old"},
                ],
                context=context,
            )
            run_records(
                rag,
                output,
                [
                    {"question": "beta", "row_id": "b-current"},
                    {"question": "alpha", "row_id": "a-current"},
                ],
                context=context,
            )
            final = json.loads((output / "final_results.json").read_text())

        self.assertEqual(["alpha", "beta"], rag.queries)
        self.assertEqual(["beta", "alpha"], [item["question"] for item in final])
        self.assertEqual(
            ["b-current", "a-current"], [item["row_id"] for item in final]
        )

    def test_same_ordinal_different_question_never_cross_reuses(self):
        rag = RecordingRAG()
        context = cache_context()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            run_records(rag, output, [{"question": "first"}], context=context)
            run_records(rag, output, [{"question": "second"}], context=context)
            query_dirs = sorted(output.glob("query_*"))
            final = json.loads((output / "final_results.json").read_text())

        self.assertEqual(["first", "second"], rag.queries)
        self.assertEqual(2, len(query_dirs))
        self.assertEqual("second", final[0]["question"])
        self.assertIn("second", final[0]["output"])

    def test_every_external_identity_dimension_invalidates_reuse(self):
        rag = RecordingRAG()
        contexts = [
            cache_context(),
            cache_context(artifact="e"),
            cache_context(artifact="e", config="f"),
            cache_context(artifact="e", config="f", model="1"),
            cache_context(artifact="e", config="f", model="1", prompt="2"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            for context in contexts:
                run_records(rag, output, [{"question": "same"}], context=context)

        self.assertEqual(["same"] * len(contexts), rag.queries)

    def test_missing_corrupt_or_mismatched_commit_is_never_reused(self):
        rag = RecordingRAG()
        context = cache_context()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            row = [{"question": "cache me"}]
            run_records(rag, output, row, context=context)
            query_dir = next(output.glob("query_*"))
            commit = query_dir / "cache-binding.json"
            result = query_dir / "result.json"

            commit.unlink()
            run_records(rag, output, row, context=context)

            commit.write_text("{not-json", encoding="utf-8")
            run_records(rag, output, row, context=context)

            payload = json.loads(commit.read_text(encoding="utf-8"))
            payload["binding"]["question_sha256"] = "0" * 64
            commit.write_text(json.dumps(payload), encoding="utf-8")
            run_records(rag, output, row, context=context)

            result.write_text("{broken", encoding="utf-8")
            run_records(rag, output, row, context=context)

        self.assertEqual(["cache me"] * 5, rag.queries)

    def test_duplicate_question_reuses_evidence_but_keeps_each_row_metadata(self):
        rag = RecordingRAG()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            run_records(
                rag,
                output,
                [
                    {"question": "duplicate", "row_id": 1, "expected": "old"},
                    {"question": "duplicate", "row_id": 2, "expected": "new"},
                ],
                context=cache_context(),
            )
            final = json.loads((output / "final_results.json").read_text())
            cached = json.loads(
                next(output.glob("query_*/result.json")).read_text(encoding="utf-8")
            )

        self.assertEqual(["duplicate"], rag.queries)
        self.assertEqual([1, 2], [item["row_id"] for item in final])
        self.assertEqual(final[0]["output"], final[1]["output"])
        self.assertEqual(2, cached["row_id"])
        self.assertEqual("new", cached["expected"])

    def test_without_safe_context_reuse_is_disabled(self):
        rag = RecordingRAG()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            row = [{"question": "uncached"}]
            run_records(rag, output, row, context=None)
            run_records(rag, output, row, context=None)
            query_dir = next(output.glob("query_*"))

        self.assertEqual(["uncached", "uncached"], rag.queries)
        self.assertFalse((query_dir / "cache-binding.json").exists())

    def test_question_bytes_are_not_normalized(self):
        rag = RecordingRAG()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            run_records(rag, output, [{"question": "text"}], context=cache_context())
            run_records(
                rag, output, [{"question": "text "}], context=cache_context()
            )

        self.assertEqual(["text", "text "], rag.queries)

    def test_generator_is_consumed_incrementally_and_final_is_atomic(self):
        rag = RecordingRAG()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)

            def records():
                yield {"question": "one"}
                self.assertTrue(any(output.glob("query_*/result.json")))
                self.assertFalse((output / "final_results.json").exists())
                yield {"question": "two"}

            run_records(rag, output, records(), context=cache_context())
            final = json.loads((output / "final_results.json").read_text())
            temporary_files = list(output.glob(".final_results.json.*"))

        self.assertEqual(["one", "two"], [item["question"] for item in final])
        self.assertEqual([], temporary_files)

    def test_jsonl_streams_and_dataframe_does_not_call_to_dict(self):
        rag = RecordingRAG()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jsonl = root / "queries.jsonl"
            jsonl.write_text(
                '{"question":"jsonl-one"}\n\n{"question":"jsonl-two"}\n',
                encoding="utf-8",
            )
            jsonl_output = root / "jsonl-output"
            run_rag(
                rag,
                jsonl_output,
                dataset_path=str(jsonl),
                cache_context=cache_context(),
                close_agent_on_exit=False,
            )

            frame_output = root / "frame-output"
            frame = pd.DataFrame(
                [{"question": "frame-one"}, {"question": "frame-two"}]
            )
            with patch.object(
                pd.DataFrame,
                "to_dict",
                side_effect=AssertionError("whole-frame materialization"),
            ):
                run_rag(
                    rag,
                    frame_output,
                    data_df=frame,
                    cache_context=cache_context(),
                    close_agent_on_exit=False,
                )

        self.assertEqual(
            ["jsonl-one", "jsonl-two", "frame-one", "frame-two"], rag.queries
        )

    def test_failure_removes_old_aggregate_and_partial_temporary_file(self):
        rag = RecordingRAG(fail_on_call=2)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            old_final = output / "final_results.json"
            old_final.write_text('[{"stale":true}]', encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "injected query failure"):
                run_records(
                    rag,
                    output,
                    [{"question": "success"}, {"question": "failure"}],
                    context=cache_context(),
                )

            query_results = list(output.glob("query_*/result.json"))
            temporary_files = list(output.glob(".final_results.json.*"))

        self.assertFalse(old_final.exists())
        self.assertEqual(1, len(query_results))
        self.assertEqual([], temporary_files)

    def test_cache_and_aggregate_files_are_private(self):
        rag = RecordingRAG()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            run_records(rag, output, [{"question": "private"}], context=cache_context())
            paths = [
                output / "final_results.json",
                *output.glob("query_*/result.json"),
                *output.glob("query_*/cache-binding.json"),
            ]
            modes = [stat.S_IMODE(path.stat().st_mode) for path in paths]

        self.assertTrue(paths)
        self.assertEqual([0o600] * len(paths), modes)


if __name__ == "__main__":
    unittest.main()
