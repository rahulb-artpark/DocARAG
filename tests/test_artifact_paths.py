from pathlib import Path
import tempfile
import unittest

from retrieval_engine.utils.artifact_paths import resolve_run_artifact_path


class ArtifactPathTests(unittest.TestCase):
    def test_relative_artifact_path_is_resolved_below_run(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "run"
            run.mkdir()

            result = resolve_run_artifact_path(
                run, "indexes/tree", label="tree index"
            )

            self.assertEqual((run / "indexes/tree").resolve(), result)

    def test_parent_absolute_and_symlink_escapes_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            run = base / "run"
            outside = base / "outside"
            run.mkdir()
            outside.mkdir()
            (run / "escape").symlink_to(outside, target_is_directory=True)

            for configured in ("../outside", outside):
                with self.subTest(configured=configured):
                    with self.assertRaisesRegex(ValueError, "must stay below"):
                        resolve_run_artifact_path(
                            run, configured, label="tree index"
                        )
            with self.assertRaisesRegex(ValueError, "symbolic links"):
                resolve_run_artifact_path(
                    run, "escape/index", label="tree index"
                )

    def test_symlink_within_run_is_rejected_before_destructive_repair(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "run"
            target = run / "other-index"
            target.mkdir(parents=True)
            (run / "configured-index").symlink_to(target, target_is_directory=True)

            with self.assertRaisesRegex(ValueError, "symbolic links"):
                resolve_run_artifact_path(
                    run, "configured-index", label="tree index"
                )


if __name__ == "__main__":
    unittest.main()
