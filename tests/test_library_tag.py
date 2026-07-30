"""A library tag is the reader's own label, stored with the document it labels.

Kept in the run directory for the same reason the chat history is: deleting a
document rmtree's its run, so the label goes with it rather than being left
behind pointing at nothing. run-manifest.json is provenance and is hashed, so a
user-editable field has no business in it.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
WEB_PATH = REPO / "Scripts" / "web.py"


def load_web_module():
    spec = importlib.util.spec_from_file_location("docarag_test_library_tag", WEB_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class LibraryTagTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.web = load_web_module()

    def setUp(self):
        self.run_dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.run_dir, ignore_errors=True)

    def tag_file(self):
        return self.run_dir / self.web.LIBRARY_TAG_FILE


class ATagIsStoredWithItsDocument(LibraryTagTest):
    def test_an_untagged_run_reports_none(self):
        self.assertIsNone(self.web.read_run_tag(self.run_dir))

    def test_a_tag_survives_a_round_trip(self):
        self.web.write_run_tag(self.run_dir, "Class XI")
        self.assertEqual(self.web.read_run_tag(self.run_dir), "Class XI")

    def test_it_is_written_inside_the_run_directory(self):
        self.web.write_run_tag(self.run_dir, "textbook")
        self.assertTrue(self.tag_file().is_file())

    def test_removing_the_run_removes_the_tag(self):
        # The cascade, same as the chat history: there is no separate cleanup
        # step because delete_run_everywhere's rmtree already is one.
        self.web.write_run_tag(self.run_dir, "textbook")
        shutil.rmtree(self.run_dir)
        self.assertFalse(self.tag_file().exists())

    def test_the_stored_file_is_versioned(self):
        self.web.write_run_tag(self.run_dir, "textbook")
        self.assertEqual(json.loads(self.tag_file().read_text())["version"], 1)

    def test_unreadable_storage_reads_as_untagged(self):
        self.tag_file().write_text("{ not json", encoding="utf-8")
        self.assertIsNone(self.web.read_run_tag(self.run_dir))

    def test_a_tag_of_the_wrong_type_reads_as_untagged(self):
        self.tag_file().write_text(json.dumps({"version": 1, "tag": 42}), encoding="utf-8")
        self.assertIsNone(self.web.read_run_tag(self.run_dir))


class ATagIsOneShortLabel(LibraryTagTest):
    def test_surrounding_space_is_dropped(self):
        self.assertEqual(self.web.normalize_tag("  textbook  "), "textbook")

    def test_inner_whitespace_is_collapsed(self):
        # It has to render on one line inside a chip.
        self.assertEqual(self.web.normalize_tag("Class   XI"), "Class XI")

    def test_newlines_cannot_break_the_chip(self):
        self.assertEqual(self.web.normalize_tag("two\nlines"), "two lines")

    def test_it_is_bounded(self):
        self.assertEqual(len(self.web.normalize_tag("x" * 500)), self.web.MAX_TAG_LENGTH)

    def test_an_empty_tag_is_no_tag(self):
        self.assertIsNone(self.web.normalize_tag("   "))
        self.assertIsNone(self.web.normalize_tag(None))


class ATagCanBeRemoved(LibraryTagTest):
    def test_writing_an_empty_tag_clears_it(self):
        self.web.write_run_tag(self.run_dir, "textbook")
        self.assertIsNone(self.web.write_run_tag(self.run_dir, ""))
        self.assertIsNone(self.web.read_run_tag(self.run_dir))

    def test_clearing_removes_the_file_rather_than_leaving_an_empty_one(self):
        self.web.write_run_tag(self.run_dir, "textbook")
        self.web.write_run_tag(self.run_dir, "")
        self.assertFalse(self.tag_file().exists())

    def test_clearing_an_already_untagged_run_is_not_an_error(self):
        self.assertIsNone(self.web.write_run_tag(self.run_dir, ""))

    def test_a_tag_can_be_replaced(self):
        self.web.write_run_tag(self.run_dir, "first")
        self.web.write_run_tag(self.run_dir, "second")
        self.assertEqual(self.web.read_run_tag(self.run_dir), "second")


class TheLibraryReportsTheTag(LibraryTagTest):
    def test_the_summary_carries_it(self):
        self.web.write_run_tag(self.run_dir, "Class XI")
        summary = self.web.run_summary(self.run_dir)
        self.assertEqual(summary["tag"], "Class XI")

    def test_the_summary_reports_none_when_untagged(self):
        # The field is always present so the card renderer needs no special case.
        self.assertIn("tag", self.web.run_summary(self.run_dir))
        self.assertIsNone(self.web.run_summary(self.run_dir)["tag"])

    def test_tagging_does_not_disturb_the_document_name(self):
        before = self.web.run_summary(self.run_dir)["name"]
        self.web.write_run_tag(self.run_dir, "textbook")
        self.assertEqual(self.web.run_summary(self.run_dir)["name"], before)


class TheTagSitsBeforeTheStatus(unittest.TestCase):
    """Ordering the request asked for: tag, then the succeeded flag."""

    def test_the_card_renders_the_tag_before_the_status_pill(self):
        app = (REPO / "Scripts" / "webui" / "app.js").read_text(encoding="utf-8")
        tag_at = app.index("runTagHtml(run.tag)")
        pill_at = app.index('<span class="pill ${escapeHtml(status)}"')
        self.assertLess(tag_at, pill_at)

    def test_the_name_takes_the_slack_so_the_row_ends_at_the_right(self):
        css = (REPO / "Scripts" / "webui" / "styles.css").read_text(encoding="utf-8")
        rule = css[css.index(".runName {") : css.index("}", css.index(".runName {"))]
        self.assertIn("flex: 1", rule)


if __name__ == "__main__":
    unittest.main()
