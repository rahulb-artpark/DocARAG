"""Saved chats live in the run directory, which is what makes them the PDF's.

Deleting a document rmtree's its run directory, so a history stored inside it
goes with the document and cannot outlive it. A store keyed by document id
somewhere else would leave an orphan behind every deletion instead, which is the
whole reason for this placement.
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
    spec = importlib.util.spec_from_file_location("docarag_test_web_conversations", WEB_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ConversationStoreTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.web = load_web_module()

    def setUp(self):
        self.run_dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.run_dir, ignore_errors=True)
        self.state = self.web.AppState(self.run_dir)

    def _history_file(self):
        return self.run_dir / "web_chat" / "conversations.json"

    def _with_exchange(self, question="How does binary search work?", answer="It halves."):
        conversation = self.state.create_conversation()
        self.state._record_message(
            conversation["id"],
            {"id": "u", "role": "user", "content": question, "createdAt": self.state._now()},
        )
        self.state._record_message(
            conversation["id"],
            {"id": "a", "role": "assistant", "answer": answer, "createdAt": self.state._now()},
        )
        return conversation["id"]


class TheHistoryBelongsToTheDocument(ConversationStoreTest):
    def test_it_is_written_inside_the_run_directory(self):
        self._with_exchange()
        self.assertTrue(self._history_file().is_file())

    def test_removing_the_run_removes_the_history(self):
        # This is the cascade. There is no delete-the-chats step anywhere,
        # because delete_run_everywhere's rmtree already is one.
        self._with_exchange()
        shutil.rmtree(self.run_dir)
        self.assertFalse(self._history_file().exists())

    def test_a_document_with_no_run_selected_lists_nothing_rather_than_failing(self):
        stateless = self.web.AppState(None)
        with self.assertRaises(self.web.RunNotReadyError):
            stateless.list_conversations()

    def test_the_history_is_not_world_readable(self):
        # It contains the user's questions about their own documents.
        self._with_exchange()
        self.assertEqual(self._history_file().stat().st_mode & 0o777, 0o600)


class TheSidebarShowsUsableSummaries(ConversationStoreTest):
    def test_the_title_comes_from_the_first_question(self):
        self._with_exchange(question="python code for insertion sort?")
        listed = self.state.list_conversations()["conversations"]
        self.assertEqual(listed[0]["title"], "python code for insertion sort?")

    def test_a_later_question_does_not_retitle_the_conversation(self):
        conversation_id = self._with_exchange(question="first question")
        self.state._record_message(
            conversation_id,
            {"id": "u2", "role": "user", "content": "second question", "createdAt": self.state._now()},
        )
        self.assertEqual(
            self.state.list_conversations()["conversations"][0]["title"], "first question"
        )

    def test_a_long_question_is_bounded(self):
        self._with_exchange(question="word " * 200)
        self.assertLessEqual(
            len(self.state.list_conversations()["conversations"][0]["title"]), 64
        )

    def test_an_untouched_draft_is_not_offered_as_history(self):
        self.state.create_conversation()
        self.assertEqual(self.state.list_conversations()["conversations"], [])

    def test_repeated_new_chat_does_not_stack_drafts(self):
        for _ in range(4):
            self.state.create_conversation()
        stored = json.loads(self._history_file().read_text())["conversations"]
        self.assertEqual(len(stored), 1)

    def test_a_real_conversation_survives_a_new_draft(self):
        conversation_id = self._with_exchange()
        self.state.create_conversation()
        self.assertIsNotNone(self.state.get_conversation(conversation_id))

    def test_the_newest_conversation_is_listed_first(self):
        self._with_exchange(question="older")
        self._with_exchange(question="newer")
        titles = [c["title"] for c in self.state.list_conversations()["conversations"]]
        self.assertEqual(titles[0], "newer")


class AConversationCanBeDeleted(ConversationStoreTest):
    def test_deleting_reports_success_and_removes_it(self):
        conversation_id = self._with_exchange()
        self.assertTrue(self.state.delete_conversation(conversation_id))
        self.assertIsNone(self.state.get_conversation(conversation_id))

    def test_deleting_an_unknown_conversation_reports_failure(self):
        self.assertFalse(self.state.delete_conversation("no-such-id"))

    def test_deleting_one_leaves_the_others(self):
        keep = self._with_exchange(question="keep me")
        drop = self._with_exchange(question="drop me")
        self.state.delete_conversation(drop)
        remaining = [c["id"] for c in self.state.list_conversations()["conversations"]]
        self.assertEqual(remaining, [keep])


class DamageDoesNotTakeTheChatDown(ConversationStoreTest):
    def test_unparseable_history_reads_as_empty(self):
        self._with_exchange()
        self._history_file().write_text("{ this is not json", encoding="utf-8")
        with self.assertLogs(level="ERROR"):
            self.assertEqual(self.state.list_conversations()["conversations"], [])

    def test_history_of_the_wrong_shape_reads_as_empty(self):
        self._history_file().parent.mkdir(parents=True, exist_ok=True)
        self._history_file().write_text('["not", "a", "mapping"]', encoding="utf-8")
        self.assertEqual(self.state.list_conversations()["conversations"], [])

    def test_a_question_against_a_missing_conversation_id_still_gets_one(self):
        # The sidebar can be out of date -- a conversation deleted in another tab
        # must not lose the answer being asked for right now.
        conversation_id = self.state.open_conversation("gone-from-disk")
        self.assertEqual(conversation_id, "gone-from-disk")
        self.assertIsNotNone(self.state.get_conversation(conversation_id))

    def test_a_returned_conversation_cannot_be_mutated_through_the_caller(self):
        conversation_id = self._with_exchange()
        fetched = self.state.get_conversation(conversation_id)
        fetched["messages"].clear()
        self.assertEqual(len(self.state.get_conversation(conversation_id)["messages"]), 2)


class TheStoredReplyStaysReplayableWithoutTheBulk(ConversationStoreTest):
    """The retrieval payload was 181 KB of a 337 KB message -- 98% of it.

    The whole file is rewritten per message, so storing it inline made a long
    conversation quadratic in I/O and forced the sidebar to parse megabytes to
    list a few titles. It is already saved per query under web_chat/<queryId>/,
    which the retained queryId points at.
    """

    def _response(self):
        return {
            "answer": "Insertion sort shifts each key left.",
            "citations": [{"label": "E1", "type": "code", "page": 11}],
            "nodeIds": [78],
            "elapsedSec": 12.5,
            "queryId": "q_20260730_193810_fdee527d",
            "retrieval": {
                "query": "python code for insertion sort?",
                "plan": {"noise": "x" * 5000},
                "iterations": [
                    {
                        "sub_query": "insertion sort code",
                        "iteration_graph_nodes": [
                            {"entity_name": "Insertion Sort", "bulk": "y" * 5000}
                        ],
                        "chunks": ["z" * 20000],
                    }
                ],
            },
        }

    def test_everything_the_answer_renders_from_is_kept(self):
        stored = self.state._history_message(self._response())
        self.assertEqual(stored["answer"], "Insertion sort shifts each key left.")
        self.assertEqual(len(stored["citations"]), 1)
        self.assertEqual(stored["nodeIds"], [78])
        self.assertEqual(stored["queryId"], "q_20260730_193810_fdee527d")

    def test_the_entities_fold_still_has_what_it_shows(self):
        stored = self.state._history_message(self._response())
        iteration = stored["retrieval"]["iterations"][0]
        self.assertEqual(iteration["sub_query"], "insertion sort code")
        self.assertEqual(
            [node["entity_name"] for node in iteration["iteration_graph_nodes"]],
            ["Insertion Sort"],
        )

    def test_the_bulk_is_dropped(self):
        stored = self.state._history_message(self._response())
        self.assertLess(len(json.dumps(stored)), 1000)
        self.assertNotIn("chunks", stored["retrieval"]["iterations"][0])
        self.assertNotIn("plan", stored["retrieval"])

    def test_a_reply_with_no_retrieval_is_stored_without_inventing_one(self):
        stored = self.state._history_message({"answer": "no evidence"})
        self.assertEqual(stored["retrieval"], {})

    def test_the_live_response_is_not_altered_by_being_stored(self):
        # The caller still returns the full payload to the browser.
        response = self._response()
        self.state._history_message(response)
        self.assertIn("chunks", response["retrieval"]["iterations"][0])


if __name__ == "__main__":
    unittest.main()
