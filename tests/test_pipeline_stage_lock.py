"""A stage's advisory lock serializes concurrent runs of the same stage."""

from __future__ import annotations

import unittest

from docarag.pipeline import StageLockedError, _lock_key, _stage_lock


class _FakeConnection:
    """Two independent sessions share ``locks`` the way real advisory locks
    are visible cluster-wide: whichever session asks first gets it."""

    def __init__(self, locks: dict[tuple[int, int], str], owner: str):
        self.locks = locks
        self.owner = owner
        self.calls: list[str] = []

    def execute(self, sql, params):
        key = tuple(params)
        if sql.startswith("SELECT pg_try_advisory_lock"):
            self.calls.append("lock")
            held_by = self.locks.get(key)
            acquired = held_by is None or held_by == self.owner
            if acquired:
                self.locks[key] = self.owner
            return _Row(acquired)
        if sql.startswith("SELECT pg_advisory_unlock"):
            self.calls.append("unlock")
            if self.locks.get(key) == self.owner:
                del self.locks[key]
            return _Row(True)
        raise AssertionError(f"unexpected SQL: {sql}")


class _Row:
    def __init__(self, value):
        self._value = value

    def fetchone(self):
        return {
            "pg_try_advisory_lock": self._value,
            "pg_advisory_unlock": self._value,
        }


class LockKeyTests(unittest.TestCase):
    def test_key_is_deterministic_and_scoped_to_revision_and_stage(self):
        self.assertEqual(_lock_key(7, "tree"), _lock_key(7, "tree"))
        self.assertNotEqual(_lock_key(7, "tree"), _lock_key(7, "graph"))
        self.assertNotEqual(_lock_key(7, "tree"), _lock_key(8, "tree"))

    def test_key_components_are_valid_postgres_int4(self):
        key1, key2 = _lock_key(123, "vdb")
        for component in (key1, key2):
            self.assertIsInstance(component, int)
            self.assertTrue(-(2**31) <= component < 2**31)


class StageLockTests(unittest.TestCase):
    def test_a_second_session_is_refused_while_the_first_holds_the_lock(self):
        shared: dict = {}
        first = _FakeConnection(shared, owner="session-a")
        second = _FakeConnection(shared, owner="session-b")

        with _stage_lock(first, revision_key=1, stage="tree"):
            with self.assertRaises(StageLockedError):
                with _stage_lock(second, revision_key=1, stage="tree"):
                    pass  # pragma: no cover - must not be entered

    def test_the_lock_releases_on_a_clean_exit(self):
        shared: dict = {}
        connection = _FakeConnection(shared, owner="session-a")

        with _stage_lock(connection, revision_key=1, stage="tree"):
            pass

        self.assertEqual({}, shared)

    def test_the_lock_releases_even_when_the_body_raises(self):
        shared: dict = {}
        connection = _FakeConnection(shared, owner="session-a")

        with self.assertRaises(ValueError):
            with _stage_lock(connection, revision_key=1, stage="tree"):
                raise ValueError("stage failed")

        self.assertEqual({}, shared)

    def test_different_stages_of_the_same_revision_do_not_contend(self):
        shared: dict = {}
        tree_session = _FakeConnection(shared, owner="session-a")
        graph_session = _FakeConnection(shared, owner="session-b")

        with _stage_lock(tree_session, revision_key=1, stage="tree"):
            with _stage_lock(graph_session, revision_key=1, stage="graph"):
                pass  # both held concurrently without contention

    def test_the_same_revision_and_stage_from_a_different_revision_do_not_contend(self):
        shared: dict = {}
        first = _FakeConnection(shared, owner="session-a")
        second = _FakeConnection(shared, owner="session-b")

        with _stage_lock(first, revision_key=1, stage="tree"):
            with _stage_lock(second, revision_key=2, stage="tree"):
                pass  # different revisions, no contention

    def test_a_locked_error_names_the_stage(self):
        shared: dict = {}
        first = _FakeConnection(shared, owner="session-a")
        second = _FakeConnection(shared, owner="session-b")

        with _stage_lock(first, revision_key=1, stage="graph"):
            with self.assertRaisesRegex(StageLockedError, "graph"):
                with _stage_lock(second, revision_key=1, stage="graph"):
                    pass  # pragma: no cover


if __name__ == "__main__":
    unittest.main()
