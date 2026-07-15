"""Request-scoped logging without mutating the process root logger.

The legacy CLI repeatedly cleared the root logger while moving between
documents.  That is unsafe when DocARAG is embedded in a web service and it
also lets concurrent document runs write into each other's files.  This
module installs handlers only on explicitly named DocARAG logger namespaces
and filters each handler with a context-local run identifier.
"""

from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass
import hashlib
import logging
from pathlib import Path
import os
import re
import threading
import time
import uuid
from typing import Iterable


_CURRENT_RUN_ID: ContextVar[str | None] = ContextVar(
    "docarag_logging_run_id",
    default=None,
)
_LOGGER_STATE_LOCK = threading.Lock()
_LOGGER_STATE: dict[str, tuple[int, int]] = {}
_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}")


def current_logging_run_id() -> str | None:
    """Return the active logging run ID, if the caller is in a run scope."""

    return _CURRENT_RUN_ID.get()


class _RunFilter(logging.Filter):
    def __init__(self, run_id: str) -> None:
        super().__init__()
        self._run_id = run_id

    def filter(self, record: logging.LogRecord) -> bool:
        if _CURRENT_RUN_ID.get() != self._run_id:
            return False
        record.docarag_run_id = self._run_id
        return True


def _acquire_logger(name: str, level: int) -> logging.Logger:
    logger = logging.getLogger(name)
    with _LOGGER_STATE_LOCK:
        state = _LOGGER_STATE.get(name)
        if state is None:
            _LOGGER_STATE[name] = (1, logger.level)
            logger.setLevel(level)
        else:
            count, original_level = state
            _LOGGER_STATE[name] = (count + 1, original_level)
            if logger.level == logging.NOTSET or logger.level > level:
                logger.setLevel(level)
    return logger


def _release_logger(name: str) -> None:
    logger = logging.getLogger(name)
    with _LOGGER_STATE_LOCK:
        count, original_level = _LOGGER_STATE[name]
        if count == 1:
            logger.setLevel(original_level)
            del _LOGGER_STATE[name]
        else:
            _LOGGER_STATE[name] = (count - 1, original_level)


@dataclass(frozen=True)
class RunLogInfo:
    """Stable metadata exposed by an entered :class:`RunLogScope`."""

    run_id: str
    log_file: Path


class RunLogScope:
    """Isolate one run's file/console records using a context variable.

    The same scope object is single-use.  Different scope objects may overlap
    across threads or asyncio tasks; filters ensure each record reaches only
    the handler belonging to its active context.  Executor work must propagate
    context variables (DocARAG's bounded executor does so).
    """

    def __init__(
        self,
        save_path: str | Path,
        *,
        logger_names: Iterable[str] = ("Core", "docarag"),
        level: int = logging.INFO,
        console_handler: logging.Handler | None = None,
        run_id: str | None = None,
    ) -> None:
        requested_names = tuple(
            dict.fromkeys(str(name) for name in logger_names if name)
        )
        # A record emitted by ``Core.inference`` propagates to ``Core``.  If a
        # handler is attached to both namespaces, the same event is written
        # twice.  Keep the shallowest requested namespace and discard only its
        # descendants; unrelated namespaces (for example ``Core`` and
        # ``docarag``) remain independent.
        names = tuple(
            name
            for name in requested_names
            if not any(
                name.startswith(f"{candidate}.")
                for candidate in requested_names
                if candidate != name
            )
        )
        if not names:
            raise ValueError("logger_names must contain at least one logger")
        self._save_path = Path(save_path)
        self._logger_names = names
        self._level = level
        self._console_handler = console_handler
        if run_id is not None and (
            not isinstance(run_id, str) or _RUN_ID.fullmatch(run_id) is None
        ):
            raise ValueError(
                "run_id must match [A-Za-z0-9][A-Za-z0-9_.:-]{0,127}"
            )
        self._requested_run_id = run_id
        self._token: Token[str | None] | None = None
        self._handlers: list[logging.Handler] = []
        self._loggers: list[logging.Logger] = []
        self._info: RunLogInfo | None = None
        self._used = False

    @property
    def info(self) -> RunLogInfo:
        if self._info is None:
            raise RuntimeError("RunLogScope has not been entered")
        return self._info

    def __enter__(self) -> RunLogInfo:
        if self._used:
            raise RuntimeError("RunLogScope objects are single-use")
        self._used = True

        run_id = self._requested_run_id or uuid.uuid4().hex
        filename_id = hashlib.sha256(run_id.encode("utf-8")).hexdigest()[:8]
        log_dir = self._save_path / "logs"
        log_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        log_dir.chmod(0o700)
        log_file = log_dir / (
            f"run_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}_{filename_id}.log"
        )

        file_handler = logging.FileHandler(log_file, mode="x", encoding="utf-8")
        log_file.chmod(0o600)
        file_handler.setLevel(self._level)
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(docarag_run_id)s - %(name)s - "
                "%(levelname)s - %(message)s",
                datefmt="[%X]",
            )
        )
        self._handlers = [file_handler]
        if self._console_handler is not None:
            self._console_handler.setLevel(self._level)
            self._handlers.append(self._console_handler)

        run_filter = _RunFilter(run_id)
        for handler in self._handlers:
            handler.addFilter(run_filter)

        try:
            self._token = _CURRENT_RUN_ID.set(run_id)
            for name in self._logger_names:
                logger = _acquire_logger(name, self._level)
                self._loggers.append(logger)
                for handler in self._handlers:
                    logger.addHandler(handler)
            self._info = RunLogInfo(run_id=run_id, log_file=log_file)
            return self._info
        except BaseException:
            self._cleanup()
            raise

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._cleanup()

    def _cleanup(self) -> None:
        for logger in reversed(self._loggers):
            for handler in self._handlers:
                logger.removeHandler(handler)
            _release_logger(logger.name)
        self._loggers.clear()

        if self._token is not None:
            _CURRENT_RUN_ID.reset(self._token)
            self._token = None

        for handler in self._handlers:
            try:
                handler.close()
            except Exception:
                pass
        self._handlers.clear()
