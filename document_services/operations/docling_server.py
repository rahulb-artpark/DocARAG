"""Loopback-only Docling conversion service for the isolated Docling venv.

This implements only the bounded async JSON flow consumed by
``DoclingServiceExportLoader``.  It is intentionally not a general replacement
for Docling Serve.  The process imports Docling only inside the Docling
environment and keeps the controller/vendor dependency boundary intact.
"""

from __future__ import annotations

import argparse
import base64
import binascii
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import hashlib
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
from threading import Lock
import tempfile
import time
from typing import Any
import uuid


MAX_REQUEST_BYTES = 48 * 1024 * 1024
MAX_SOURCE_BYTES = 32 * 1024 * 1024
MAX_JOBS = 32
JOB_TTL_SECONDS = 3600
_TASK = re.compile(r"[0-9a-f]{32}")


@dataclass
class Job:
    task_id: str
    status: str = "pending"
    result: dict[str, object] | None = None
    error_code: str | None = None
    updated: float = field(default_factory=time.monotonic)


class JobStore:
    def __init__(self, *, maximum: int = MAX_JOBS) -> None:
        self.maximum = maximum
        self.lock = Lock()
        self.jobs: dict[str, Job] = {}

    def create(self) -> Job:
        with self.lock:
            self._reap()
            if len(self.jobs) >= self.maximum:
                raise RuntimeError("job_capacity_exhausted")
            job = Job(uuid.uuid4().hex)
            self.jobs[job.task_id] = job
            return job

    def update(self, task_id: str, **values: object) -> None:
        with self.lock:
            job = self.jobs[task_id]
            for name, value in values.items():
                setattr(job, name, value)
            job.updated = time.monotonic()

    def get(self, task_id: str) -> Job | None:
        with self.lock:
            self._reap()
            return self.jobs.get(task_id)

    def _reap(self) -> None:
        cutoff = time.monotonic() - JOB_TTL_SECONDS
        self.jobs = {key: value for key, value in self.jobs.items() if value.updated >= cutoff}


class DoclingCompatApplication:
    def __init__(self, *, maximum_jobs: int = MAX_JOBS) -> None:
        self.jobs = JobStore(maximum=maximum_jobs)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="docling-convert")

    def submit(self, payload: dict[str, object]) -> Job:
        source, filename, options = _validated_submission(payload)
        job = self.jobs.create()
        self.executor.submit(self._convert, job.task_id, source, filename, options)
        return job

    def _convert(
        self, task_id: str, source: bytes, filename: str, options: dict[str, object]
    ) -> None:
        self.jobs.update(task_id, status="started")
        started = time.monotonic()
        path: Path | None = None
        try:
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            from docling.document_converter import DocumentConverter, PdfFormatOption

            pipeline = PdfPipelineOptions()
            pipeline.do_ocr = bool(options.get("do_ocr", True))
            pipeline.do_table_structure = bool(options.get("do_table_structure", True))
            pipeline.do_formula_enrichment = bool(options.get("do_formula_enrichment", False))
            pipeline.do_picture_classification = bool(options.get("do_picture_classification", False))
            pipeline.do_code_enrichment = bool(options.get("do_code_enrichment", False))
            pipeline.generate_picture_images = bool(options.get("include_images", False))
            pipeline.generate_page_images = False
            converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline)
                }
            )
            descriptor, raw_path = tempfile.mkstemp(prefix="docarag-docling-", suffix=".pdf")
            path = Path(raw_path)
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(source)
                handle.flush()
                os.fsync(handle.fileno())
            page_range = options["page_range"]
            converted = converter.convert(
                path,
                page_range=(int(page_range[0]), int(page_range[1])),
                max_file_size=MAX_SOURCE_BYTES,
            )
            exported = converted.document.export_to_dict()
            # The temporary source name is an implementation detail.  Bind the
            # strict response to the client-submitted safe filename.
            origin = dict(exported.get("origin", {}))
            origin["filename"] = filename
            exported["origin"] = origin
            result: dict[str, object] = {
                "document": {
                    "filename": filename,
                    "md_content": None,
                    "json_content": exported,
                    "html_content": None,
                    "text_content": None,
                    "doctags_content": None,
                    "doclang_content": None,
                },
                "status": "success",
                "processing_time": time.monotonic() - started,
                "timings": {},
                "errors": [],
            }
            self.jobs.update(task_id, status="success", result=result)
        except BaseException as exc:
            code = hashlib.sha256(
                f"{type(exc).__name__}:{exc}".encode("utf-8", errors="replace")
            ).hexdigest()[:16]
            self.jobs.update(task_id, status="failure", error_code=code)
        finally:
            if path is not None:
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass

    def close(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=True)


def _validated_submission(
    payload: dict[str, object],
) -> tuple[bytes, str, dict[str, object]]:
    if set(payload) != {"file_sources", "options", "target"}:
        raise ValueError("invalid_submission_shape")
    sources = payload.get("file_sources")
    options = payload.get("options")
    target = payload.get("target")
    if not isinstance(sources, list) or len(sources) != 1 or not isinstance(sources[0], dict):
        raise ValueError("one_file_source_required")
    if not isinstance(options, dict) or not isinstance(target, dict) or target.get("kind") != "inbody":
        raise ValueError("invalid_options_or_target")
    source = sources[0]
    if set(source) != {"base64_string", "filename"}:
        raise ValueError("invalid_file_source")
    filename = source.get("filename")
    encoded = source.get("base64_string")
    if (
        not isinstance(filename, str)
        or not filename.lower().endswith(".pdf")
        or len(filename.encode("utf-8")) > 255
        or filename in {".", ".."}
        or "/" in filename
        or "\\" in filename
    ):
        raise ValueError("invalid_filename")
    if not isinstance(encoded, str) or len(encoded) > (MAX_SOURCE_BYTES * 4 // 3 + 8):
        raise ValueError("invalid_source_encoding")
    try:
        data = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("invalid_source_encoding") from exc
    if not 1 <= len(data) <= MAX_SOURCE_BYTES or not data.startswith(b"%PDF-"):
        raise ValueError("invalid_pdf_source")
    page_range = options.get("page_range")
    if (
        not isinstance(page_range, list)
        or len(page_range) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) for value in page_range)
        or not 1 <= page_range[0] <= page_range[1]
        or page_range[1] - page_range[0] + 1 > 64
    ):
        raise ValueError("invalid_page_range")
    return data, filename, options


def _task_payload(job: Job) -> dict[str, object]:
    return {
        "task_id": job.task_id,
        "task_type": "convert",
        "task_status": job.status,
        "task_position": None,
        "task_meta": None,
    }


class Handler(BaseHTTPRequestHandler):
    server_version = "DocARAG-Docling-Compat/1.0"

    @property
    def app(self) -> DoclingCompatApplication:
        return self.server.app  # type: ignore[attr-defined]

    def do_GET(self) -> None:
        if self.path == "/health":
            self._json(HTTPStatus.OK, {"status": "ok", "schema_version": "docarag.docling-compat.v1"})
            return
        for prefix, result in (
            ("/v1/status/poll/", False),
            ("/v1/result/", True),
        ):
            if self.path.startswith(prefix):
                task_id = self.path[len(prefix) :]
                if _TASK.fullmatch(task_id) is None:
                    self._json(HTTPStatus.NOT_FOUND, {"error": "not_found"}); return
                job = self.app.jobs.get(task_id)
                if job is None:
                    self._json(HTTPStatus.NOT_FOUND, {"error": "not_found"}); return
                if result:
                    if job.status != "success" or job.result is None:
                        self._json(HTTPStatus.CONFLICT, {"error": job.status}); return
                    self._json(HTTPStatus.OK, job.result); return
                self._json(HTTPStatus.OK, _task_payload(job)); return
        self._json(HTTPStatus.NOT_FOUND, {"error": "not_found"})

    def do_POST(self) -> None:
        if self.path != "/v1/convert/source/async":
            self._json(HTTPStatus.NOT_FOUND, {"error": "not_found"}); return
        content_type = self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
        try:
            length = int(self.headers.get("Content-Length", "-1"))
        except ValueError:
            length = -1
        if content_type != "application/json" or not 1 <= length <= MAX_REQUEST_BYTES:
            self._json(HTTPStatus.BAD_REQUEST, {"error": "invalid_request"}); return
        try:
            raw = self.rfile.read(length + 1)
            if len(raw) != length:
                raise ValueError("request_length_mismatch")
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                raise ValueError("request_not_object")
            job = self.app.submit(payload)
        except (ValueError, json.JSONDecodeError):
            self._json(HTTPStatus.BAD_REQUEST, {"error": "invalid_request"}); return
        except RuntimeError:
            self._json(HTTPStatus.TOO_MANY_REQUESTS, {"error": "capacity_exhausted"}); return
        self._json(HTTPStatus.OK, _task_payload(job))

    def _json(self, status: HTTPStatus, payload: object) -> None:
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        self.send_response(status.value)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, format: str, *args: object) -> None:
        return


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the bounded local Docling compatibility service")
    parser.add_argument("--host", default="127.0.0.1", choices=("127.0.0.1", "localhost"))
    parser.add_argument("--port", type=int, default=5001)
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    app = DoclingCompatApplication()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.app = app  # type: ignore[attr-defined]
    try:
        server.serve_forever(poll_interval=0.1)
    finally:
        server.server_close()
        app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
