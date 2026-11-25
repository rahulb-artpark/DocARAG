"""Render .docx/.pptx to PDF via headless LibreOffice.

Word and PowerPoint files have no native page: a .docx is a reflowing text
stream with no fixed page boundaries in the file, and while a .pptx slide is
close to a page, neither carries the render pipeline's page-anchored evidence
model (page_index/page geometry, see domain/evidence.py). Converting to PDF
first gives every downstream stage -- parsing, page citations, crops, the
browser preview -- the same real, fixed page a native PDF upload has, at the
cost of the original file's native structure (headings, styles, ...).

The conversion is cached in a directory keyed by the source file's content
hash, so re-indexing the same upload never re-runs LibreOffice, while the PDF
inside it keeps the source's own name -- callers display it (run name,
citations) and it must read as "the document", not as a hash.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

OFFICE_SUFFIXES = (".docx", ".pptx")
CONVERT_TIMEOUT_SECONDS = 300
PDF_MAGIC = b"%PDF-"


class OfficeConversionError(RuntimeError):
    pass


def soffice_binary() -> str | None:
    """The LibreOffice CLI entry point, if installed."""

    return shutil.which("soffice") or shutil.which("libreoffice")


def convert_office_to_pdf(source: Path, *, cache_dir: Path, sha256: str) -> Path:
    """Render ``source`` (.docx/.pptx) to PDF, cached under ``cache_dir`` by ``sha256``.

    ``sha256`` is the caller's already-computed content hash of ``source`` --
    passed in rather than recomputed, since callers need it anyway for the
    document's identity. Raises OfficeConversionError if LibreOffice is
    missing or the conversion fails.
    """

    suffix = source.suffix.lower()
    if suffix not in OFFICE_SUFFIXES:
        raise OfficeConversionError(f"Not a supported Office document: {source.name}")

    # LibreOffice names its output after the input stem, so writing straight
    # into a hash-keyed directory keeps both properties at once: the cache is
    # addressed by content, and the PDF itself is still named like the source.
    out_dir = cache_dir / sha256
    cached = out_dir / f"{source.stem}.pdf"
    if cached.is_file() and cached.stat().st_size > 0:
        return cached

    binary = soffice_binary()
    if binary is None:
        raise OfficeConversionError(
            "LibreOffice is not installed, so this .docx/.pptx cannot be converted "
            "to PDF for indexing. Install it (e.g. `sudo apt install libreoffice`) "
            "and try again."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    # A fresh, isolated profile per conversion avoids the lock conflicts
    # headless LibreOffice hits when two conversions run at once against the
    # same default user profile. It is only ever read by this one soffice
    # call and never moved, so it can live in the system temp dir.
    with tempfile.TemporaryDirectory(prefix="docarag_office_profile_") as profile_tmp:
        env = os.environ.copy()
        env["SAL_USE_VCLPLUGIN"] = "svp"  # headless rendering backend, no display needed
        cmd = [
            binary,
            "--headless", "--norestore", "--nolockcheck", "--nodefault",
            f"-env:UserInstallation={Path(profile_tmp).as_uri()}",
            "--convert-to", "pdf", "--outdir", str(out_dir), str(source),
        ]
        try:
            result = subprocess.run(
                cmd, env=env, capture_output=True, text=True, timeout=CONVERT_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired as exc:
            raise OfficeConversionError(
                f"Converting {source.name} to PDF timed out after {CONVERT_TIMEOUT_SECONDS}s"
            ) from exc

    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()[:2000]
        raise OfficeConversionError(
            f"LibreOffice could not convert {source.name} to PDF"
            + (f": {detail}" if detail else "")
        )
    if not cached.is_file():
        raise OfficeConversionError(
            f"LibreOffice reported success but produced no PDF for {source.name}"
        )
    if cached.read_bytes()[: len(PDF_MAGIC)] != PDF_MAGIC:
        cached.unlink(missing_ok=True)
        raise OfficeConversionError(f"Conversion of {source.name} did not produce a valid PDF")

    return cached
