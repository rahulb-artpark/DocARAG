"""Publish canonical image crops into a content-addressed store at index time.

Parser adapters emit visual crops as mutable local files with no digest. The
BookRAG projection refuses to treat such an asset as evidence unless it is a
verified ``cas://sha256`` asset supplied through a materialization lease. This
publishes each crop into a content-addressed store and rebuilds the block with
the verified CAS asset, so figures/diagrams (and table/formula crops) become
verified visual evidence that also hydrates into answers from the same store.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from urllib.parse import unquote, urlparse

from docarag.artifact_store import LocalContentAddressedArtifactStore
from docarag.domain.evidence import AssetKind, AssetRef, CanonicalBlock

_IMAGE_KINDS = {
    AssetKind.REGION_CROP,
    AssetKind.EXTRACTED_IMAGE,
    AssetKind.PAGE_RENDER,
}
_MIME = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
}


def _looks_image(asset: AssetRef) -> bool:
    if asset.media_type and asset.media_type.lower().startswith("image/"):
        return True
    return asset.kind in _IMAGE_KINDS


def _resolve_local(asset: AssetRef, run: Path) -> Path | None:
    uri = asset.uri
    if not isinstance(uri, str) or not uri or uri.startswith("cas://"):
        return None
    if uri.startswith("file://"):
        uri = unquote(urlparse(uri).path)
    path = Path(uri)
    candidates = [path if path.is_absolute() else run / path]
    if not path.is_absolute() and len(path.parts) == 1:
        candidates += [run / "vlm" / "images" / path.name, run / "auto" / "images" / path.name]
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
            resolved.relative_to(run)
        except (OSError, RuntimeError, ValueError):
            continue
        if resolved.is_file():
            return resolved
    return None


def materialize_visual_crops(
    blocks: tuple[CanonicalBlock, ...] | list[CanonicalBlock],
    store: LocalContentAddressedArtifactStore,
    run_dir: str | Path,
) -> tuple[tuple[CanonicalBlock, ...], tuple[AssetRef, ...]]:
    """Return (blocks with verified CAS crop assets, unique CAS assets)."""

    run = Path(run_dir).expanduser().resolve()
    new_blocks: list[CanonicalBlock] = []
    cas_assets: dict[str, AssetRef] = {}
    for block in blocks:
        new_assets: list[AssetRef] = []
        changed = False
        for asset in block.assets:
            local = _resolve_local(asset, run) if _looks_image(asset) else None
            if local is None:
                new_assets.append(asset)
                continue
            data = local.read_bytes()
            media_type = asset.media_type or _MIME.get(local.suffix.lower(), "image/jpeg")
            stored = store.publish(data, media_type=media_type)
            cas_asset = stored.to_asset_ref(
                kind=asset.kind if asset.kind in _IMAGE_KINDS else AssetKind.EXTRACTED_IMAGE,
                metadata=asset.metadata,
            )
            new_assets.append(cas_asset)
            cas_assets[cas_asset.asset_id] = cas_asset
            changed = True
        new_blocks.append(replace(block, assets=tuple(new_assets)) if changed else block)
    return tuple(new_blocks), tuple(cas_assets.values())


__all__ = ["materialize_visual_crops"]
