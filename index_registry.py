import json
from pathlib import Path
from typing import Any


def get_index_dirs(index_root: Path) -> list[Path]:
    if not index_root.exists():
        return []
    return [path for path in sorted(index_root.iterdir()) if path.is_dir()]


def load_index_manifest(index_dir: Path) -> dict[str, Any] | None:
    manifest_path = index_dir / "index_manifest.json"
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["index_dir"] = str(index_dir)
    return manifest


def load_all_manifests(index_root: Path) -> list[dict[str, Any]]:
    manifests = []
    for index_dir in get_index_dirs(index_root):
        manifest = load_index_manifest(index_dir)
        if manifest is not None:
            manifests.append(manifest)
    return manifests


def find_manifest_by_name(index_root: Path, index_name: str) -> dict[str, Any] | None:
    index_dir = index_root / index_name
    return load_index_manifest(index_dir)
