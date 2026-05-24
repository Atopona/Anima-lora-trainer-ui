from __future__ import annotations

import json
from pathlib import Path


OUTPUT_SUFFIXES = {".safetensors", ".pt", ".pth", ".ckpt"}


def read_json(path: Path) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def scan_run_manifests(*roots: str | Path) -> list[dict]:
    manifests: list[dict] = []
    seen: set[Path] = set()
    for root in roots:
        if not root:
            continue
        path = Path(root)
        if not path.exists():
            continue
        candidates = path.rglob("*_run_manifest.json") if path.is_dir() else [path]
        for manifest_path in candidates:
            resolved = manifest_path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            payload = read_json(manifest_path)
            if not payload:
                continue
            stat = manifest_path.stat()
            manifests.append({
                "manifest": str(manifest_path),
                "project": payload.get("project_name", ""),
                "backend": payload.get("backend", ""),
                "status": payload.get("status", ""),
                "created_at": payload.get("created_at", ""),
                "updated_at": payload.get("updated_at", ""),
                "output_dir": str(Path(manifest_path).parent),
                "log_file": payload.get("log_file", ""),
                "latest_output": latest_output_file(Path(manifest_path).parent),
                "size_mb": round(stat.st_size / (1024 * 1024), 3),
            })
    manifests.sort(key=lambda item: item.get("created_at") or "", reverse=True)
    return manifests


def latest_output_file(output_dir: str | Path) -> str:
    files = scan_output_files(output_dir)
    return files[0]["path"] if files else ""


def scan_output_files(output_dir: str | Path) -> list[dict]:
    root = Path(output_dir)
    if not root.exists() or not root.is_dir():
        return []
    rows = []
    for item in root.rglob("*"):
        if not item.is_file() or item.suffix.lower() not in OUTPUT_SUFFIXES:
            continue
        stat = item.stat()
        rows.append({
            "path": str(item),
            "name": item.name,
            "size_mb": round(stat.st_size / (1024 * 1024), 2),
            "modified": stat.st_mtime,
        })
    rows.sort(key=lambda item: item["modified"], reverse=True)
    return rows


def model_status_rows(model_paths: dict[str, str], urls: dict[str, str] | None = None) -> list[dict]:
    rows = []
    urls = urls or {}
    for label, value in model_paths.items():
        path = Path(value)
        exists = path.exists()
        size_gb = round(path.stat().st_size / (1024 ** 3), 2) if exists else 0.0
        rows.append({
            "model": label,
            "status": "ready" if exists else "missing",
            "path": str(path),
            "size_gb": size_gb,
            "url": urls.get(label, ""),
        })
    return rows
