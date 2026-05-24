from __future__ import annotations

import json
import platform
import sys
from datetime import datetime
from pathlib import Path


def _package_version(name: str) -> str | None:
    try:
        from importlib import metadata

        return metadata.version(name)
    except Exception:
        return None


def create_run_manifest(
    *,
    output_dir: str,
    project_name: str,
    backend: str,
    command: list[str],
    dataset: dict,
    models: dict,
    configs: dict,
    training: dict,
    tensorboard: dict,
    preflight: list[dict],
) -> str:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    path = Path(output_dir) / f"{project_name}_{backend}_{timestamp}_run_manifest.json"
    payload = {
        "schema_version": 1,
        "project_name": project_name,
        "backend": backend,
        "created_at": datetime.now().isoformat(),
        "status": "running",
        "command": command,
        "dataset": dataset,
        "models": models,
        "configs": configs,
        "training": training,
        "tensorboard": tensorboard,
        "preflight": preflight,
        "environment": {
            "python": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "packages": {
                "torch": _package_version("torch"),
                "accelerate": _package_version("accelerate"),
                "diffsynth": _package_version("diffsynth"),
                "peft": _package_version("peft"),
                "torchao": _package_version("torchao"),
                "gradio": _package_version("gradio"),
            },
        },
    }
    write_manifest(path, payload)
    return str(path)


def write_manifest(path: str | Path, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def update_run_manifest(path: str | Path, **updates) -> None:
    path = Path(path)
    if not path.exists():
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        payload = {}
    payload.update(updates)
    payload["updated_at"] = datetime.now().isoformat()
    write_manifest(path, payload)


def list_output_files(output_dir: str) -> list[str]:
    p = Path(output_dir)
    if not p.exists():
        return []
    return [
        str(item)
        for item in sorted(p.iterdir())
        if item.is_file() and item.suffix.lower() in {".safetensors", ".pt", ".pth", ".ckpt", ".json"}
    ]

