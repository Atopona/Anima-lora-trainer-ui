from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from .diffsynth import TORCHAO_MIN_EXCLUSIVE_VERSION, is_version_at_most


@dataclass
class PreflightCheck:
    name: str
    status: str
    message: str

    def to_dict(self) -> dict:
        return asdict(self)


def _ok(name: str, message: str) -> PreflightCheck:
    return PreflightCheck(name, "ok", message)


def _warn(name: str, message: str) -> PreflightCheck:
    return PreflightCheck(name, "warn", message)


def _fail(name: str, message: str) -> PreflightCheck:
    return PreflightCheck(name, "fail", message)


def _package_version(name: str) -> str | None:
    try:
        from importlib import metadata

        return metadata.version(name)
    except Exception:
        return None


def run_preflight(
    *,
    backend: str,
    dataset_dir: str,
    output_dir: str,
    model_paths: dict[str, str],
    train_script: str,
    accelerate_cmd: list[str],
    diffsynth_dir: str = "",
    metadata_path: str = "",
    resume_lora_path: str = "",
    min_free_gb: float = 2.0,
) -> list[PreflightCheck]:
    checks: list[PreflightCheck] = []

    dataset = Path(dataset_dir)
    checks.append(_ok("dataset", f"Dataset directory exists: {dataset}") if dataset.is_dir() else _fail("dataset", f"Dataset directory not found: {dataset}"))

    output = Path(output_dir)
    try:
        output.mkdir(parents=True, exist_ok=True)
        usage = shutil.disk_usage(output)
        free_gb = usage.free / (1024 ** 3)
        if free_gb < min_free_gb:
            checks.append(_warn("disk", f"Only {free_gb:.1f} GB free at {output}"))
        else:
            checks.append(_ok("disk", f"{free_gb:.1f} GB free at {output}"))
    except Exception as exc:
        checks.append(_fail("disk", f"Cannot create or inspect output directory {output}: {exc}"))

    for label, value in model_paths.items():
        path = Path(value)
        if path.exists():
            size_gb = path.stat().st_size / (1024 ** 3)
            checks.append(_ok(f"model:{label}", f"{label} found ({size_gb:.2f} GB): {path}"))
        else:
            checks.append(_fail(f"model:{label}", f"{label} missing: {path}"))

    script = Path(train_script)
    checks.append(_ok("train_script", f"Training script found: {script}") if script.exists() else _fail("train_script", f"Training script missing: {script}"))

    launcher = Path(accelerate_cmd[0]) if accelerate_cmd else Path("")
    if accelerate_cmd and (launcher.exists() or accelerate_cmd[:3] == [sys.executable, "-m", "accelerate.commands.launch"]):
        checks.append(_ok("accelerate", "Accelerate launcher resolved."))
    else:
        checks.append(_warn("accelerate", f"Accelerate launcher may not exist: {' '.join(accelerate_cmd)}"))

    if resume_lora_path and str(resume_lora_path).strip():
        resume = Path(resume_lora_path)
        checks.append(_ok("resume", f"Resume LoRA/checkpoint found: {resume}") if resume.exists() else _fail("resume", f"Resume LoRA/checkpoint not found: {resume}"))

    if backend == "diffsynth":
        ds_dir = Path(diffsynth_dir)
        checks.append(_ok("diffsynth_dir", f"DiffSynth-Studio directory ready: {ds_dir}") if ds_dir.is_dir() else _fail("diffsynth_dir", f"DiffSynth-Studio directory missing: {ds_dir}"))
        if metadata_path:
            meta = Path(metadata_path)
            checks.append(_ok("metadata", f"Metadata CSV found: {meta}") if meta.exists() else _fail("metadata", f"Metadata CSV missing: {meta}"))
        torchao_version = _package_version("torchao")
        if torchao_version is None:
            checks.append(_warn("torchao", "torchao is not installed; PEFT may skip torchao dispatch."))
        elif is_version_at_most(torchao_version, TORCHAO_MIN_EXCLUSIVE_VERSION):
            checks.append(_fail("torchao", f"torchao {torchao_version} is incompatible; install torchao>0.16.0."))
        else:
            checks.append(_ok("torchao", f"torchao {torchao_version} is compatible."))
        result = subprocess.run(
            [sys.executable, "-c", "import diffsynth; print(diffsynth.__file__)"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            checks.append(_ok("diffsynth_import", f"diffsynth imports from {result.stdout.strip()}"))
        else:
            checks.append(_fail("diffsynth_import", result.stderr.strip() or "diffsynth import failed"))

    return checks


def has_failures(checks: list[PreflightCheck]) -> bool:
    return any(check.status == "fail" for check in checks)
