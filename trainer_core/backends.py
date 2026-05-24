from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class BackendRunSpec:
    backend: str
    command: list[str]
    cwd: str
    train_script: str
    metadata_path: str = ""
    configs: dict = field(default_factory=dict)
    train_entrypoint: str = ""
    train_args: list[str] = field(default_factory=list)


def build_kohya_run_spec(
    *,
    accelerate_launch: list[str],
    accelerate_config: str,
    threads: int,
    gpu_idx: str,
    train_script: Path,
    train_config: str,
    dataset_config: str,
    cwd: Path,
) -> BackendRunSpec:
    command = [
        *accelerate_launch,
        "--config_file", str(accelerate_config),
        "--num_cpu_threads_per_process", str(int(threads)),
        "--gpu_ids", str(gpu_idx),
        str(train_script),
        "--config_file", str(train_config),
        "--dataset_config", str(dataset_config),
    ]
    return BackendRunSpec(
        backend="kohya",
        command=command,
        cwd=str(cwd),
        train_script=str(train_script),
        configs={
            "training_config": str(train_config),
            "dataset_config": str(dataset_config),
        },
        train_entrypoint=str(train_script),
        train_args=["--config_file", str(train_config), "--dataset_config", str(dataset_config)],
    )


def build_diffsynth_run_spec(
    *,
    accelerate_launch: list[str],
    accelerate_config: str,
    threads: int,
    gpu_idx: str,
    train_script: Path,
    train_entrypoint: Path,
    train_args: list[str],
    args_path: str,
    metadata_path: str,
    cwd: Path,
) -> BackendRunSpec:
    command = [
        *accelerate_launch,
        "--config_file", str(accelerate_config),
        "--num_cpu_threads_per_process", str(int(threads)),
        "--gpu_ids", str(gpu_idx),
        str(train_entrypoint),
        *train_args,
    ]
    return BackendRunSpec(
        backend="diffsynth",
        command=command,
        cwd=str(cwd),
        train_script=str(train_script),
        metadata_path=str(metadata_path),
        configs={
            "diffsynth_args": str(args_path),
            "metadata": str(metadata_path),
        },
        train_entrypoint=str(train_entrypoint),
        train_args=list(train_args),
    )
