from __future__ import annotations

import json
import re
from pathlib import Path

from .dataset import migrate_diffsynth_metadata_for_anima


LEGACY_ANIMA_TARGET_MODULES = "q,k,v,o,ffn.0,ffn.2"
TORCHAO_MIN_EXCLUSIVE_VERSION = (0, 16, 0)
TORCHAO_PIP_SPEC = "torchao>0.16.0"


def normalize_lora_target_modules(value: str) -> str:
    """Use DiffSynth's Anima defaults unless the user provided a custom list."""
    normalized = (value or "").strip()
    if normalized == LEGACY_ANIMA_TARGET_MODULES:
        return ""
    return normalized


def _set_or_append_arg(args: list[str], key: str, value: str) -> list[str]:
    if key in args:
        idx = args.index(key)
        if idx + 1 < len(args):
            args[idx + 1] = value
        else:
            args.append(value)
    else:
        args.extend([key, value])
    return args


def migrate_args_for_anima(args: list[str]) -> list[str]:
    """Repair saved DiffSynth arg files created by older UI versions."""
    args = list(args)
    if "--lora_target_modules" in args:
        idx = args.index("--lora_target_modules")
        if idx + 1 < len(args):
            args[idx + 1] = normalize_lora_target_modules(args[idx + 1])

    if "--data_file_keys" not in args:
        try:
            metadata_idx = args.index("--dataset_metadata_path")
            args[metadata_idx:metadata_idx] = ["--data_file_keys", "image"]
        except ValueError:
            args.extend(["--data_file_keys", "image"])

    try:
        metadata_idx = args.index("--dataset_metadata_path") + 1
    except ValueError:
        return args
    if metadata_idx < len(args):
        args[metadata_idx] = str(migrate_diffsynth_metadata_for_anima(Path(args[metadata_idx])))
    return args


def parse_version_tuple(value: str) -> tuple[int, ...]:
    parts = re.findall(r"\d+", value.split("+", 1)[0])
    return tuple(int(part) for part in parts[:3])


def is_version_at_most(value: str, limit: tuple[int, ...]) -> bool:
    parsed = parse_version_tuple(value)
    if not parsed:
        return False
    max_len = max(len(parsed), len(limit))
    return parsed + (0,) * (max_len - len(parsed)) <= limit + (0,) * (max_len - len(limit))


def create_training_args(
    *,
    args_path: Path,
    output_dir: str,
    dit_model_path: Path,
    qwen3_model_path: Path,
    vae_model_path: Path,
    image_dir: str,
    metadata_csv: str,
    learning_rate: float,
    max_train_epochs: int,
    dataset_repeat: int,
    max_pixels: int,
    lora_rank: int,
    lora_target_modules: str,
    use_gradient_checkpointing: bool,
    gradient_accumulation_steps: int,
    save_steps: int,
    resume_lora_path: str = "",
) -> tuple[list[str], str]:
    model_paths_json = json.dumps([str(dit_model_path), str(qwen3_model_path), str(vae_model_path)])
    lora_target_modules = normalize_lora_target_modules(lora_target_modules)

    args: list[str] = [
        "--dataset_base_path", str(image_dir),
        "--dataset_metadata_path", str(metadata_csv),
        "--data_file_keys", "image",
        "--max_pixels", str(int(max_pixels)),
        "--dataset_repeat", str(int(dataset_repeat)),
        "--model_paths", model_paths_json,
        "--learning_rate", str(float(learning_rate)),
        "--num_epochs", str(int(max_train_epochs)),
        "--remove_prefix_in_ckpt", "pipe.dit.",
        "--output_path", str(output_dir),
        "--lora_base_model", "dit",
        "--lora_target_modules", lora_target_modules,
        "--lora_rank", str(int(lora_rank)),
        "--gradient_accumulation_steps", str(int(gradient_accumulation_steps)),
    ]
    if use_gradient_checkpointing:
        args.append("--use_gradient_checkpointing")
    if save_steps and int(save_steps) > 0:
        args += ["--save_steps", str(int(save_steps))]
    if resume_lora_path and str(resume_lora_path).strip():
        args += ["--lora_checkpoint", str(resume_lora_path).strip()]

    args_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args_path, "w", encoding="utf-8") as f:
        json.dump(args, f, indent=2, ensure_ascii=False)
    return args, str(args_path)

