from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from .dataset import migrate_diffsynth_metadata_for_anima


LEGACY_ANIMA_TARGET_MODULES = "q,k,v,o,ffn.0,ffn.2"
TORCHAO_MIN_EXCLUSIVE_VERSION = (0, 16, 0)
TORCHAO_PIP_SPEC = "torchao>0.16.0"


@dataclass(frozen=True)
class DiffSynthParameter:
    name: str
    cli_arg: str
    note: str


APPLIED_PARAMETERS: tuple[DiffSynthParameter, ...] = (
    DiffSynthParameter("Dataset base path", "--dataset_base_path", "Image root used with metadata image paths."),
    DiffSynthParameter("Metadata CSV", "--dataset_metadata_path", "CSV with image,prompt columns."),
    DiffSynthParameter("Data file keys", "--data_file_keys", "Always image for Anima metadata."),
    DiffSynthParameter("Max pixels", "--max_pixels", "Dynamic resolution pixel budget."),
    DiffSynthParameter("Dataset repeat", "--dataset_repeat", "DiffSynth repeat count per epoch."),
    DiffSynthParameter("Model paths", "--model_paths", "DiT, Qwen3 text encoder, and VAE safetensors."),
    DiffSynthParameter("Tokenizer path", "--tokenizer_path", "Local Qwen tokenizer directory; avoids DiffSynth downloading Qwen/Qwen3-0.6B at train time."),
    DiffSynthParameter("T5 tokenizer path", "--tokenizer_t5xxl_path", "Local SD3.5 tokenizer_3 directory; avoids DiffSynth downloading tokenizer files at train time."),
    DiffSynthParameter("Learning rate", "--learning_rate", "Optimizer learning rate used by DiffSynth."),
    DiffSynthParameter("Epochs", "--num_epochs", "Number of full dataset passes."),
    DiffSynthParameter("Output path", "--output_path", "Where LoRA checkpoints are saved."),
    DiffSynthParameter("LoRA base model", "--lora_base_model", "Fixed to dit for Anima LoRA."),
    DiffSynthParameter("LoRA target modules", "--lora_target_modules", "Blank lets DiffSynth auto-detect Anima modules."),
    DiffSynthParameter("LoRA rank", "--lora_rank", "Uses the UI Network Dim value."),
    DiffSynthParameter("Gradient accumulation", "--gradient_accumulation_steps", "Affects optimizer stepping, not per-step batch VRAM."),
    DiffSynthParameter("Gradient checkpointing", "--use_gradient_checkpointing", "Boolean flag when enabled."),
    DiffSynthParameter("Save steps", "--save_steps", "Only present when greater than 0."),
    DiffSynthParameter("Resume checkpoint", "--lora_checkpoint", "Only present when continuing from an existing LoRA."),
)

IGNORED_KOHYA_PARAMETERS: tuple[str, ...] = (
    "Network Alpha",
    "Train Batch Size",
    "Max Grad Norm",
    "Optimizer",
    "LR Scheduler",
    "Resolution",
    "Kohya Repeats",
    "Caption Dropout",
    "Latent Cache",
    "Text Encoder Cache",
    "VAE Chunk Size",
    "Noise Offset",
    "Multires Noise",
    "Timestep Sampling",
)


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


def set_anima_tokenizer_args(
    args: list[str],
    tokenizer_path: str = "",
    tokenizer_t5xxl_path: str = "",
) -> list[str]:
    args = list(args)
    if tokenizer_path:
        _set_or_append_arg(args, "--tokenizer_path", str(tokenizer_path))
    if tokenizer_t5xxl_path:
        _set_or_append_arg(args, "--tokenizer_t5xxl_path", str(tokenizer_t5xxl_path))
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


def get_arg_value(args: list[str], key: str, default: str = "") -> str:
    try:
        idx = args.index(key)
    except ValueError:
        return default
    value_idx = idx + 1
    return args[value_idx] if value_idx < len(args) else default


def parameter_rows_from_args(args: list[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    flags = set(args)
    for spec in APPLIED_PARAMETERS:
        if spec.cli_arg in flags:
            if spec.cli_arg in {"--use_gradient_checkpointing"}:
                value = "enabled"
            else:
                value = get_arg_value(args, spec.cli_arg)
            status = "applied"
        else:
            value = ""
            status = "optional/off"
        rows.append({
            "parameter": spec.name,
            "diffsynth_arg": spec.cli_arg,
            "value": value,
            "status": status,
            "note": spec.note,
        })
    for name in IGNORED_KOHYA_PARAMETERS:
        rows.append({
            "parameter": name,
            "diffsynth_arg": "",
            "value": "",
            "status": "not used by DiffSynth",
            "note": "This is a Kohya-only UI setting and is intentionally not passed.",
        })
    return rows


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
    tokenizer_path: str = "",
    tokenizer_t5xxl_path: str = "",
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
    args = set_anima_tokenizer_args(args, tokenizer_path, tokenizer_t5xxl_path)
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
