"""
Anima LoRA Trainer — Local Gradio UI
Supports kohya-ss/sd-scripts and DiffSynth-Studio backends,
with TensorBoard logging and Chinese / English UI.
"""

import json
import os
import re
import shutil
import signal
import shlex
import socket
import subprocess
import sys
import threading
import time
import atexit
import importlib.metadata as importlib_metadata
import traceback
import urllib.request
from collections import deque
from datetime import datetime
from pathlib import Path

import gradio as gr
import toml

from i18n import t, get_lang, set_lang, SUPPORTED_LANGS
from trainer_core import dataset as dataset_core
from trainer_core import diffsynth as diffsynth_core
from trainer_core import diffsynth_support as diffsynth_support_core
from trainer_core import steps as steps_core
from trainer_core.backends import build_diffsynth_run_spec, build_kohya_run_spec
from trainer_core.catalog import model_status_rows, scan_output_files, scan_run_manifests
from trainer_core.manifest import create_run_manifest, list_output_files, update_run_manifest
from trainer_core.preflight import has_failures, run_preflight
from trainer_core.progress import ProgressTracker, parse_structured_progress, parse_tqdm_progress
from trainer_core.sample_queue import format_sample_elapsed, is_terminal_sample_status

try:
    from pyngrok import ngrok as _ngrok
    PYNGROK_AVAILABLE = True
except ImportError:
    PYNGROK_AVAILABLE = False
    _ngrok = None

IS_COLAB = ("COLAB_GPU" in os.environ) or ("COLAB_RELEASE_TAG" in os.environ)

# ---------------------------------------------------------------------------
# Paths (all relative to the project root where app.py lives)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
CONFIG_FILE = ROOT / "config.json"
CONFIGS_DIR = ROOT / "configs"
LOGS_DIR = ROOT / "logs"
TB_LOGS_ROOT = LOGS_DIR / "tb"
SAMPLES_DIR = LOGS_DIR / "samples"
TRAINING_STATE_FILE = LOGS_DIR / "current_training.json"
SAMPLE_QUEUE_FILE = LOGS_DIR / "sample_queue.json"
SAMPLE_LOG_TAIL_LINES = 80
SAMPLE_POLL_SECONDS = 2.0
SAMPLE_LORA_STABLE_SECONDS = 3.0
TRAINING_POLL_SECONDS = 1.0
MODELS_DIR = ROOT / "models" / "anima"
SD_SCRIPTS_DIR = ROOT / "sd-scripts"
DIFFSYNTH_DEFAULT_DIR = ROOT / "DiffSynth-Studio"
DIFFSYNTH_GIT_URL = "https://github.com/modelscope/DiffSynth-Studio.git"
DIFFSYNTH_LEGACY_ANIMA_TARGET_MODULES = "q,k,v,o,ffn.0,ffn.2"
DIFFSYNTH_TENSORBOARD_WRAPPER = ROOT / "tools" / "diffsynth_tensorboard_wrapper.py"
ANIMA_SAMPLE_SCRIPT = ROOT / "tools" / "anima_sample.py"
TORCHAO_MIN_EXCLUSIVE_VERSION = (0, 16, 0)
TORCHAO_PIP_SPEC = "torchao>0.16.0"

DIT_MODEL = MODELS_DIR / "dit" / "anima-preview.safetensors"
QWEN3_MODEL = MODELS_DIR / "text_encoder" / "qwen_3_06b_base.safetensors"
VAE_MODEL = MODELS_DIR / "vae" / "qwen_image_vae.safetensors"
TRAIN_SCRIPT = SD_SCRIPTS_DIR / "anima_train_network.py"
DIFFSYNTH_TRAIN_SCRIPT_REL = "examples/anima/model_training/train.py"

BASE_MODEL_URLS = {
    "anima-base-v1.0": "https://huggingface.co/circlestone-labs/Anima/resolve/main/split_files/diffusion_models/anima-base-v1.0.safetensors",
    "anima-preview3-base": "https://huggingface.co/circlestone-labs/Anima/resolve/main/split_files/diffusion_models/anima-preview3-base.safetensors",
    "anima-preview": "https://huggingface.co/circlestone-labs/Anima/resolve/main/split_files/diffusion_models/anima-preview.safetensors",
}

SUPPORT_MODEL_URLS = {
    "qwen3": "https://huggingface.co/circlestone-labs/Anima/resolve/main/split_files/text_encoders/qwen_3_06b_base.safetensors",
    "vae": "https://huggingface.co/circlestone-labs/Anima/resolve/main/split_files/vae/qwen_image_vae.safetensors",
}


def get_dit_model_path(base_model: str) -> Path:
    filenames = {
        "anima-base-v1.0": "anima-base-v1.0.safetensors",
        "anima-preview": "anima-preview.safetensors",
        "anima-preview3-base": "anima-preview3-base.safetensors",
    }
    return MODELS_DIR / "dit" / filenames.get(base_model, "anima-base-v1.0.safetensors")


def diffsynth_support_path(label: str, diffsynth_dir: str | Path = "") -> Path:
    for spec in diffsynth_support_core.SUPPORT_SPECS:
        if spec.label == label:
            return diffsynth_support_core.support_path(spec, resolve_diffsynth_dir(str(diffsynth_dir or "")))
    raise KeyError(label)


CONFIGS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)
TB_LOGS_ROOT.mkdir(exist_ok=True, parents=True)
SAMPLES_DIR.mkdir(exist_ok=True, parents=True)

# Project-local accelerate config — keeps use_cpu=false and mixed_precision=bf16
# scoped to this app only. See app_configs/accelerate_gpu.yaml to change these.
# Absolute path so cwd switching (DiffSynth backend uses ds_dir as cwd) doesn't break resolution.
ACCELERATE_CONFIG = str(ROOT / "app_configs" / "accelerate_gpu.yaml")


def resolve_accelerate_launch_cmd() -> list[str]:
    """Return a working Accelerate launcher prefix."""
    candidates: list[list[str]] = []
    exe_dir = Path(sys.executable).resolve().parent

    if os.name == "nt":
        candidates.extend([
            [str(exe_dir / "Scripts" / "accelerate.exe"), "launch"],
            [str(exe_dir / "accelerate.exe"), "launch"],
            [str(exe_dir / "Scripts" / "accelerate-launch.exe")],
            [str(exe_dir / "accelerate-launch.exe")],
        ])
    else:
        candidates.extend([
            [str(exe_dir / "accelerate"), "launch"],
            [str(exe_dir / "accelerate-launch")],
        ])

    path_accelerate = shutil.which("accelerate")
    if path_accelerate:
        candidates.append([path_accelerate, "launch"])

    path_accelerate_launch = shutil.which("accelerate-launch")
    if path_accelerate_launch:
        candidates.append([path_accelerate_launch])

    seen = set()
    for cmd in candidates:
        executable = cmd[0]
        if executable in seen:
            continue
        seen.add(executable)
        if Path(executable).exists():
            return cmd

    return [sys.executable, "-m", "accelerate.commands.launch"]

# ---------------------------------------------------------------------------
# Default settings
# ---------------------------------------------------------------------------
DEFAULTS = {
    # Language / backend (new)
    "language": "en",
    "backend": "kohya",
    "diffsynth_dir": "",
    # TensorBoard (new)
    "use_tensorboard": True,
    "tb_port": 6006,
    "tb_logdir": "",  # auto-derived per run if empty
    "ngrok_enable": False,
    "ngrok_token": "",
    # DiffSynth-specific (new)
    "lora_target_modules": "",
    "dataset_repeat": 50,
    "max_pixels": 1048576,
    "save_steps_ds": 0,
    "resume_lora_path": "",
    # Basic
    "project_name": "my_lora",
    "base_model": "anima-base-v1.0",
    "image_directory": "",
    "output_directory": "",
    "network_dim": 20,
    "network_alpha": 20,
    "learning_rate": 0.0001,
    "max_train_epochs": 10,
    "resolution": 768,
    "repeats": 10,
    "caption_dropout": 0.1,
    "gpu_index": "0",
    # Advanced
    "optimizer_type": "AdamW8bit",
    "lr_scheduler": "cosine_with_restarts",
    "lr_scheduler_num_cycles": 1,
    "lr_warmup_steps": 100,
    "train_batch_size": 1,
    "gradient_accumulation_steps": 1,
    "max_grad_norm": 1.0,
    "save_every_n_epochs": 1,
    "save_last_n_epochs": 4,
    "mixed_precision": "bf16",
    "gradient_checkpointing": True,
    "seed": 42,
    "noise_offset": 0.03,
    "multires_noise_discount": 0.3,
    "timestep_sampling": "sigmoid",
    "discrete_flow_shift": 1.0,
    "cache_latents": True,
    "cache_text_encoder_outputs": True,
    "vae_chunk_size": 64,
    "vae_disable_cache": True,
    "num_cpu_threads_per_process": 1,
    # Sampling
    "sample_enabled": False,
    "sample_every_n_epochs": 1,
    "sample_prompt": "1girl, masterpiece, best quality",
    "sample_negative_prompt": "low quality, blurry, bad anatomy",
    "sample_width": 768,
    "sample_height": 768,
    "sample_steps": 30,
    "sample_cfg_scale": 4.0,
    "sample_seed": 42,
    "sample_low_vram": False,
    # UI
    "log_tail_lines": 500,
    # Internal
    "last_train_config": "",
    "last_dataset_config": "",
    "last_diffsynth_args": "",
    "last_tb_logdir": "",
    "last_run_manifest": "",
    "last_config_status": "",
}


# ---------------------------------------------------------------------------
# Config persistence
# ---------------------------------------------------------------------------

def load_config() -> dict:
    cfg = dict(DEFAULTS)
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)
            cfg.update({k: v for k, v in saved.items() if k in DEFAULTS})
        except Exception:
            pass
    cfg["lora_target_modules"] = normalize_diffsynth_lora_target_modules(
        cfg.get("lora_target_modules", "")
    )
    return cfg


def save_config(cfg: dict):
    # Preserve any keys not in DEFAULTS (e.g. language was already saved by i18n)
    existing = {}
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except Exception:
            existing = {}
    existing.update(cfg)
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# GPU detection
# ---------------------------------------------------------------------------

def detect_gpus() -> list[str]:
    try:
        import torch
        if not torch.cuda.is_available():
            return ["CPU (no CUDA detected)"]
        choices = []
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            choices.append(f"{i}: {name}")
        return choices if choices else ["0", "1"]
    except ImportError:
        return ["0", "1"]


GPU_CHOICES = detect_gpus()


def gpu_index_from_choice(choice: str) -> str:
    if not choice:
        return "0"
    return str(choice).split(":")[0].strip()


def gpu_choice_from_index(index: str | int | None) -> str:
    saved = str(index if index is not None else "0")
    return next(
        (choice for choice in GPU_CHOICES if str(choice).startswith(saved + ":")),
        GPU_CHOICES[0] if GPU_CHOICES else saved,
    )


# ---------------------------------------------------------------------------
# Dataset validation
# ---------------------------------------------------------------------------

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}


def validate_dataset(image_dir: str) -> tuple[int, list[str], list[str]]:
    return dataset_core.validate_dataset(image_dir)


# ---------------------------------------------------------------------------
# DiffSynth metadata.csv generation
# ---------------------------------------------------------------------------

def generate_diffsynth_metadata(image_dir: str, output_path: Path) -> tuple[Path, int]:
    return dataset_core.generate_diffsynth_metadata(image_dir, output_path)


# ---------------------------------------------------------------------------
# kohya TOML config generation (ported directly from the notebook)
# ---------------------------------------------------------------------------

def create_kohya_training_config(
    project_name, output_dir, dit_model_path, qwen3_model_path, vae_model_path,
    network_dim=20, network_alpha=20, learning_rate=1e-4, max_train_epochs=10,
    optimizer_type="AdamW8bit", lr_scheduler="cosine_with_restarts",
    lr_scheduler_num_cycles=1, lr_warmup_steps=100,
    train_batch_size=1, gradient_accumulation_steps=1, max_grad_norm=1.0,
    save_every_n_epochs=1, save_last_n_epochs=4,
    mixed_precision="bf16", gradient_checkpointing=True,
    seed=42, noise_offset=0.03, multires_noise_discount=0.3,
    timestep_sampling="sigmoid", discrete_flow_shift=1.0,
    cache_latents=True, cache_text_encoder_outputs=True,
    vae_chunk_size=64, vae_disable_cache=True,
    logging_dir: str = "",
    resume_lora_path: str = "",
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    current_date = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    config_path = CONFIGS_DIR / f"{project_name}_training_{current_date}.toml"

    training_config = {
        "pretrained_model_name_or_path": str(dit_model_path),
        "qwen3": str(qwen3_model_path),
        "vae": str(vae_model_path),
        "network_module": "networks.lora_anima",
        "network_dim": int(network_dim),
        "network_alpha": int(network_alpha),
        "network_train_unet_only": True,
        "learning_rate": float(learning_rate),
        "optimizer_type": optimizer_type,
        "optimizer_args": ["weight_decay=0.1", "betas=[0.9, 0.99]"],
        "lr_scheduler": lr_scheduler,
        "lr_scheduler_num_cycles": int(lr_scheduler_num_cycles),
        "lr_warmup_steps": int(lr_warmup_steps),
        "max_train_epochs": int(max_train_epochs),
        "train_batch_size": int(train_batch_size),
        "gradient_accumulation_steps": int(gradient_accumulation_steps),
        "max_grad_norm": float(max_grad_norm),
        "seed": int(seed),
        "timestep_sampling": timestep_sampling,
        "discrete_flow_shift": float(discrete_flow_shift),
        "qwen3_max_token_length": 512,
        "t5_max_token_length": 512,
        "mixed_precision": mixed_precision,
        "gradient_checkpointing": bool(gradient_checkpointing),
        "cache_latents": bool(cache_latents),
        "cache_text_encoder_outputs": bool(cache_text_encoder_outputs),
        "vae_chunk_size": int(vae_chunk_size),
        "vae_disable_cache": bool(vae_disable_cache),
        "output_dir": str(output_dir),
        "output_name": project_name,
        "save_model_as": "safetensors",
        "save_precision": "bf16",
        "save_every_n_epochs": int(save_every_n_epochs),
        "save_last_n_epochs": int(save_last_n_epochs),
        "shuffle_caption": False,
        "caption_extension": ".txt",
        "noise_offset": float(noise_offset),
        "multires_noise_discount": float(multires_noise_discount),
        "training_comment": f"Anima LoRA - {datetime.now().strftime('%Y-%m-%d')}",
    }
    if logging_dir:
        training_config["log_with"] = "tensorboard"
        training_config["logging_dir"] = str(logging_dir)
    if resume_lora_path and str(resume_lora_path).strip():
        training_config["network_weights"] = str(resume_lora_path).strip()

    with open(config_path, "w", encoding="utf-8") as f:
        toml.dump(training_config, f)
    return str(config_path)


def create_dataset_config(project_name, image_dir, resolution=768, repeats=5, caption_dropout_rate=0.1) -> str:
    current_date = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    config_path = CONFIGS_DIR / f"{project_name}_dataset_{current_date}.toml"
    dataset_config = {
        "general": {
            "resolution": int(resolution),
            "enable_bucket": True,
            "bucket_no_upscale": False,
            "bucket_reso_steps": 64,
            "min_bucket_reso": 256,
            "max_bucket_reso": 4096,
        },
        "datasets": [
            {
                "resolution": int(resolution),
                "subsets": [
                    {
                        "num_repeats": int(repeats),
                        "image_dir": str(image_dir),
                        "caption_extension": ".txt",
                        "caption_dropout_rate": float(caption_dropout_rate),
                    }
                ],
            }
        ],
    }
    with open(config_path, "w", encoding="utf-8") as f:
        toml.dump(dataset_config, f)
    return str(config_path)


# ---------------------------------------------------------------------------
# DiffSynth CLI-arg generation
# ---------------------------------------------------------------------------

def normalize_diffsynth_lora_target_modules(value: str) -> str:
    """Use DiffSynth's Anima defaults unless the user provided a custom list."""
    return diffsynth_core.normalize_lora_target_modules(value)


def migrate_diffsynth_args_for_anima(args: list[str]) -> list[str]:
    """Repair saved DiffSynth arg files created by older UI versions."""
    return diffsynth_core.migrate_args_for_anima(args)


def migrate_diffsynth_metadata_for_anima(metadata_path: Path) -> Path:
    """Convert legacy DiffSynth metadata file_name/text columns to image/prompt."""
    return dataset_core.migrate_diffsynth_metadata_for_anima(metadata_path)


def parse_version_tuple(value: str) -> tuple[int, ...]:
    return diffsynth_core.parse_version_tuple(value)


def is_version_at_most(value: str, limit: tuple[int, ...]) -> bool:
    return diffsynth_core.is_version_at_most(value, limit)


def installed_package_version(name: str) -> str | None:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def create_diffsynth_training_args(
    project_name: str, output_dir: str,
    dit_model_path: Path, qwen3_model_path: Path, vae_model_path: Path,
    image_dir: str, metadata_csv: str,
    learning_rate: float, max_train_epochs: int,
    dataset_repeat: int, max_pixels: int,
    lora_rank: int, lora_target_modules: str,
    use_gradient_checkpointing: bool,
    gradient_accumulation_steps: int,
    save_steps: int,
    resume_lora_path: str = "",
    diffsynth_dir: str | Path = "",
) -> tuple[list[str], str]:
    """Build the DiffSynth CLI args list and persist them next to other configs."""
    current_date = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    args_path = CONFIGS_DIR / f"{project_name}_diffsynth_args_{current_date}.json"
    ds_dir = resolve_diffsynth_dir(str(diffsynth_dir or load_config().get("diffsynth_dir", "")))
    return diffsynth_core.create_training_args(
        args_path=args_path,
        output_dir=output_dir,
        dit_model_path=dit_model_path,
        qwen3_model_path=qwen3_model_path,
        vae_model_path=vae_model_path,
        image_dir=image_dir,
        metadata_csv=metadata_csv,
        learning_rate=learning_rate,
        max_train_epochs=max_train_epochs,
        dataset_repeat=dataset_repeat,
        max_pixels=max_pixels,
        lora_rank=lora_rank,
        lora_target_modules=lora_target_modules,
        use_gradient_checkpointing=use_gradient_checkpointing,
        gradient_accumulation_steps=gradient_accumulation_steps,
        save_steps=save_steps,
        resume_lora_path=resume_lora_path,
        tokenizer_path=str(diffsynth_support_path("DiffSynth:Qwen tokenizer", ds_dir)),
        tokenizer_t5xxl_path=str(diffsynth_support_path("DiffSynth:SD3.5 tokenizer_3", ds_dir)),
    )


def resolve_diffsynth_dir(cfg_value: str) -> Path:
    if cfg_value and cfg_value.strip():
        return Path(cfg_value).expanduser()
    return DIFFSYNTH_DEFAULT_DIR

def _ensure_diffsynth_installed_legacy(diffsynth_dir: Path):
    """Generator yielding log lines, ensuring `import diffsynth` works in `sys.executable`.

    Final yield is a tuple ('__done__', ok: bool, message: str).
    Clones the repo if missing, then runs `pip install -e` against the current Python.
    This protects against venv / system-Python mix-ups (e.g. Colab + .venv) where the
    user's previous setup installed DiffSynth into a different interpreter than the one
    actually launching training.
    """
    # Quick check first: maybe it's already importable in this interpreter.
    check = subprocess.run(
        [sys.executable, "-c", "import diffsynth"],
        capture_output=True, text=True,
    )
    if check.returncode == 0:
        yield "✓ DiffSynth-Studio is already importable in current Python."
        yield ("__done__", True, "already installed")
        return

    yield f"DiffSynth-Studio not importable from {sys.executable} — installing now."

    # Clone if missing
    if not diffsynth_dir.exists():
        yield f"Cloning {DIFFSYNTH_GIT_URL} → {diffsynth_dir} ..."
        try:
            proc = subprocess.Popen(
                ["git", "clone", "--depth", "1", DIFFSYNTH_GIT_URL, str(diffsynth_dir)],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                universal_newlines=True, bufsize=1, encoding="utf-8", errors="ignore",
            )
        except FileNotFoundError:
            yield ("__done__", False, "git not found on PATH — install git and retry")
            return
        for line in iter(proc.stdout.readline, ""):
            yield line.rstrip("\n")
        proc.wait()
        if proc.returncode != 0:
            yield ("__done__", False, f"git clone failed (exit {proc.returncode})")
            return

    if not (diffsynth_dir / DIFFSYNTH_TRAIN_SCRIPT_REL).exists():
        yield ("__done__", False, f"Cloned dir is missing {DIFFSYNTH_TRAIN_SCRIPT_REL}")
        return

    # Editable install against the CURRENT interpreter
    yield f"Running: {sys.executable} -m pip install -e {diffsynth_dir}"
    proc = subprocess.Popen(
        [sys.executable, "-m", "pip", "install", "-e", str(diffsynth_dir)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        universal_newlines=True, bufsize=1, encoding="utf-8", errors="ignore",
    )
    for line in iter(proc.stdout.readline, ""):
        yield line.rstrip("\n")
    proc.wait()
    if proc.returncode != 0:
        yield ("__done__", False, f"pip install -e failed (exit {proc.returncode})")
        return

    # Verify
    check2 = subprocess.run(
        [sys.executable, "-c", "import diffsynth; print(diffsynth.__file__)"],
        capture_output=True, text=True,
    )
    if check2.returncode != 0:
        yield ("__done__", False, f"After install, `import diffsynth` still fails:\n{check2.stderr}")
        return
    yield f"✓ DiffSynth installed at: {check2.stdout.strip()}"
    yield ("__done__", True, "installed")


def ensure_diffsynth_installed(diffsynth_dir: Path):
    """Ensure a local DiffSynth-Studio checkout and an importable package exist."""
    if diffsynth_dir.exists() and not diffsynth_dir.is_dir():
        yield ("__done__", False, f"{diffsynth_dir} exists but is not a directory")
        return

    ds_script = diffsynth_dir / DIFFSYNTH_TRAIN_SCRIPT_REL

    def stream_process(cmd: list[str], cwd: str | None = None):
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
            cwd=cwd,
            encoding="utf-8",
            errors="ignore",
        )
        for line in iter(proc.stdout.readline, ""):
            yield line.rstrip("\n")
        proc.wait()
        return proc.returncode

    def ensure_torchao_compatible():
        version = installed_package_version("torchao")
        if version is None:
            return True, "torchao not installed"
        if not is_version_at_most(version, TORCHAO_MIN_EXCLUSIVE_VERSION):
            yield f"torchao {version} is compatible with PEFT."
            return True, "torchao compatible"

        yield (
            f"torchao {version} is incompatible with current PEFT; "
            f"upgrading to {TORCHAO_PIP_SPEC} ..."
        )
        returncode = yield from stream_process(
            [sys.executable, "-m", "pip", "install", "--upgrade", TORCHAO_PIP_SPEC]
        )
        if returncode == 0:
            new_version = installed_package_version("torchao") or "unknown"
            yield f"torchao upgraded to {new_version}."
            return True, "torchao upgraded"

        yield "torchao upgrade failed; uninstalling incompatible torchao so PEFT can use standard LoRA dispatch."
        returncode = yield from stream_process(
            [sys.executable, "-m", "pip", "uninstall", "-y", "torchao"]
        )
        if returncode != 0:
            return False, f"torchao upgrade/uninstall failed (exit {returncode})"
        yield "Incompatible torchao removed."
        return True, "torchao removed"

    def clone_repo():
        yield f"Cloning {DIFFSYNTH_GIT_URL} -> {diffsynth_dir} ..."
        try:
            diffsynth_dir.parent.mkdir(parents=True, exist_ok=True)
            returncode = yield from stream_process(
                ["git", "clone", "--depth", "1", DIFFSYNTH_GIT_URL, str(diffsynth_dir)]
            )
        except FileNotFoundError:
            yield ("__done__", False, "git not found on PATH - install git and retry")
            return
        if returncode != 0:
            yield ("__done__", False, f"git clone failed (exit {returncode})")

    if not ds_script.exists():
        if not diffsynth_dir.exists() or not any(diffsynth_dir.iterdir()):
            for item in clone_repo():
                if isinstance(item, tuple):
                    yield item
                    return
                yield item
        elif (diffsynth_dir / ".git").exists():
            yield f"DiffSynth-Studio exists but Anima train script is missing; updating {diffsynth_dir} ..."
            try:
                returncode = yield from stream_process(["git", "pull", "--ff-only"], cwd=str(diffsynth_dir))
            except FileNotFoundError:
                yield ("__done__", False, "git not found on PATH - install git and retry")
                return
            if returncode != 0:
                yield ("__done__", False, f"git pull failed (exit {returncode})")
                return
        else:
            yield (
                "__done__",
                False,
                f"{diffsynth_dir} exists but is missing {DIFFSYNTH_TRAIN_SCRIPT_REL}. "
                "Choose an empty directory or a DiffSynth-Studio git clone.",
            )
            return

    ds_script = diffsynth_dir / DIFFSYNTH_TRAIN_SCRIPT_REL
    if not ds_script.exists():
        yield ("__done__", False, f"DiffSynth-Studio is missing {DIFFSYNTH_TRAIN_SCRIPT_REL}")
        return

    check = subprocess.run(
        [sys.executable, "-c", "import diffsynth; print(diffsynth.__file__)"],
        capture_output=True,
        text=True,
    )
    if check.returncode == 0:
        imported_path = Path(check.stdout.strip()).resolve()
        try:
            imported_path.relative_to(diffsynth_dir.resolve())
            yield f"DiffSynth-Studio is already installed from: {imported_path}"
            torchao_ok, torchao_msg = yield from ensure_torchao_compatible()
            if not torchao_ok:
                yield ("__done__", False, torchao_msg)
                return
            yield ("__done__", True, "already installed")
            return
        except ValueError:
            yield f"Found diffsynth at {imported_path}; reinstalling local DiffSynth-Studio checkout."

    if check.returncode != 0:
        yield f"DiffSynth-Studio not importable from {sys.executable} - installing now."
    yield f"Running: {sys.executable} -m pip install -e {diffsynth_dir}"
    try:
        returncode = yield from stream_process([sys.executable, "-m", "pip", "install", "-e", str(diffsynth_dir)])
    except FileNotFoundError:
        yield ("__done__", False, "python or pip not found")
        return
    if returncode != 0:
        yield ("__done__", False, f"pip install -e failed (exit {returncode})")
        return

    check2 = subprocess.run(
        [sys.executable, "-c", "import diffsynth; print(diffsynth.__file__)"],
        capture_output=True,
        text=True,
    )
    if check2.returncode != 0:
        yield ("__done__", False, f"After install, import diffsynth still fails:\n{check2.stderr}")
        return
    yield f"DiffSynth installed at: {check2.stdout.strip()}"
    torchao_ok, torchao_msg = yield from ensure_torchao_compatible()
    if not torchao_ok:
        yield ("__done__", False, torchao_msg)
        return
    yield ("__done__", True, "installed")


# ---------------------------------------------------------------------------
# Configure Training handler
# ---------------------------------------------------------------------------

def configure_training(
    backend, diffsynth_dir,
    project_name, base_model, image_directory, output_directory,
    network_dim, network_alpha, learning_rate, max_train_epochs,
    resolution, repeats, caption_dropout, gpu_index_choice,
    # advanced (kohya)
    optimizer_type, lr_scheduler, lr_scheduler_num_cycles, lr_warmup_steps,
    train_batch_size, gradient_accumulation_steps, max_grad_norm,
    save_every_n_epochs, save_last_n_epochs, mixed_precision,
    gradient_checkpointing, seed, noise_offset, multires_noise_discount,
    timestep_sampling, discrete_flow_shift,
    cache_latents, cache_text_encoder_outputs, vae_chunk_size, vae_disable_cache,
    num_cpu_threads_per_process, log_tail_lines,
    # DiffSynth-specific
    lora_target_modules, dataset_repeat, max_pixels, save_steps_ds,
    # TensorBoard
    use_tensorboard, tb_logdir_input, tb_port,
    # Resume / continuation
    resume_lora_path,
) -> tuple[str, str, str, str, str]:
    """
    Returns (status_message, last_train_config_path, last_dataset_config_path,
             last_diffsynth_args_path, last_tb_logdir).
    """
    lines = []
    backend = (backend or "kohya").lower()

    # --- Validate inputs ---
    if not project_name.strip():
        return t("err_project_empty"), "", "", "", ""
    if not image_directory.strip():
        return t("err_image_dir_empty"), "", "", "", ""
    if not output_directory.strip():
        return t("err_output_dir_empty"), "", "", "", ""

    lines.append(t("info_backend", backend=backend))
    lines.append(t("info_project", name=project_name))
    lines.append(t("info_image_dir", dir=image_directory))
    lines.append(t("info_output_dir", dir=output_directory))
    lines.append("")

    # --- Validate dataset ---
    try:
        n_images, missing, warnings = validate_dataset(image_directory)
    except (FileNotFoundError, NotADirectoryError) as e:
        return f"❌ {e}", "", "", "", ""

    lines.append(t("info_images_found", n=n_images))
    if missing:
        lines.append(t("info_missing_captions", n=len(missing)))
        for m in missing[:20]:
            lines.append(f"    • {m}")
        if len(missing) > 20:
            lines.append(t("info_more", n=len(missing) - 20))
    else:
        lines.append(t("info_all_have_captions"))

    for w in warnings:
        lines.append(f"⚠ {w}")

    if n_images == 0:
        lines.append("")
        lines.append(t("info_no_images"))
        return "\n".join(lines), "", "", "", ""

    # --- Step estimate ---
    step_estimate = steps_core.estimate_steps(
        backend=backend,
        n_images=n_images,
        repeats=int(repeats),
        dataset_repeat=int(dataset_repeat),
        epochs=int(max_train_epochs),
        train_batch_size=int(train_batch_size),
        gradient_accumulation_steps=int(gradient_accumulation_steps),
    )
    effective_repeats = step_estimate["effective_repeats"]
    spe = step_estimate["progress_per_epoch"]
    total = step_estimate["progress_total"]
    lines.append("")
    lines.append(t("info_step_header"))
    lines.append(t("info_step_per_epoch", n=spe, imgs=n_images, repeats=effective_repeats))
    lines.append(t("info_step_total", n=total, spe=spe, ep=int(max_train_epochs)))
    if backend == "diffsynth":
        lines.append(t("info_optimizer_step_total", n=step_estimate["optimizer_total"], grad=int(gradient_accumulation_steps)))
        lines.append(t("info_step_diffsynth_note"))
    lines.append(t("info_step_footer"))

    if resume_lora_path and str(resume_lora_path).strip():
        lines.append("")
        lines.append(t("info_resume_enabled", path=str(resume_lora_path).strip()))

    # --- Validate models ---
    lines.append("")
    lines.append(t("info_checking_models"))
    dit_model = get_dit_model_path(base_model)
    missing_models = []
    for label, path in [("DiT", dit_model), ("Qwen3", QWEN3_MODEL), ("VAE", VAE_MODEL)]:
        if Path(path).exists():
            lines.append(f"  ✓ {label}: {path}")
        else:
            if label == "DiT":
                lines.append(f"  ℹ {label}: {path}")
                lines.append(t("info_will_download"))
            else:
                lines.append(f"  ✗ {label} missing: {path}")
                missing_models.append(label)
    if missing_models:
        lines.append("")
        lines.append(t("info_missing_models", list=", ".join(missing_models)))
        lines.append(t("info_run_setup"))
        return "\n".join(lines), "", "", "", ""

    # --- Resolve TensorBoard logdir ---
    tb_logdir = ""
    if use_tensorboard:
        candidate = (tb_logdir_input or "").strip()
        if not candidate:
            ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
            candidate = str(TB_LOGS_ROOT / f"{project_name}_{ts}")
        Path(candidate).mkdir(parents=True, exist_ok=True)
        tb_logdir = candidate
        lines.append("")
        lines.append(t("info_tb_enabled", dir=tb_logdir))

    # --- Backend-specific config generation ---
    lines.append("")
    train_cfg = ""
    dataset_cfg = ""
    diffsynth_args_path = ""

    if backend == "kohya":
        lines.append(t("info_generating_toml"))
        try:
            train_cfg = create_kohya_training_config(
                project_name=project_name, output_dir=output_directory,
                dit_model_path=dit_model, qwen3_model_path=QWEN3_MODEL, vae_model_path=VAE_MODEL,
                network_dim=network_dim, network_alpha=network_alpha,
                learning_rate=learning_rate, max_train_epochs=max_train_epochs,
                optimizer_type=optimizer_type, lr_scheduler=lr_scheduler,
                lr_scheduler_num_cycles=lr_scheduler_num_cycles, lr_warmup_steps=lr_warmup_steps,
                train_batch_size=train_batch_size, gradient_accumulation_steps=gradient_accumulation_steps,
                max_grad_norm=max_grad_norm,
                save_every_n_epochs=save_every_n_epochs, save_last_n_epochs=save_last_n_epochs,
                mixed_precision=mixed_precision, gradient_checkpointing=gradient_checkpointing,
                seed=seed, noise_offset=noise_offset, multires_noise_discount=multires_noise_discount,
                timestep_sampling=timestep_sampling, discrete_flow_shift=discrete_flow_shift,
                cache_latents=cache_latents, cache_text_encoder_outputs=cache_text_encoder_outputs,
                vae_chunk_size=vae_chunk_size, vae_disable_cache=vae_disable_cache,
                logging_dir=tb_logdir,
                resume_lora_path=resume_lora_path,
            )
            dataset_cfg = create_dataset_config(
                project_name=project_name, image_dir=image_directory,
                resolution=resolution, repeats=repeats, caption_dropout_rate=caption_dropout,
            )
        except Exception as e:
            lines.append(t("err_generate_failed", err=e))
            return "\n".join(lines), "", "", "", ""

        lines.append(t("info_train_cfg_written", path=train_cfg))
        lines.append(t("info_dataset_cfg_written", path=dataset_cfg))

    elif backend == "diffsynth":
        # Soft check — if DiffSynth-Studio isn't cloned yet, just inform the user.
        # Actual install happens at training time via ensure_diffsynth_installed().
        ds_dir = resolve_diffsynth_dir(diffsynth_dir)
        ds_script = ds_dir / DIFFSYNTH_TRAIN_SCRIPT_REL
        if not ds_script.exists():
            lines.append("")
            lines.append(t("info_diffsynth_will_install", path=str(ds_dir)))

        lines.append(t("info_generating_metadata"))
        try:
            ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
            metadata_path = CONFIGS_DIR / f"{project_name}_metadata_{ts}.csv"
            metadata_path, n_rows = generate_diffsynth_metadata(image_directory, metadata_path)
            lines.append(t("info_metadata_written", path=str(metadata_path), n=n_rows))

            _, diffsynth_args_path = create_diffsynth_training_args(
                project_name=project_name,
                output_dir=output_directory,
                dit_model_path=dit_model,
                qwen3_model_path=QWEN3_MODEL,
                vae_model_path=VAE_MODEL,
                image_dir=image_directory,
                metadata_csv=str(metadata_path),
                learning_rate=learning_rate,
                max_train_epochs=max_train_epochs,
                dataset_repeat=dataset_repeat,
                max_pixels=max_pixels,
                lora_rank=network_dim,
                lora_target_modules=lora_target_modules,
                use_gradient_checkpointing=gradient_checkpointing,
                gradient_accumulation_steps=gradient_accumulation_steps,
                save_steps=save_steps_ds,
                resume_lora_path=resume_lora_path,
                diffsynth_dir=ds_dir,
            )
            lines.append(t("info_args_written", path=diffsynth_args_path))
        except Exception as e:
            lines.append(t("err_generate_failed", err=e))
            return "\n".join(lines), "", "", "", ""

    else:
        lines.append(f"❌ Unknown backend: {backend}")
        return "\n".join(lines), "", "", "", ""

    # --- Save all settings to config.json ---
    cfg = {
        "backend": backend,
        "diffsynth_dir": diffsynth_dir or "",
        "use_tensorboard": bool(use_tensorboard),
        "tb_port": int(tb_port),
        "tb_logdir": tb_logdir,
        "lora_target_modules": normalize_diffsynth_lora_target_modules(lora_target_modules),
        "dataset_repeat": int(dataset_repeat),
        "max_pixels": int(max_pixels),
        "save_steps_ds": int(save_steps_ds),
        "resume_lora_path": (resume_lora_path or "").strip(),
        "project_name": project_name,
        "base_model": base_model,
        "image_directory": image_directory,
        "output_directory": output_directory,
        "network_dim": int(network_dim),
        "network_alpha": int(network_alpha),
        "learning_rate": float(learning_rate),
        "max_train_epochs": int(max_train_epochs),
        "resolution": int(resolution),
        "repeats": int(repeats),
        "caption_dropout": float(caption_dropout),
        "gpu_index": gpu_index_from_choice(gpu_index_choice),
        "optimizer_type": optimizer_type,
        "lr_scheduler": lr_scheduler,
        "lr_scheduler_num_cycles": int(lr_scheduler_num_cycles),
        "lr_warmup_steps": int(lr_warmup_steps),
        "train_batch_size": int(train_batch_size),
        "gradient_accumulation_steps": int(gradient_accumulation_steps),
        "max_grad_norm": float(max_grad_norm),
        "save_every_n_epochs": int(save_every_n_epochs),
        "save_last_n_epochs": int(save_last_n_epochs),
        "mixed_precision": mixed_precision,
        "gradient_checkpointing": bool(gradient_checkpointing),
        "seed": int(seed),
        "noise_offset": float(noise_offset),
        "multires_noise_discount": float(multires_noise_discount),
        "timestep_sampling": timestep_sampling,
        "discrete_flow_shift": float(discrete_flow_shift),
        "cache_latents": bool(cache_latents),
        "cache_text_encoder_outputs": bool(cache_text_encoder_outputs),
        "vae_chunk_size": int(vae_chunk_size),
        "vae_disable_cache": bool(vae_disable_cache),
        "num_cpu_threads_per_process": int(num_cpu_threads_per_process),
        "log_tail_lines": int(log_tail_lines),
        "last_train_config": train_cfg,
        "last_dataset_config": dataset_cfg,
        "last_diffsynth_args": diffsynth_args_path,
        "last_tb_logdir": tb_logdir,
        "last_run_manifest": "",
        "last_config_status": "",
    }

    lines.append("")
    lines.append(t("info_ready"))
    cfg["last_config_status"] = "\n".join(lines)
    save_config(cfg)
    return cfg["last_config_status"], train_cfg, dataset_cfg, diffsynth_args_path, tb_logdir


# ---------------------------------------------------------------------------
# DiffSynth stdout → TensorBoard loss writer
# ---------------------------------------------------------------------------

_LOSS_RE = re.compile(r"loss[=:\s]+([0-9]*\.?[0-9]+(?:[eE][+-]?\d+)?)")


class DiffSynthLossWriter:
    """Parse tqdm/stdout loss values and stream them to a TF events file."""
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        self.step = 0
        self.writer = None
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir)
        except Exception as e:
            print(f"[DiffSynthLossWriter] tensorboard unavailable: {e}", file=sys.stderr)

    def feed(self, line: str):
        if self.writer is None:
            return
        m = _LOSS_RE.search(line)
        if m:
            try:
                value = float(m.group(1))
                if 0.0 < value < 1e6:  # filter junk matches
                    self.writer.add_scalar("loss", value, self.step)
                    self.step += 1
            except ValueError:
                pass

    def close(self):
        if self.writer is not None:
            try:
                self.writer.flush()
                self.writer.close()
            except Exception:
                pass


def get_cli_arg_value(args: list[str], key: str, default: str = "") -> str:
    try:
        idx = args.index(key)
    except ValueError:
        return default
    value_idx = idx + 1
    return args[value_idx] if value_idx < len(args) else default


def format_preflight_checks(checks) -> list[str]:
    icons = {"ok": "✓", "warn": "⚠", "fail": "✗"}
    lines = [t("info_preflight_header")]
    for check in checks:
        lines.append(f"  {icons.get(check.status, '?')} {check.name}: {check.message}")
    return lines


def estimate_progress_total_from_config(backend: str, cfg: dict) -> int:
    try:
        n_images, _, _ = validate_dataset(cfg.get("image_directory", ""))
        estimate = steps_core.estimate_steps(
            backend=backend,
            n_images=n_images,
            repeats=int(cfg.get("repeats", 1)),
            dataset_repeat=int(cfg.get("dataset_repeat", 1)),
            epochs=int(cfg.get("max_train_epochs", 1)),
            train_batch_size=int(cfg.get("train_batch_size", 1)),
            gradient_accumulation_steps=int(cfg.get("gradient_accumulation_steps", 1)),
        )
        return int(estimate["progress_total"])
    except Exception:
        return 0


def estimate_progress_epoch_total_from_config(backend: str, cfg: dict) -> int:
    try:
        n_images, _, _ = validate_dataset(cfg.get("image_directory", ""))
        estimate = steps_core.estimate_steps(
            backend=backend,
            n_images=n_images,
            repeats=int(cfg.get("repeats", 1)),
            dataset_repeat=int(cfg.get("dataset_repeat", 1)),
            epochs=int(cfg.get("max_train_epochs", 1)),
            train_batch_size=int(cfg.get("train_batch_size", 1)),
            gradient_accumulation_steps=int(cfg.get("gradient_accumulation_steps", 1)),
        )
        return int(estimate["progress_per_epoch"])
    except Exception:
        return 0


class TailLogBuffer:
    def __init__(self, max_lines: int = 500):
        self.max_lines = max(int(max_lines or 500), 50)
        self.lines: list[str] = []

    def append(self, line: str) -> None:
        self.lines.append(line)
        overflow = len(self.lines) - self.max_lines
        if overflow > 0:
            del self.lines[:overflow]

    def text(self) -> str:
        return "\n".join(self.lines)


def _write_json_atomic(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    tmp.replace(path)


def _read_json_file(path: Path, default):
    if not path.exists():
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def _pid_is_running(pid: int | str | None) -> bool:
    try:
        pid = int(pid or 0)
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    if os.name == "nt":
        try:
            result = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                capture_output=True,
                text=True,
                timeout=3,
            )
            return str(pid) in result.stdout
        except Exception:
            return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except Exception:
        return False


_training_lock = threading.Lock()
_current_training: dict = {
    "process": None,
    "thread": None,
    "manifest_path": "",
    "log_file": "",
    "stop_requested": False,
    "last_log_tail": "",
    "status": "idle",
    "started_at": "",
    "finished_at": "",
    "backend": "",
    "project_name": "",
}
_sample_lock = threading.Lock()
_sample_jobs: list[dict] = []
_pending_sample_jobs: list[dict] = []
_sample_job_counter = 0
_UNSET = object()

MODEL_TABLE_HEADERS = ["model", "status", "size_gb", "path", "url"]
DIFFSYNTH_PARAMETER_TABLE_HEADERS = ["parameter", "diffsynth_arg", "value", "status", "note"]
HISTORY_TABLE_HEADERS = ["created_at", "project", "backend", "status", "latest_output", "log_file", "manifest"]
OUTPUT_TABLE_HEADERS = ["name", "size_mb", "modified", "path"]
SAMPLE_QUEUE_HEADERS = ["id", "created_at", "status", "elapsed", "output", "log_path", "lora", "seed", "size", "steps", "cfg_scale", "prompt"]


def _training_state_for_disk(state: dict) -> dict:
    payload = {
        key: value
        for key, value in state.items()
        if key not in {"process", "thread"} and not str(key).startswith("_")
    }
    process = state.get("process")
    if process is not None:
        payload["pid"] = getattr(process, "pid", None)
    return payload


def _persist_current_training_unlocked() -> None:
    _write_json_atomic(TRAINING_STATE_FILE, _training_state_for_disk(_current_training))


def _set_current_training(
    process=_UNSET,
    manifest_path: str | None = None,
    log_file: str | None = None,
    *,
    status: str | None = None,
    thread=_UNSET,
    stop_requested: bool | None = None,
    started_at: str | None = None,
    finished_at: str | None = None,
    backend: str | None = None,
    project_name: str | None = None,
    last_log_tail: str | None = None,
    error: str | None = None,
    clear_error: bool = False,
) -> None:
    with _training_lock:
        if process is not _UNSET:
            _current_training["process"] = process
        if thread is not _UNSET:
            _current_training["thread"] = thread
        if manifest_path is not None:
            _current_training["manifest_path"] = manifest_path
        if log_file is not None:
            _current_training["log_file"] = log_file
        if status is not None:
            _current_training["status"] = status
        if stop_requested is not None:
            _current_training["stop_requested"] = bool(stop_requested)
        if started_at is not None:
            _current_training["started_at"] = started_at
        if finished_at is not None:
            _current_training["finished_at"] = finished_at
        if backend is not None:
            _current_training["backend"] = backend
        if project_name is not None:
            _current_training["project_name"] = project_name
        if last_log_tail is not None:
            _current_training["last_log_tail"] = last_log_tail
        if clear_error:
            _current_training.pop("error", None)
        if error is not None:
            _current_training["error"] = error
        _persist_current_training_unlocked()


def _update_current_training_tail(text: str) -> None:
    with _training_lock:
        _current_training["last_log_tail"] = text or ""
        _persist_current_training_unlocked()


def _get_current_training() -> dict:
    with _training_lock:
        return dict(_current_training)


def _is_training_active() -> bool:
    process = _get_current_training().get("process")
    return process is not None and process.poll() is None


def _is_training_busy() -> bool:
    state = _get_current_training()
    process = state.get("process")
    if process is not None and process.poll() is None:
        return True
    thread = state.get("thread")
    return thread is not None and thread.is_alive()


def _load_persisted_training_state() -> dict:
    state = _read_json_file(TRAINING_STATE_FILE, {})
    return state if isinstance(state, dict) else {}


def _last_manifest_payload(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    manifest = cfg.get("last_run_manifest", "")
    if manifest:
        payload = _read_json_file(Path(manifest), {})
        if payload:
            payload["_manifest_path"] = manifest
            return payload
    roots = [LOGS_DIR]
    if cfg.get("output_directory"):
        roots.append(cfg["output_directory"])
    rows = scan_run_manifests(*roots)
    if rows:
        payload = _read_json_file(Path(rows[0]["manifest"]), {})
        if payload:
            payload["_manifest_path"] = rows[0]["manifest"]
            return payload
    return {}


def _training_log_path_from_state_or_manifest(cfg: dict | None = None) -> str:
    state = _get_current_training()
    if state.get("log_file"):
        return str(state["log_file"])
    persisted = _load_persisted_training_state()
    if persisted.get("log_file"):
        return str(persisted["log_file"])
    manifest = _last_manifest_payload(cfg)
    if manifest.get("log_file"):
        return str(manifest["log_file"])
    return ""


def restored_training_log() -> str:
    cfg = load_config()
    state = _get_current_training()
    if state.get("last_log_tail"):
        return str(state["last_log_tail"])
    persisted = _load_persisted_training_state()
    if persisted.get("last_log_tail") and persisted.get("status") in {"starting", "running"}:
        return str(persisted["last_log_tail"])
    log_path = _training_log_path_from_state_or_manifest(cfg)
    if log_path:
        return _read_tail(Path(log_path), max_lines=int(cfg.get("log_tail_lines", 500)))
    return ""


def restored_config_status() -> str:
    cfg = load_config()
    lines: list[str] = []
    state = _get_current_training()
    persisted = _load_persisted_training_state()
    active = _is_training_active()
    if active:
        lines.append(t("training_status_active", pid=getattr(state.get("process"), "pid", "")))
    elif _is_training_busy():
        lines.append(t("training_status_thread_active", status=state.get("status", "running")))
    elif state.get("status") in {"success", "failed", "cancelled", "preflight_failed"}:
        lines.append(t("training_status_last", status=state.get("status", "")))
    elif persisted.get("status") in {"starting", "running"} and _pid_is_running(persisted.get("pid")):
        lines.append(t("training_status_detached", pid=persisted.get("pid")))
    elif persisted.get("status") in {"success", "failed", "cancelled", "preflight_failed"}:
        lines.append(t("training_status_last", status=persisted.get("status", "")))

    manifest_path = state.get("manifest_path") or persisted.get("manifest_path") or cfg.get("last_run_manifest", "")
    log_file = state.get("log_file") or persisted.get("log_file") or _training_log_path_from_state_or_manifest(cfg)
    if manifest_path:
        lines.append(t("training_status_manifest", path=manifest_path))
    if log_file:
        lines.append(t("training_status_log", path=log_file))

    saved_status = cfg.get("last_config_status", "")
    if saved_status:
        if lines:
            lines.append("")
        lines.append(saved_status)
    elif not lines:
        lines.append(t("training_status_no_saved_config"))
    return "\n".join(lines)


def _finish_current_training(status: str, error: str = "") -> None:
    with _training_lock:
        _current_training["process"] = None
        _current_training["thread"] = None
        _current_training["status"] = status
        _current_training["finished_at"] = datetime.now().isoformat(timespec="seconds")
        if error:
            _current_training["error"] = error
        _persist_current_training_unlocked()


def stop_training() -> str:
    state = _get_current_training()
    process = state.get("process")
    pid = getattr(process, "pid", None) if process is not None else None
    if process is None or process.poll() is not None:
        persisted = _load_persisted_training_state()
        pid = persisted.get("pid")
        if not _pid_is_running(pid):
            return t("stop_status_idle")
        process = None

    if not pid:
        return t("stop_status_idle")

    with _training_lock:
        _current_training["stop_requested"] = True
        _current_training["status"] = "cancel_requested"
        _persist_current_training_unlocked()

    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10,
            )
        else:
            os.killpg(os.getpgid(int(pid)), signal.SIGTERM)
    except Exception:
        try:
            if process is not None:
                process.kill()
            elif pid:
                os.kill(int(pid), signal.SIGTERM)
        except Exception:
            pass

    manifest_path = state.get("manifest_path") or _load_persisted_training_state().get("manifest_path")
    if manifest_path:
        update_run_manifest(
            manifest_path,
            status="cancel_requested",
            log_file=state.get("log_file", "") or _load_persisted_training_state().get("log_file", ""),
            output_files=list_output_files(load_config().get("output_directory", "")),
        )
    return t("stop_status_requested")


def _process_popen_kwargs() -> dict:
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"preexec_fn": os.setsid}


def _rows(items: list[dict], headers: list[str]) -> list[list]:
    return [[item.get(header, "") for header in headers] for item in items]


def _all_model_paths(base_model: str, diffsynth_dir: str = "") -> dict[str, str]:
    ds_dir = resolve_diffsynth_dir(diffsynth_dir or load_config().get("diffsynth_dir", ""))
    paths = {
        f"DiT:{name}": str(get_dit_model_path(name))
        for name in BASE_MODEL_URLS
    } | {
        "Qwen3": str(QWEN3_MODEL),
        "VAE": str(VAE_MODEL),
    }
    for spec in diffsynth_support_core.SUPPORT_SPECS:
        paths[spec.label] = str(diffsynth_support_core.support_path(spec, ds_dir))
    return paths


def _model_urls() -> dict[str, str]:
    urls = {
        **{f"DiT:{name}": url for name, url in BASE_MODEL_URLS.items()},
        "Qwen3": SUPPORT_MODEL_URLS["qwen3"],
        "VAE": SUPPORT_MODEL_URLS["vae"],
    }
    for spec in diffsynth_support_core.SUPPORT_SPECS:
        urls[spec.label] = f"modelscope://{spec.model_id} ({', '.join(spec.allow_patterns)})"
    return urls


def refresh_model_table(base_model: str, diffsynth_dir: str = ""):
    return _rows(model_status_rows(_all_model_paths(base_model, diffsynth_dir), _model_urls()), MODEL_TABLE_HEADERS)


def refresh_diffsynth_param_table(args_path: str = ""):
    path = Path(args_path or load_config().get("last_diffsynth_args", ""))
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            args = json.load(f)
    except Exception:
        return []
    if not isinstance(args, list):
        return []
    return _rows(diffsynth_core.parameter_rows_from_args(args), DIFFSYNTH_PARAMETER_TABLE_HEADERS)


def ensure_diffsynth_support_files(ds_dir: Path):
    for spec in diffsynth_support_core.SUPPORT_SPECS:
        path = diffsynth_support_core.support_path(spec, ds_dir)
        if diffsynth_support_core.is_support_ready(spec, ds_dir):
            yield f"{spec.label}: {t('model_status_ready')}"
            continue
        yield t("model_download_start", name=spec.label, path=str(path))
        try:
            diffsynth_support_core.download_support_model(spec, ds_dir)
        except Exception as exc:
            yield ("__done__", False, t("model_download_failed", name=spec.label, err=exc))
            return
        if not diffsynth_support_core.is_support_ready(spec, ds_dir):
            yield ("__done__", False, t("model_download_failed", name=spec.label, err=f"missing after download: {path}"))
            return
        yield t("model_download_done", name=spec.label)
    yield ("__done__", True, "ready")


def download_missing_models(base_model: str, diffsynth_dir: str = ""):
    buffer = TailLogBuffer(300)
    ds_dir = resolve_diffsynth_dir(diffsynth_dir or load_config().get("diffsynth_dir", ""))

    def emit(line: str):
        buffer.append(line)
        return refresh_model_table(base_model, str(ds_dir)), buffer.text()

    targets = {
        f"DiT:{base_model}": (get_dit_model_path(base_model), BASE_MODEL_URLS.get(base_model, "")),
        "Qwen3": (QWEN3_MODEL, SUPPORT_MODEL_URLS["qwen3"]),
        "VAE": (VAE_MODEL, SUPPORT_MODEL_URLS["vae"]),
    }
    for label, (path, url) in targets.items():
        if path.exists():
            yield emit(f"{label}: {t('model_status_ready')}")
            continue
        if not url:
            yield emit(f"{label}: missing download URL")
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        yield emit(t("model_download_start", name=label, path=str(path)))
        try:
            with urllib.request.urlopen(url) as response, open(path, "wb") as f:
                total = int(response.headers.get("Content-Length") or 0)
                downloaded = 0
                last_update = time.monotonic()
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    now = time.monotonic()
                    if now - last_update >= 2:
                        last_update = now
                        if total:
                            pct = downloaded / total * 100
                            yield emit(t("model_download_progress", name=label, pct=f"{pct:.1f}", mb=downloaded // (1024 * 1024)))
                        else:
                            yield emit(t("model_download_progress_unknown", name=label, mb=downloaded // (1024 * 1024)))
        except Exception as exc:
            yield emit(t("model_download_failed", name=label, err=exc))
            continue
        yield emit(t("model_download_done", name=label))

    yield emit(t("info_diffsynth_check"))
    install_ok = False
    install_msg = ""
    for item in ensure_diffsynth_installed(ds_dir):
        if isinstance(item, tuple) and item and item[0] == "__done__":
            _, install_ok, install_msg = item
        else:
            yield emit(str(item))
    if install_ok:
        for item in ensure_diffsynth_support_files(ds_dir):
            if isinstance(item, tuple) and item and item[0] == "__done__":
                if not item[1]:
                    yield emit(str(item[2]))
            else:
                yield emit(str(item))
    else:
        yield emit(t("model_diffsynth_support_skipped", path=f"{ds_dir}: {install_msg}"))
    yield emit(t("model_download_all_done"))


def refresh_history_table() -> list[dict]:
    cfg = load_config()
    roots = [LOGS_DIR]
    if cfg.get("output_directory"):
        roots.append(cfg["output_directory"])
    return _rows(scan_run_manifests(*roots), HISTORY_TABLE_HEADERS)


def refresh_outputs_table(output_dir: str = "") -> list[dict]:
    cfg = load_config()
    return _rows(scan_output_files(output_dir or cfg.get("output_directory", "")), OUTPUT_TABLE_HEADERS)


def latest_output_path(output_dir: str = "") -> str:
    cfg = load_config()
    return _find_latest_lora(output_dir or cfg.get("output_directory", ""))


def save_sample_settings(
    enabled: bool,
    every_n_epochs: int,
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    steps: int,
    cfg_scale: float,
    seed: int,
    low_vram: bool,
) -> str:
    save_config({
        "sample_enabled": bool(enabled),
        "sample_every_n_epochs": max(int(every_n_epochs or 1), 1),
        "sample_prompt": prompt or "",
        "sample_negative_prompt": negative_prompt or "",
        "sample_width": int(width or 768),
        "sample_height": int(height or 768),
        "sample_steps": int(steps or 30),
        "sample_cfg_scale": float(cfg_scale or 4.0),
        "sample_seed": int(seed or 0),
        "sample_low_vram": bool(low_vram),
    })
    return t("sample_settings_saved")


def refresh_sample_gallery() -> list[str]:
    if not SAMPLES_DIR.exists():
        return []
    return [
        str(item)
        for item in sorted(SAMPLES_DIR.rglob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
    ]


def _read_tail(path: Path, max_lines: int = SAMPLE_LOG_TAIL_LINES) -> str:
    if not path.exists() or not path.is_file():
        return ""
    lines: deque[str] = deque(maxlen=max_lines)
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                lines.append(line.rstrip("\n"))
    except Exception as exc:
        return f"Could not read log tail: {exc}"
    return "\n".join(lines).strip()


def _sample_job_for_disk(job: dict) -> dict:
    return {
        key: value
        for key, value in job.items()
        if not str(key).startswith("_") and isinstance(value, (str, int, float, bool, type(None)))
    }


def _persist_sample_jobs_unlocked() -> None:
    _write_json_atomic(SAMPLE_QUEUE_FILE, [_sample_job_for_disk(job) for job in _sample_jobs[-500:]])


def load_sample_jobs_from_disk() -> None:
    global _sample_job_counter
    rows = _read_json_file(SAMPLE_QUEUE_FILE, [])
    if not isinstance(rows, list):
        return
    loaded: list[dict] = []
    max_id = 0
    for item in rows:
        if not isinstance(item, dict):
            continue
        job = dict(item)
        if str(job.get("status", "")) in {"running", "deferred"}:
            job["status"] = "failed:interrupted"
            job["error"] = "The UI process restarted before this sample job reported completion."
        try:
            max_id = max(max_id, int(job.get("id") or 0))
        except (TypeError, ValueError):
            pass
        loaded.append(job)
    with _sample_lock:
        if _sample_jobs:
            return
        _sample_jobs.extend(loaded[-500:])
        _sample_job_counter = max(_sample_job_counter, max_id)
        _persist_sample_jobs_unlocked()


def _touch_sample_job(job: dict) -> None:
    with _sample_lock:
        if job not in _sample_jobs:
            _assign_sample_job_id_unlocked(job)
            _sample_jobs.append(job)
        _persist_sample_jobs_unlocked()


def _sample_status_label(job: dict) -> str:
    status = str(job.get("status", ""))
    if status == "running":
        return t("sample_status_running")
    elif status == "deferred":
        return t("sample_status_deferred")
    elif status == "done":
        return t("sample_status_done")
    elif status.startswith("failed"):
        display_status = t("sample_status_failed")
        if job.get("returncode") is not None:
            display_status += f":{job['returncode']}"
        return display_status
    return status


def _sample_job_elapsed(job: dict) -> str:
    started = job.get("_started_monotonic")
    finished = job.get("_finished_monotonic")
    if started is None:
        return ""
    if finished is None and str(job.get("status", "")) == "running":
        finished = time.monotonic()
    if finished is None:
        return ""
    try:
        return format_sample_elapsed(float(finished) - float(started))
    except (TypeError, ValueError):
        return ""


def _sample_prompt_preview(prompt: str, limit: int = 120) -> str:
    text = " ".join(str(prompt or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "..."


def _sample_queue_row(job: dict) -> dict:
    return {
        "id": job.get("id", ""),
        "created_at": job.get("created_at", ""),
        "status": _sample_status_label(job),
        "elapsed": _sample_job_elapsed(job),
        "output": job.get("output", ""),
        "log_path": job.get("log_path", ""),
        "lora": job.get("lora", ""),
        "seed": job.get("seed", ""),
        "size": f"{job.get('width', '')}x{job.get('height', '')}",
        "steps": job.get("steps", ""),
        "cfg_scale": job.get("cfg_scale", ""),
        "prompt": _sample_prompt_preview(job.get("prompt", "")),
    }


def refresh_sample_queue_table() -> list[list]:
    with _sample_lock:
        rows = [dict(job) for job in reversed(_sample_jobs[-100:])]
    return _rows([_sample_queue_row(row) for row in rows], SAMPLE_QUEUE_HEADERS)


def refresh_sample_queue_ui(status: str | None = None):
    return status or _sample_job_status(), refresh_sample_queue_table(), refresh_sample_gallery()


def _format_sample_job(job: dict) -> str:
    status = str(job.get("status", ""))
    lines = [f"**#{job.get('id', '?')} | {job.get('created_at', '')} | {_sample_status_label(job)}**"]
    elapsed = _sample_job_elapsed(job)
    if elapsed:
        lines.append(f"{t('sample_status_elapsed')}: `{elapsed}`")
    if job.get("output"):
        lines.append(f"{t('sample_status_output')}: `{job['output']}`")
    if job.get("log_path"):
        lines.append(f"{t('sample_status_log')}: `{job['log_path']}`")
    if job.get("error") and str(job.get("error")) != str(job.get("log_path", "")):
        lines.append(str(job["error"]))

    if status.startswith("failed"):
        tail = job.get("log_tail") or _read_tail(Path(job.get("log_path", "")))
        if not tail and job.get("traceback"):
            tail = str(job["traceback"])
        if tail:
            safe_tail = str(tail).replace("```", "` ` `")
            lines.append(f"{t('sample_status_error_tail')}:\n```text\n{safe_tail}\n```")
    return "\n".join(lines)


def _sample_job_status() -> str:
    with _sample_lock:
        rows = list(_sample_jobs[-20:])
    if not rows:
        return t("sample_status_idle")
    return "\n\n".join(_format_sample_job(row) for row in rows)


def clear_finished_sample_jobs():
    with _sample_lock:
        before = len(_sample_jobs)
        _sample_jobs[:] = [
            job for job in _sample_jobs
            if not is_terminal_sample_status(str(job.get("status", "")))
        ]
        live_ids = {id(job) for job in _sample_jobs}
        _pending_sample_jobs[:] = [job for job in _pending_sample_jobs if id(job) in live_ids]
        removed = before - len(_sample_jobs)
        _persist_sample_jobs_unlocked()
    return refresh_sample_queue_ui(t("sample_queue_cleared", count=removed))


def retry_latest_failed_sample():
    with _sample_lock:
        failed_jobs = [
            job for job in reversed(_sample_jobs)
            if str(job.get("status", "")).startswith("failed")
        ]
        source = dict(failed_jobs[0]) if failed_jobs else None
    if not source:
        return refresh_sample_queue_ui(t("sample_retry_no_failed"))
    job, message = _build_sample_job(
        lora_path=str(source.get("lora", "")),
        prompt=str(source.get("prompt", "")),
        negative_prompt=str(source.get("negative_prompt", "")),
        width=int(source.get("width") or 768),
        height=int(source.get("height") or 768),
        steps=int(source.get("steps") or 30),
        cfg_scale=float(source.get("cfg_scale") or 4.0),
        seed=int(source.get("seed") or 0),
        low_vram=bool(source.get("low_vram", False)),
        base_model=str(source.get("base_model") or load_config().get("base_model", "anima-base-v1.0")),
        diffsynth_dir=str(source.get("cwd") or load_config().get("diffsynth_dir", "")),
    )
    if job is None:
        return refresh_sample_queue_ui(message)
    _register_sample_job(job)
    threading.Thread(target=_run_sample_job, args=(job,), daemon=True).start()
    return refresh_sample_queue_ui(t("sample_retry_started", old_id=source.get("id", "?")))


def _find_latest_lora(output_dir: str) -> str:
    files = scan_output_files(output_dir)
    return files[0]["path"] if files else ""


def _lora_output_signature(row: dict) -> str:
    return f"{row.get('path', '')}|{row.get('modified', '')}|{row.get('size_mb', '')}"


def _seen_lora_signatures(output_dir: str) -> set[str]:
    return {
        _lora_output_signature(row)
        for row in scan_output_files(output_dir)
        if row.get("path")
    }


def _find_next_unseen_lora(output_dir: str, seen_loras: set[str]) -> tuple[str, str]:
    # Oldest first pairs pending epoch requests with the checkpoint that appeared next.
    now = time.time()
    for row in reversed(scan_output_files(output_dir)):
        signature = _lora_output_signature(row)
        try:
            file_age = now - float(row.get("modified", 0) or 0)
        except (TypeError, ValueError):
            file_age = SAMPLE_LORA_STABLE_SECONDS
        if (
            row.get("path")
            and signature not in seen_loras
            and float(row.get("size_mb", 0) or 0) > 0
            and file_age >= SAMPLE_LORA_STABLE_SECONDS
        ):
            return str(row["path"]), signature
    return "", ""


def _build_sample_job(
    lora_path: str,
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    steps: int,
    cfg_scale: float,
    seed: int,
    low_vram: bool,
    base_model: str,
    diffsynth_dir: str | Path = "",
) -> tuple[dict | None, str]:
    lora_path = (lora_path or "").strip() or _find_latest_lora(load_config().get("output_directory", ""))
    if not lora_path:
        return None, t("sample_no_lora")

    dit_model = get_dit_model_path(base_model)
    required_paths = {
        "LoRA": Path(lora_path),
        "DiT": dit_model,
        "Qwen3": QWEN3_MODEL,
        "VAE": VAE_MODEL,
    }
    ds_dir = resolve_diffsynth_dir(str(diffsynth_dir or load_config().get("diffsynth_dir", "")))
    support_paths = {
        spec.label: diffsynth_support_core.support_path(spec, ds_dir)
        for spec in diffsynth_support_core.SUPPORT_SPECS
    }
    required_paths.update(support_paths)
    missing = [
        f"{name}: {path}"
        for name, path in required_paths.items()
        if not diffsynth_support_core.directory_has_files(path)
    ]
    if missing:
        return None, t("sample_missing_files", files="\n".join(missing))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output = SAMPLES_DIR / Path(lora_path).stem / f"sample_{timestamp}.png"
    log_path = output.with_suffix(".log")
    job = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": "running",
        "lora": lora_path,
        "dit": str(dit_model),
        "qwen3": str(QWEN3_MODEL),
        "vae": str(VAE_MODEL),
        "qwen_tokenizer": str(support_paths.get("DiffSynth:Qwen tokenizer", "")),
        "sd35_tokenizer": str(support_paths.get("DiffSynth:SD3.5 tokenizer_3", "")),
        "output": str(output),
        "log_path": str(log_path),
        "prompt": prompt or "",
        "negative_prompt": negative_prompt or "",
        "width": int(width),
        "height": int(height),
        "steps": int(steps),
        "cfg_scale": float(cfg_scale),
        "seed": int(seed),
        "low_vram": bool(low_vram),
        "base_model": base_model,
        "cwd": str(ds_dir),
    }
    return job, t("sample_started", path=str(output))


def _assign_sample_job_id_unlocked(job: dict) -> None:
    global _sample_job_counter
    if job.get("id"):
        return
    _sample_job_counter += 1
    job["id"] = _sample_job_counter


def _register_sample_job(job: dict) -> None:
    with _sample_lock:
        _assign_sample_job_id_unlocked(job)
        if job not in _sample_jobs:
            _sample_jobs.append(job)
        _persist_sample_jobs_unlocked()


def _queue_sample_job(job: dict) -> None:
    with _sample_lock:
        _assign_sample_job_id_unlocked(job)
        job["status"] = "deferred"
        if job not in _sample_jobs:
            _sample_jobs.append(job)
        if job not in _pending_sample_jobs:
            _pending_sample_jobs.append(job)
        _persist_sample_jobs_unlocked()


def _pop_pending_sample_jobs() -> list[dict]:
    with _sample_lock:
        jobs = list(_pending_sample_jobs)
        _pending_sample_jobs.clear()
    return jobs


def _run_sample_job(job: dict) -> None:
    _register_sample_job(job)
    job["status"] = "running"
    job["started_at"] = datetime.now().isoformat(timespec="seconds")
    job["_started_monotonic"] = time.monotonic()
    job.pop("_finished_monotonic", None)
    _touch_sample_job(job)
    log_path = Path(job.get("log_path") or Path(job["output"]).with_suffix(".log"))
    cwd = Path(job.get("cwd") or ROOT)
    cmd = [
        sys.executable,
        str(ANIMA_SAMPLE_SCRIPT),
        "--dit", job["dit"],
        "--qwen3", job["qwen3"],
        "--vae", job["vae"],
        "--qwen_tokenizer", job.get("qwen_tokenizer", ""),
        "--sd35_tokenizer", job.get("sd35_tokenizer", ""),
        "--lora", job["lora"],
        "--output", job["output"],
        "--prompt", job["prompt"],
        "--negative_prompt", job["negative_prompt"],
        "--width", str(job["width"]),
        "--height", str(job["height"]),
        "--steps", str(job["steps"]),
        "--cfg_scale", str(job["cfg_scale"]),
        "--seed", str(job["seed"]),
    ]
    if job.get("low_vram"):
        cmd.append("--low_vram")
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w", encoding="utf-8", errors="ignore") as log_f:
            log_f.write("Command: " + shlex.join(cmd) + "\n\n")
            log_f.write(f"Working directory: {cwd}\n\n")
            proc = subprocess.run(
                cmd,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                cwd=str(cwd),
                text=True,
            )
        job["returncode"] = proc.returncode
        if proc.returncode == 0 and Path(job["output"]).exists():
            job["status"] = "done"
        elif proc.returncode == 0:
            job["status"] = "failed:missing-output"
            job["error"] = "Sample command exited successfully, but the image file was not created."
        else:
            job["status"] = f"failed:{proc.returncode}"
            job["error"] = str(log_path)
        job["log_tail"] = _read_tail(log_path)
        job["finished_at"] = datetime.now().isoformat(timespec="seconds")
        job["_finished_monotonic"] = time.monotonic()
        _touch_sample_job(job)
    except Exception as exc:
        job["status"] = "failed"
        job["error"] = str(exc)
        job["traceback"] = traceback.format_exc()
        job["finished_at"] = datetime.now().isoformat(timespec="seconds")
        job["_finished_monotonic"] = time.monotonic()
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "a", encoding="utf-8", errors="ignore") as log_f:
                log_f.write("\n" + job["traceback"])
            job["log_tail"] = _read_tail(log_path)
        except Exception:
            pass
        _touch_sample_job(job)


def launch_sample(
    lora_path: str,
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    steps: int,
    cfg_scale: float,
    seed: int,
    low_vram: bool,
    base_model: str,
):
    job, message = _build_sample_job(
        lora_path, prompt, negative_prompt, width, height, steps, cfg_scale, seed, low_vram, base_model
    )
    if job is None:
        return refresh_sample_queue_ui(message)
    _register_sample_job(job)
    threading.Thread(target=_run_sample_job, args=(job,), daemon=True).start()
    return refresh_sample_queue_ui(_format_sample_job(job))


def launch_sample_and_wait(
    lora_path: str,
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    steps: int,
    cfg_scale: float,
    seed: int,
    low_vram: bool,
    base_model: str,
    diffsynth_dir: str = "",
):
    sample_runtime_log: list[str] = []
    cfg = load_config()
    ds_dir = resolve_diffsynth_dir(diffsynth_dir or cfg.get("diffsynth_dir", ""))
    yield refresh_sample_queue_ui(t("sample_runtime_check", path=str(ds_dir)))
    for item in ensure_diffsynth_installed(ds_dir):
        if isinstance(item, tuple) and item and item[0] == "__done__":
            ok = bool(item[1])
            message = str(item[2])
            if not ok:
                sample_runtime_log.append(message)
                yield refresh_sample_queue_ui(t("sample_runtime_failed", err="\n".join(sample_runtime_log[-20:])))
                return
            break
        sample_runtime_log.append(str(item))
        yield refresh_sample_queue_ui("\n".join(sample_runtime_log[-20:]))

    for item in ensure_diffsynth_support_files(ds_dir):
        if isinstance(item, tuple) and item and item[0] == "__done__":
            ok = bool(item[1])
            message = str(item[2])
            if not ok:
                sample_runtime_log.append(message)
                yield refresh_sample_queue_ui(t("sample_runtime_failed", err="\n".join(sample_runtime_log[-20:])))
                return
            break
        sample_runtime_log.append(str(item))
        yield refresh_sample_queue_ui("\n".join(sample_runtime_log[-20:]))

    job, message = _build_sample_job(
        lora_path, prompt, negative_prompt, width, height, steps, cfg_scale, seed, low_vram, base_model, diffsynth_dir
    )
    if job is None:
        yield refresh_sample_queue_ui(message)
        return

    _register_sample_job(job)
    thread = threading.Thread(target=_run_sample_job, args=(job,), daemon=True)
    thread.start()
    yield refresh_sample_queue_ui(_format_sample_job(job))
    while thread.is_alive():
        time.sleep(SAMPLE_POLL_SECONDS)
        yield refresh_sample_queue_ui(_format_sample_job(job))
    yield refresh_sample_queue_ui(_format_sample_job(job))


def _should_auto_sample_epoch(epoch_index: int, cfg: dict) -> bool:
    if not cfg.get("sample_enabled"):
        return False
    every = max(int(cfg.get("sample_every_n_epochs", 1) or 1), 1)
    return epoch_index > 0 and epoch_index % every == 0


def maybe_launch_auto_sample(epoch_index: int, cfg: dict, base_model: str, seen_loras: set[str]) -> str | None:
    if not _should_auto_sample_epoch(epoch_index, cfg):
        return None
    lora_path, signature = _find_next_unseen_lora(cfg.get("output_directory", ""), seen_loras)
    if not lora_path:
        return None

    job, message = _build_sample_job(
        lora_path=lora_path,
        prompt=cfg.get("sample_prompt", ""),
        negative_prompt=cfg.get("sample_negative_prompt", ""),
        width=int(cfg.get("sample_width", 768)),
        height=int(cfg.get("sample_height", 768)),
        steps=int(cfg.get("sample_steps", 30)),
        cfg_scale=float(cfg.get("sample_cfg_scale", 4.0)),
        seed=int(cfg.get("sample_seed", 42)) + epoch_index,
        low_vram=bool(cfg.get("sample_low_vram", False)),
        base_model=base_model,
        diffsynth_dir=cfg.get("diffsynth_dir", ""),
    )
    if job is None:
        return message
    seen_loras.add(signature)
    _register_sample_job(job)
    threading.Thread(target=_run_sample_job, args=(job,), daemon=True).start()
    return t("sample_auto_started", epoch=epoch_index, path=job["output"])


def maybe_launch_pending_auto_samples(
    pending_epochs: set[int],
    cfg: dict,
    base_model: str,
    seen_loras: set[str],
) -> list[str]:
    messages: list[str] = []
    if not pending_epochs:
        return messages
    if not cfg.get("sample_enabled"):
        pending_epochs.clear()
        return messages
    for epoch_index in sorted(list(pending_epochs)):
        message = maybe_launch_auto_sample(epoch_index, cfg, base_model, seen_loras)
        if not message:
            continue
        pending_epochs.discard(epoch_index)
        messages.append(message)
        # One new LoRA file should only be consumed by one epoch request.
        break
    return messages


def run_pending_sample_jobs() -> list[str]:
    messages: list[str] = []
    for job in _pop_pending_sample_jobs():
        messages.append(t("sample_pending_start", path=job["output"]))
        _run_sample_job(job)
        if job.get("status") == "done":
            messages.append(t("sample_pending_done", path=job["output"]))
        else:
            messages.append(t("sample_pending_failed", path=job.get("log_path", "")))
    return messages


# ---------------------------------------------------------------------------
# Training runner (generator — streams logs live to Gradio)
# ---------------------------------------------------------------------------

def _start_training_stream(
    backend: str,
    diffsynth_dir: str,
    custom_config_path: str,
    gpu_index_choice: str,
    num_cpu_threads_per_process: int,
    base_model: str,
    use_tensorboard: bool,
):
    """Generator: yields growing log text as training runs."""
    saved_cfg = load_config()
    log_buffer = TailLogBuffer(saved_cfg.get("log_tail_lines", 500))
    last_emit = 0.0
    backend = (backend or "kohya").lower()

    def emit(line: str, force: bool = False):
        nonlocal last_emit
        log_buffer.append(line)
        now = time.monotonic()
        if force or now - last_emit >= 0.35:
            last_emit = now
            text = log_buffer.text()
            _update_current_training_tail(text)
            return text
        return None

    def emit_force(line: str):
        if line is None:
            text = log_buffer.text()
            _update_current_training_tail(text)
            return text
        return emit(line, force=True)

    # --- Auto-download DiT model if needed ---
    dit_model = get_dit_model_path(base_model)
    if not dit_model.exists():
        url = BASE_MODEL_URLS.get(base_model)
        if not url:
            yield emit_force(t("err_unknown_base_model", name=base_model))
            return
        yield emit_force(t("info_downloading_model", name=base_model))
        yield emit_force(t("info_download_destination", path=str(dit_model)))
        yield emit_force("")
        os.makedirs(dit_model.parent, exist_ok=True)
        try:
            dl_proc = subprocess.Popen(
                ["wget", "-c", "--show-progress", "-O", str(dit_model), url],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1,
            )
            for line in iter(dl_proc.stdout.readline, ""):
                text = emit(line.rstrip("\n"))
                if text is not None:
                    yield text
            dl_proc.wait()
            if dl_proc.returncode != 0:
                yield emit_force(t("err_download_failed", code=dl_proc.returncode))
                return
            yield emit_force(t("info_download_done"))
            yield emit_force("")
        except FileNotFoundError:
            yield emit_force(t("err_wget_missing"))
            return

    threads = max(int(num_cpu_threads_per_process), 1)
    gpu_idx = gpu_index_from_choice(gpu_index_choice)
    tb_logdir = saved_cfg.get("last_tb_logdir", "")
    accelerate_launch = resolve_accelerate_launch_cmd()

    # --- Backend-specific command assembly ---
    if backend == "kohya":
        train_cfg = custom_config_path.strip() if custom_config_path.strip() else saved_cfg.get("last_train_config", "")
        dataset_cfg = saved_cfg.get("last_dataset_config", "")

        if not train_cfg:
            yield emit_force(t("err_no_train_cfg"))
            return
        if not Path(train_cfg).exists():
            yield emit_force(t("err_train_cfg_not_found", path=train_cfg))
            return
        if not dataset_cfg:
            yield emit_force(t("err_no_dataset_cfg"))
            return
        if not Path(dataset_cfg).exists():
            yield emit_force(t("err_dataset_cfg_not_found", path=dataset_cfg))
            return
        if not TRAIN_SCRIPT.exists():
            yield emit_force(t("err_train_script_missing", path=str(TRAIN_SCRIPT)))
            return

        spec = build_kohya_run_spec(
            accelerate_launch=accelerate_launch,
            accelerate_config=ACCELERATE_CONFIG,
            threads=threads,
            gpu_idx=gpu_idx,
            train_script=TRAIN_SCRIPT,
            train_config=train_cfg,
            dataset_config=dataset_cfg,
            cwd=ROOT,
        )
        cmd = spec.command
        loss_writer = None
        cwd = spec.cwd
        train_script_for_preflight = str(TRAIN_SCRIPT)
        metadata_path = spec.metadata_path
        resolved_diffsynth_dir = ""
        manifest_configs = spec.configs

    elif backend == "diffsynth":
        diffsynth_args_path = saved_cfg.get("last_diffsynth_args", "")
        if not diffsynth_args_path or not Path(diffsynth_args_path).exists():
            yield emit_force(t("err_no_train_cfg"))
            return

        ds_dir = resolve_diffsynth_dir(diffsynth_dir or saved_cfg.get("diffsynth_dir", ""))

        try:
            with open(diffsynth_args_path, "r", encoding="utf-8") as f:
                ds_args = json.load(f)
            ds_args = migrate_diffsynth_args_for_anima(ds_args)
            ds_args = diffsynth_core.set_anima_tokenizer_args(
                ds_args,
                tokenizer_path=str(diffsynth_support_path("DiffSynth:Qwen tokenizer", ds_dir)),
                tokenizer_t5xxl_path=str(diffsynth_support_path("DiffSynth:SD3.5 tokenizer_3", ds_dir)),
            )
        except Exception as e:
            yield emit_force(t("err_generate_failed", err=e))
            return

        ds_dir = resolve_diffsynth_dir(diffsynth_dir or saved_cfg.get("diffsynth_dir", ""))

        # Make sure DiffSynth is installed in *this* interpreter — auto-clones
        # and pip-installs if missing. Streams progress to the log box.
        yield emit_force(t("info_diffsynth_check"))
        install_ok = False
        install_msg = ""
        for item in ensure_diffsynth_installed(ds_dir):
            if isinstance(item, tuple) and item and item[0] == "__done__":
                _, install_ok, install_msg = item
            else:
                text = emit(str(item))
                if text is not None:
                    yield text
        if not install_ok:
            yield emit_force(t("err_diffsynth_install_failed", err=install_msg))
            return

        ds_script = ds_dir / DIFFSYNTH_TRAIN_SCRIPT_REL
        if not ds_script.exists():
            yield emit_force(t("err_diffsynth_missing", path=str(ds_dir)))
            return

        expected_total = estimate_progress_total_from_config(backend, saved_cfg)
        expected_epoch_total = estimate_progress_epoch_total_from_config(backend, saved_cfg)
        expected_epochs = int(saved_cfg.get("max_train_epochs", 1) or 1)
        train_entrypoint = str(ds_script)
        train_args = [*ds_args]
        loss_writer = None
        if DIFFSYNTH_TENSORBOARD_WRAPPER.exists():
            train_entrypoint = str(DIFFSYNTH_TENSORBOARD_WRAPPER)
            train_args = [
                "--diffsynth-train-script", str(ds_script),
                "--expected-total", str(expected_total),
                "--expected-epoch-total", str(expected_epoch_total),
                "--expected-epochs", str(expected_epochs),
                *ds_args,
            ]
            if use_tensorboard and tb_logdir:
                train_args[2:2] = ["--tensorboard-logdir", str(tb_logdir)]
        elif use_tensorboard and tb_logdir:
            loss_writer = DiffSynthLossWriter(tb_logdir)

        metadata_path = get_cli_arg_value(ds_args, "--dataset_metadata_path")
        spec = build_diffsynth_run_spec(
            accelerate_launch=accelerate_launch,
            accelerate_config=ACCELERATE_CONFIG,
            threads=threads,
            gpu_idx=gpu_idx,
            train_script=ds_script,
            train_entrypoint=Path(train_entrypoint),
            train_args=train_args,
            args_path=diffsynth_args_path,
            metadata_path=metadata_path,
            cwd=ds_dir,
        )
        cmd = spec.command
        cwd = spec.cwd
        train_script_for_preflight = str(ds_script)
        resolved_diffsynth_dir = str(ds_dir)
        manifest_configs = spec.configs

    else:
        yield emit_force(f"❌ Unknown backend: {backend}")
        return

    yield emit_force(t("info_using_gpu", idx=gpu_idx))
    yield emit_force(t("info_command", cmd=" ".join(shlex.quote(c) for c in cmd)))
    yield emit_force("")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    project_name = saved_cfg.get("project_name", "run")
    log_file_path = LOGS_DIR / f"{project_name}_{backend}_{timestamp}.log"
    output_dir = saved_cfg.get("output_directory", "")
    resume_lora_path = saved_cfg.get("resume_lora_path", "")

    model_paths = {
        "DiT": str(dit_model),
        "Qwen3": str(QWEN3_MODEL),
        "VAE": str(VAE_MODEL),
    }
    if backend == "diffsynth":
        for spec in diffsynth_support_core.SUPPORT_SPECS:
            model_paths[spec.label] = str(diffsynth_support_core.support_path(spec, resolved_diffsynth_dir))
    preflight_checks = run_preflight(
        backend=backend,
        dataset_dir=saved_cfg.get("image_directory", ""),
        output_dir=output_dir,
        model_paths=model_paths,
        train_script=train_script_for_preflight,
        accelerate_cmd=accelerate_launch,
        diffsynth_dir=resolved_diffsynth_dir,
        metadata_path=metadata_path,
        resume_lora_path=resume_lora_path,
    )
    for line in format_preflight_checks(preflight_checks):
        yield emit_force(line)

    try:
        n_images, missing_captions, dataset_warnings = validate_dataset(saved_cfg.get("image_directory", ""))
    except Exception:
        n_images, missing_captions, dataset_warnings = 0, [], []

    manifest_path = create_run_manifest(
        output_dir=output_dir,
        project_name=project_name,
        backend=backend,
        command=cmd,
        dataset={
            "image_directory": saved_cfg.get("image_directory", ""),
            "image_count": n_images,
            "missing_caption_count": len(missing_captions),
            "warnings": dataset_warnings,
            "repeats": int(saved_cfg.get("repeats", 1)),
            "dataset_repeat": int(saved_cfg.get("dataset_repeat", 1)),
            "metadata_path": metadata_path,
        },
        models=model_paths,
        configs=manifest_configs,
        training={
            "base_model": base_model,
            "epochs": int(saved_cfg.get("max_train_epochs", 1)),
            "network_dim": int(saved_cfg.get("network_dim", 1)),
            "network_alpha": int(saved_cfg.get("network_alpha", 1)),
            "learning_rate": float(saved_cfg.get("learning_rate", 0.0)),
            "train_batch_size": int(saved_cfg.get("train_batch_size", 1)),
            "gradient_accumulation_steps": int(saved_cfg.get("gradient_accumulation_steps", 1)),
            "resume_lora_path": resume_lora_path,
        },
        tensorboard={
            "enabled": bool(use_tensorboard),
            "logdir": tb_logdir,
        },
        preflight=[check.to_dict() for check in preflight_checks],
    )
    save_config({"last_run_manifest": manifest_path})
    _set_current_training(
        manifest_path=manifest_path,
        log_file=str(log_file_path),
        status="starting",
        backend=backend,
        project_name=project_name,
    )
    yield emit_force(t("info_manifest_written", path=manifest_path))
    if has_failures(preflight_checks):
        update_run_manifest(
            manifest_path,
            status="preflight_failed",
            exit_code=None,
            log_file=str(log_file_path),
            output_files=list_output_files(output_dir),
        )
        yield emit_force(t("info_preflight_failed"))
        _finish_current_training("preflight_failed")
        return
    yield emit_force("")

    expected_epoch_total = estimate_progress_epoch_total_from_config(backend, saved_cfg)
    progress_tracker = ProgressTracker(
        estimate_progress_total_from_config(backend, saved_cfg),
        expected_epoch_total=expected_epoch_total,
    )
    samples_seen = _seen_lora_signatures(saved_cfg.get("output_directory", ""))
    pending_auto_sample_epochs: set[int] = set()
    last_epoch_seen = 0
    tqdm_epoch_completed = False

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_idx
    env["PYTHONUNBUFFERED"] = "1"

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
            env=env,
            cwd=cwd,
            encoding="utf-8",
            errors="ignore",
            **_process_popen_kwargs(),
        )
        _set_current_training(
            process=process,
            manifest_path=manifest_path,
            log_file=str(log_file_path),
            status="running",
            backend=backend,
            project_name=project_name,
        )
    except FileNotFoundError:
        yield emit_force(t("err_accelerate_missing"))
        if loss_writer:
            loss_writer.close()
        update_run_manifest(
            manifest_path,
            status="failed",
            exit_code=None,
            log_file=str(log_file_path),
            error="accelerate missing",
            output_files=list_output_files(output_dir),
        )
        return

    with open(log_file_path, "w", encoding="utf-8", errors="ignore") as log_f:
        log_f.write(f"Command: {' '.join(cmd)}\n")
        log_f.write(f"Started: {datetime.now().isoformat()}\n\n")
        for line in iter(process.stdout.readline, ""):
            line = line.rstrip("\n")
            log_f.write(line + "\n")
            log_f.flush()
            if loss_writer:
                loss_writer.feed(line)
            progress_line = progress_tracker.feed(line)
            if progress_line:
                log_f.write(progress_line + "\n")
                log_f.flush()
                text = emit(progress_line, force=True)
                if text is not None:
                    yield text
            structured = parse_structured_progress(line)
            if structured and expected_epoch_total:
                try:
                    epoch_current = int(structured.get("epoch_step", 0))
                    epoch_total = int(structured.get("epoch_total", 0))
                    epoch_index = int(structured.get("epoch", 0))
                except (TypeError, ValueError):
                    epoch_current = epoch_total = epoch_index = 0
                if epoch_total == expected_epoch_total and epoch_current >= epoch_total and epoch_index > last_epoch_seen:
                    last_epoch_seen = epoch_index
                    if _should_auto_sample_epoch(epoch_index, saved_cfg):
                        pending_auto_sample_epochs.add(epoch_index)
            parsed = None if structured else parse_tqdm_progress(line)
            if parsed:
                current, total = parsed
                if total and current < total:
                    tqdm_epoch_completed = False
                if (
                    total
                    and current >= total
                    and (not expected_epoch_total or total == expected_epoch_total)
                    and not tqdm_epoch_completed
                ):
                    tqdm_epoch_completed = True
                    last_epoch_seen += 1
                    if _should_auto_sample_epoch(last_epoch_seen, saved_cfg):
                        pending_auto_sample_epochs.add(last_epoch_seen)
            for sample_status in maybe_launch_pending_auto_samples(
                pending_auto_sample_epochs, saved_cfg, base_model, samples_seen
            ):
                log_f.write(sample_status + "\n")
                log_f.flush()
                text = emit(sample_status, force=True)
                if text is not None:
                    yield text
            text = emit(line)
            if text is not None:
                yield text

    exit_code = process.wait()
    stop_requested = bool(_get_current_training().get("stop_requested"))
    if loss_writer:
        loss_writer.close()

    if pending_auto_sample_epochs:
        deadline = time.monotonic() + 15
        while pending_auto_sample_epochs and time.monotonic() < deadline:
            messages = maybe_launch_pending_auto_samples(
                pending_auto_sample_epochs, saved_cfg, base_model, samples_seen
            )
            if messages:
                for sample_status in messages:
                    text = emit(sample_status, force=True)
                    if text is not None:
                        yield text
                continue
            time.sleep(1)

    pending_sample_messages = run_pending_sample_jobs()
    if pending_sample_messages:
        for message in pending_sample_messages:
            text = emit(message, force=True)
            if text is not None:
                yield text

    status = "success" if exit_code == 0 else "cancelled" if stop_requested or exit_code < 0 else "failed"
    update_run_manifest(
        manifest_path,
        status=status,
        exit_code=exit_code,
        log_file=str(log_file_path),
        output_files=list_output_files(output_dir),
    )

    if exit_code == 0:
        yield emit_force(t("info_train_done", output=saved_cfg.get("output_directory", "output dir"), log=str(log_file_path)))
    else:
        yield emit_force(t("info_train_failed", code=exit_code, log=str(log_file_path)))
        try:
            result = subprocess.run(["dmesg", "-T"], capture_output=True, text=True, timeout=5)
            tail = "\n".join(result.stdout.splitlines()[-40:])
            if any(s in tail for s in ("Out of memory", "Killed process", "oom_reaper", "OOM")):
                yield emit_force(t("info_oom_hint"))
        except Exception:
            pass

    _finish_current_training(status)


def _consume_training_stream(args: tuple) -> None:
    try:
        for _ in _start_training_stream(*args):
            pass
    except Exception:
        error = traceback.format_exc()
        state = _get_current_training()
        process = state.get("process")
        if process is not None and process.poll() is None:
            try:
                process.terminate()
            except Exception:
                try:
                    process.kill()
                except Exception:
                    pass
        current_tail = restored_training_log()
        combined = (current_tail + "\n\n" + error).strip() if current_tail else error
        _update_current_training_tail(combined)
        manifest_path = state.get("manifest_path", "")
        if manifest_path:
            try:
                update_run_manifest(
                    manifest_path,
                    status="failed",
                    log_file=state.get("log_file", ""),
                    error=error,
                    output_files=list_output_files(load_config().get("output_directory", "")),
                )
            except Exception:
                pass
        _finish_current_training("failed", error)
        return

    state = _get_current_training()
    if state.get("status") in {"starting", "running", "cancel_requested"}:
        manifest_status = ""
        manifest_path = state.get("manifest_path", "")
        if manifest_path:
            manifest_payload = _read_json_file(Path(manifest_path), {})
            manifest_status = str(manifest_payload.get("status", ""))
        final_status = manifest_status if manifest_status and manifest_status != "created" else "failed"
        if state.get("stop_requested"):
            final_status = "cancelled"
        _finish_current_training(final_status)


def start_training(
    backend: str,
    diffsynth_dir: str,
    custom_config_path: str,
    gpu_index_choice: str,
    num_cpu_threads_per_process: int,
    base_model: str,
    use_tensorboard: bool,
) -> str:
    if _is_training_busy():
        active = _get_current_training()
        pid = getattr(active.get("process"), "pid", "")
        log_text = restored_training_log()
        prefix = t("training_status_already_running", pid=pid)
        return (prefix + "\n\n" + log_text).strip()

    persisted = _load_persisted_training_state()
    if persisted.get("status") in {"starting", "running", "cancel_requested"} and _pid_is_running(persisted.get("pid")):
        log_text = restored_training_log()
        prefix = t("training_status_detached", pid=persisted.get("pid"))
        return (prefix + "\n\n" + log_text).strip()

    cfg = load_config()
    started = datetime.now().isoformat(timespec="seconds")
    message = t(
        "training_started_background",
        project=cfg.get("project_name", ""),
        backend=(backend or cfg.get("backend", "kohya")),
    )
    _set_current_training(
        process=None,
        thread=None,
        manifest_path="",
        log_file="",
        status="starting",
        stop_requested=False,
        started_at=started,
        finished_at="",
        backend=(backend or cfg.get("backend", "kohya")),
        project_name=cfg.get("project_name", ""),
        last_log_tail=message,
        clear_error=True,
    )
    args = (
        backend,
        diffsynth_dir,
        custom_config_path,
        gpu_index_choice,
        num_cpu_threads_per_process,
        base_model,
        use_tensorboard,
    )
    thread = threading.Thread(target=_consume_training_stream, args=(args,), daemon=True)
    _set_current_training(thread=thread)
    thread.start()
    return restored_training_log()


def refresh_training_ui():
    load_sample_jobs_from_disk()
    return restored_config_status(), restored_training_log(), *refresh_sample_queue_ui()


# ---------------------------------------------------------------------------
# TensorBoard server management
# ---------------------------------------------------------------------------

_tb_proc: subprocess.Popen | None = None
_tb_lock = threading.Lock()
_ngrok_tunnels: dict[int, object] = {}
_ngrok_lock = threading.Lock()


def _port_alive(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", int(port))) == 0


def start_ngrok_tunnel(port: int, token: str) -> tuple[str | None, str | None]:
    """Open (or reuse) an ngrok HTTPS tunnel to a local port. Returns (public_url, error)."""
    if not PYNGROK_AVAILABLE:
        return None, t("ngrok_status_no_pyngrok")
    token = (token or "").strip() or os.environ.get("NGROK_AUTHTOKEN", "").strip()
    if not token:
        return None, t("ngrok_status_no_token")

    with _ngrok_lock:
        try:
            _ngrok.set_auth_token(token)
            old = _ngrok_tunnels.pop(int(port), None)
            if old is not None:
                try:
                    _ngrok.disconnect(old.public_url)
                except Exception:
                    pass
            tunnel = _ngrok.connect(int(port), "http", bind_tls=True)
            _ngrok_tunnels[int(port)] = tunnel
            url = tunnel.public_url
            if url.startswith("http://"):
                url = "https://" + url[len("http://"):]
            return url, None
        except Exception as e:
            return None, t("ngrok_status_tunnel_failed", err=e)


def stop_ngrok_tunnels():
    if not PYNGROK_AVAILABLE:
        return
    with _ngrok_lock:
        for port, tunnel in list(_ngrok_tunnels.items()):
            try:
                _ngrok.disconnect(tunnel.public_url)
            except Exception:
                pass
            _ngrok_tunnels.pop(port, None)
        try:
            _ngrok.kill()
        except Exception:
            pass


atexit.register(stop_ngrok_tunnels)


def start_tensorboard(
    log_dir: str, port: int,
    use_ngrok: bool = False, ngrok_token: str = "",
) -> tuple[str, str]:
    """Returns (status_message, iframe_html). When use_ngrok is True, also opens an ngrok tunnel."""
    global _tb_proc
    port = int(port) if port else 6006

    if not log_dir or not log_dir.strip():
        return t("tb_status_no_logdir"), _empty_iframe()
    if not Path(log_dir).exists():
        return t("tb_status_no_logdir"), _empty_iframe()

    # When tunneling, TB must bind to all interfaces so ngrok can reach it.
    bind_host = "0.0.0.0" if use_ngrok else "127.0.0.1"

    with _tb_lock:
        if _tb_proc is None or _tb_proc.poll() is not None:
            try:
                _tb_proc = subprocess.Popen(
                    [sys.executable, "-m", "tensorboard.main",
                     "--logdir", log_dir,
                     "--port", str(port),
                     "--host", bind_host,
                     "--reload_interval", "5"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except FileNotFoundError as e:
                return t("tb_status_failed", err=e), _empty_iframe()

            for _ in range(60):
                if _port_alive(port):
                    break
                time.sleep(0.5)
            else:
                return t("tb_status_failed", err="timeout waiting for port"), _empty_iframe()
            already_running = False
        else:
            already_running = True

    local_url = f"http://127.0.0.1:{port}"
    status_lines: list[str] = []
    if already_running:
        status_lines.append(t("tb_status_already", url=local_url))
    else:
        status_lines.append(t("tb_status_running", url=local_url))

    if use_ngrok:
        public_url, err = start_ngrok_tunnel(port, ngrok_token)
        if public_url:
            status_lines.append(t("ngrok_status_tunnel_open", url=public_url))
            return "\n\n".join(status_lines), _iframe_for(public_url, local_url)
        else:
            status_lines.append(err or t("tb_status_failed", err="ngrok"))
            # Fall back to local URL — useful when running locally even if user accidentally toggled ngrok
            return "\n\n".join(status_lines), _iframe_for(local_url)

    return "\n\n".join(status_lines), _iframe_for(local_url)


def stop_tensorboard() -> tuple[str, str]:
    global _tb_proc
    with _tb_lock:
        if _tb_proc is not None and _tb_proc.poll() is None:
            _tb_proc.terminate()
            try:
                _tb_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                _tb_proc.kill()
        _tb_proc = None
    stop_ngrok_tunnels()
    return t("tb_status_stopped"), _empty_iframe()


def _iframe_for(url: str, alt_url: str | None = None) -> str:
    alt_block = ""
    if alt_url and alt_url != url:
        alt_block = (
            f' · <a href="{alt_url}" target="_blank" rel="noopener noreferrer">'
            f'{t("btn_open_local")}</a>'
        )
    return (
        f'<div style="border:1px solid #ccc;border-radius:6px;overflow:hidden;">'
        f'<iframe src="{url}" width="100%" height="800" style="border:0;"></iframe>'
        f'<div style="padding:6px;font-size:13px;">'
        f'🔗 <a href="{url}" target="_blank" rel="noopener noreferrer">{t("btn_open_tb")}</a>'
        f'{alt_block}'
        f'</div></div>'
    )


def _empty_iframe() -> str:
    return f'<div style="padding:30px;text-align:center;color:#888;">{t("tb_view_placeholder")}</div>'


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def build_ui() -> gr.Blocks:
    load_sample_jobs_from_disk()
    cfg = load_config()
    current_lang = get_lang()

    saved_gpu_idx = str(cfg.get("gpu_index", "0"))
    default_gpu = gpu_choice_from_index(saved_gpu_idx)

    is_diffsynth = cfg.get("backend", "kohya") == "diffsynth"

    with gr.Blocks(title=t("app_title")) as demo:
        gr.Markdown(t("header_markdown"))

        # ── Language + Backend top bar ───────────────────────────────────
        with gr.Row():
            language_dd = gr.Dropdown(
                label=t("language"),
                choices=SUPPORTED_LANGS,
                value=current_lang,
                info=t("language_info"),
                scale=1,
            )
            save_lang_btn = gr.Button(t("btn_save_language"), scale=0)
            backend_radio = gr.Radio(
                label=t("backend"),
                choices=["kohya", "diffsynth"],
                value=cfg.get("backend", "kohya"),
                info=t("backend_info"),
                scale=2,
            )

        language_status = gr.Markdown("")

        # ── Shared state for last-generated config paths ────────────────
        last_train_cfg = gr.State(cfg.get("last_train_config", ""))
        last_dataset_cfg = gr.State(cfg.get("last_dataset_config", ""))
        last_diffsynth_args = gr.State(cfg.get("last_diffsynth_args", ""))
        last_tb_logdir_state = gr.State(cfg.get("last_tb_logdir", ""))

        with gr.Tabs():

            # ================================================================
            # TAB 1 — Training
            # ================================================================
            with gr.Tab(t("tab_training")):

                with gr.Group():
                    gr.Markdown(f"### {t('section_project_paths')}")
                    with gr.Row():
                        project_name = gr.Textbox(label=t("project_name"), value=cfg["project_name"], placeholder="my_lora")
                        gpu_dropdown = gr.Dropdown(label=t("gpu"), choices=GPU_CHOICES, value=default_gpu)
                    with gr.Row():
                        base_model_dropdown = gr.Dropdown(
                            label=t("base_model"),
                            choices=["anima-base-v1.0", "anima-preview3-base", "anima-preview"],
                            value=cfg.get("base_model", "anima-base-v1.0"),
                            info=t("base_model_info"),
                        )
                    image_directory = gr.Textbox(
                        label=t("image_directory"),
                        value=cfg["image_directory"],
                        placeholder="/path/to/my_dataset",
                    )
                    output_directory = gr.Textbox(
                        label=t("output_directory"),
                        value=cfg["output_directory"],
                        placeholder="/path/to/output",
                    )

                with gr.Group():
                    gr.Markdown(f"### {t('section_network')}")
                    with gr.Row():
                        network_dim = gr.Number(label=t("network_dim"), value=cfg["network_dim"], precision=0, minimum=1)
                        network_alpha = gr.Number(label=t("network_alpha"), value=cfg["network_alpha"], precision=0, minimum=1, visible=not is_diffsynth)
                        learning_rate = gr.Number(label=t("learning_rate"), value=cfg["learning_rate"])
                        max_train_epochs = gr.Number(label=t("max_epochs"), value=cfg["max_train_epochs"], precision=0, minimum=1)

                with gr.Group():
                    gr.Markdown(f"### {t('section_dataset')}")
                    with gr.Row():
                        resolution = gr.Number(label=t("resolution"), value=cfg["resolution"], precision=0, minimum=64, visible=not is_diffsynth)
                        repeats = gr.Number(label=t("repeats"), value=cfg["repeats"], precision=0, minimum=1, visible=not is_diffsynth)
                        caption_dropout = gr.Slider(label=t("caption_dropout"), minimum=0.0, maximum=1.0, step=0.05, value=cfg["caption_dropout"], visible=not is_diffsynth)

                gr.Markdown("---")
                gr.Markdown(f"### {t('section_config_training')}")

                with gr.Row():
                    configure_btn = gr.Button(t("btn_configure"), variant="secondary", size="lg")
                    train_btn = gr.Button(t("btn_start"), variant="primary", size="lg")
                    stop_train_btn = gr.Button(t("btn_stop_training"), variant="stop", size="lg")
                    refresh_training_btn = gr.Button(t("btn_refresh_training_status"), variant="secondary", size="lg")

                custom_config_input = gr.Textbox(
                    label=t("override_config_label"),
                    value="",
                    placeholder="/path/to/custom_training_config.toml",
                )

                status_box = gr.Textbox(
                    label=t("status_label"),
                    value=restored_config_status(),
                    lines=14,
                    interactive=False,
                    show_copy_button=True,
                )
                stop_status_md = gr.Markdown("")
                log_box = gr.Textbox(
                    label=t("log_label"),
                    value=restored_training_log(),
                    lines=25,
                    interactive=False,
                    show_copy_button=True,
                    autoscroll=True,
                )

            # ================================================================
            # TAB 2 — Advanced Settings
            # ================================================================
            with gr.Tab(t("tab_advanced")):
                gr.Markdown(
                    "_These settings are applied when you click "
                    f"**{t('btn_configure')}**._"
                )

                # DiffSynth-specific group (visible only when backend == diffsynth)
                with gr.Group(visible=is_diffsynth) as diffsynth_group:
                    gr.Markdown(f"### {t('section_diffsynth')}")
                    gr.Markdown(t("diffsynth_lr_note"))
                    with gr.Row():
                        lora_target_modules_in = gr.Textbox(
                            label=t("lora_target_modules"),
                            value=cfg["lora_target_modules"],
                            info=t("lora_target_modules_info"),
                        )
                    with gr.Row():
                        dataset_repeat_in = gr.Number(label=t("dataset_repeat"), value=cfg["dataset_repeat"], precision=0, minimum=1)
                        max_pixels_in = gr.Number(label=t("max_pixels"), value=cfg["max_pixels"], precision=0, minimum=65536)
                        save_steps_ds_in = gr.Number(
                            label=t("save_steps_ds"), value=cfg["save_steps_ds"],
                            precision=0, minimum=0, info=t("save_steps_ds_info"),
                        )
                    diffsynth_dir_in = gr.Textbox(
                        label=t("diffsynth_dir"),
                        value=cfg["diffsynth_dir"],
                        placeholder=str(DIFFSYNTH_DEFAULT_DIR),
                        info=t("diffsynth_dir_info"),
                    )
                    with gr.Row():
                        refresh_diffsynth_params_btn = gr.Button(t("btn_refresh_diffsynth_params"), variant="secondary")
                    diffsynth_param_table = gr.Dataframe(
                        headers=DIFFSYNTH_PARAMETER_TABLE_HEADERS,
                        value=refresh_diffsynth_param_table(cfg.get("last_diffsynth_args", "")),
                        interactive=False,
                    )

                # Kohya-specific groups (visible only when backend == kohya)
                with gr.Group(visible=not is_diffsynth) as kohya_optimizer_group:
                    gr.Markdown(f"### {t('section_optimizer')}")
                    with gr.Row():
                        optimizer_type = gr.Dropdown(
                            label=t("optimizer"),
                            choices=["AdamW8bit", "AdamW", "Lion", "SGD", "Prodigy"],
                            value=cfg["optimizer_type"],
                        )
                        lr_scheduler = gr.Dropdown(
                            label=t("lr_scheduler"),
                            choices=["cosine_with_restarts", "cosine", "linear", "constant", "constant_with_warmup", "polynomial"],
                            value=cfg["lr_scheduler"],
                        )
                    with gr.Row():
                        lr_scheduler_num_cycles = gr.Number(label=t("lr_scheduler_cycles"), value=cfg["lr_scheduler_num_cycles"], precision=0, minimum=1)
                        lr_warmup_steps = gr.Number(label=t("lr_warmup_steps"), value=cfg["lr_warmup_steps"], precision=0, minimum=0)

                with gr.Group():
                    gr.Markdown(f"### {t('section_batch')}")
                    with gr.Row():
                        train_batch_size = gr.Number(label=t("train_batch_size"), value=cfg["train_batch_size"], precision=0, minimum=1, visible=not is_diffsynth)
                        gradient_accumulation_steps = gr.Number(label=t("grad_accum_steps"), value=cfg["gradient_accumulation_steps"], precision=0, minimum=1)
                        max_grad_norm = gr.Number(label=t("max_grad_norm"), value=cfg["max_grad_norm"], visible=not is_diffsynth)

                with gr.Group():
                    gr.Markdown(f"### {t('section_resume')}")
                    resume_lora_path_in = gr.Textbox(
                        label=t("resume_lora_path"),
                        value=cfg.get("resume_lora_path", ""),
                        info=t("resume_lora_path_info"),
                        placeholder="/path/to/existing_lora.safetensors",
                    )

                with gr.Group(visible=not is_diffsynth) as kohya_saving_group:
                    gr.Markdown(f"### {t('section_saving')}")
                    with gr.Row():
                        save_every_n_epochs = gr.Number(label=t("save_every_n_epochs"), value=cfg["save_every_n_epochs"], precision=0, minimum=1)
                        save_last_n_epochs = gr.Number(label=t("save_last_n"), value=cfg["save_last_n_epochs"], precision=0, minimum=1)

                with gr.Group():
                    gr.Markdown(f"### {t('section_precision')}")
                    with gr.Row():
                        mixed_precision = gr.Dropdown(label=t("mixed_precision"), choices=["bf16", "fp16", "no"], value=cfg["mixed_precision"])
                        vae_chunk_size = gr.Number(label=t("vae_chunk_size"), value=cfg["vae_chunk_size"], precision=0, minimum=1, visible=not is_diffsynth)
                    with gr.Row():
                        gradient_checkpointing = gr.Checkbox(label=t("gradient_checkpointing"), value=cfg["gradient_checkpointing"])
                        cache_latents = gr.Checkbox(label=t("cache_latents"), value=cfg["cache_latents"], visible=not is_diffsynth)
                        cache_text_encoder_outputs = gr.Checkbox(label=t("cache_text_encoder"), value=cfg["cache_text_encoder_outputs"], visible=not is_diffsynth)
                        vae_disable_cache = gr.Checkbox(label=t("vae_disable_cache"), value=cfg["vae_disable_cache"], visible=not is_diffsynth)

                with gr.Group(visible=not is_diffsynth) as kohya_noise_group:
                    gr.Markdown(f"### {t('section_noise')}")
                    with gr.Row():
                        noise_offset = gr.Number(label=t("noise_offset"), value=cfg["noise_offset"])
                        multires_noise_discount = gr.Number(label=t("multires_noise_discount"), value=cfg["multires_noise_discount"])
                        timestep_sampling = gr.Dropdown(label=t("timestep_sampling"), choices=["sigmoid", "uniform", "logit_normal"], value=cfg["timestep_sampling"])
                        discrete_flow_shift = gr.Number(label=t("discrete_flow_shift"), value=cfg["discrete_flow_shift"])

                with gr.Group():
                    gr.Markdown(f"### {t('section_misc')}")
                    with gr.Row():
                        seed = gr.Number(label=t("seed"), value=cfg["seed"], precision=0)
                        num_cpu_threads = gr.Number(label=t("cpu_threads"), value=cfg["num_cpu_threads_per_process"], precision=0, minimum=1)
                        log_tail_lines = gr.Number(label=t("log_tail_lines"), value=cfg["log_tail_lines"], precision=0, minimum=50)

            # ================================================================
            # TAB 3 — TensorBoard
            # ================================================================
            with gr.Tab(t("tab_tensorboard")):
                gr.Markdown(f"### {t('section_tb_settings')}")
                with gr.Row():
                    use_tb_chk = gr.Checkbox(label=t("tb_use"), value=cfg["use_tensorboard"], info=t("tb_use_info"))
                    tb_port_in = gr.Number(label=t("tb_port"), value=cfg["tb_port"], precision=0, minimum=1024)
                tb_logdir_in = gr.Textbox(
                    label=t("tb_logdir"),
                    value=cfg.get("last_tb_logdir", "") or cfg.get("tb_logdir", ""),
                    info=t("tb_logdir_info"),
                    placeholder=str(TB_LOGS_ROOT / "<project>_<timestamp>"),
                )

                with gr.Group():
                    gr.Markdown(f"### {t('section_sharing')}")
                    if IS_COLAB:
                        gr.Markdown(t("info_colab_detected"))
                    if not PYNGROK_AVAILABLE:
                        gr.Markdown(t("ngrok_status_no_pyngrok"))
                    ngrok_enable_chk = gr.Checkbox(
                        label=t("ngrok_enable"),
                        value=bool(cfg.get("ngrok_enable", False)) or IS_COLAB,
                        info=t("ngrok_enable_info"),
                    )
                    ngrok_token_in = gr.Textbox(
                        label=t("ngrok_token"),
                        value=cfg.get("ngrok_token", ""),
                        type="password",
                        info=t("ngrok_token_info"),
                        placeholder="2x...your token...",
                    )

                with gr.Row():
                    start_tb_btn = gr.Button(t("btn_start_tb"), variant="primary")
                    stop_tb_btn = gr.Button(t("btn_stop_tb"), variant="stop")
                tb_status_md = gr.Markdown(t("tb_status_stopped"))
                tb_iframe = gr.HTML(_empty_iframe())

            # ================================================================
            # TAB 4 — Samples
            # ================================================================
            with gr.Tab(t("tab_samples")):
                gr.Markdown(f"### {t('section_sample_settings')}")
                with gr.Row():
                    sample_enabled = gr.Checkbox(label=t("sample_enabled"), value=cfg["sample_enabled"], info=t("sample_enabled_info"))
                    sample_every_n_epochs = gr.Number(label=t("sample_every_n_epochs"), value=cfg["sample_every_n_epochs"], precision=0, minimum=1)
                    sample_low_vram = gr.Checkbox(label=t("sample_low_vram"), value=cfg["sample_low_vram"], info=t("sample_low_vram_info"))
                sample_prompt = gr.Textbox(label=t("sample_prompt"), value=cfg["sample_prompt"], lines=4)
                sample_negative_prompt = gr.Textbox(label=t("sample_negative_prompt"), value=cfg["sample_negative_prompt"], lines=2)
                with gr.Row():
                    sample_width = gr.Number(label=t("sample_width"), value=cfg["sample_width"], precision=0, minimum=256)
                    sample_height = gr.Number(label=t("sample_height"), value=cfg["sample_height"], precision=0, minimum=256)
                    sample_steps = gr.Number(label=t("sample_steps"), value=cfg["sample_steps"], precision=0, minimum=1)
                    sample_cfg_scale = gr.Number(label=t("sample_cfg_scale"), value=cfg["sample_cfg_scale"])
                    sample_seed = gr.Number(label=t("sample_seed"), value=cfg["sample_seed"], precision=0)
                with gr.Row():
                    save_sample_btn = gr.Button(t("btn_save_sample_settings"), variant="secondary")
                    refresh_samples_btn = gr.Button(t("btn_refresh_samples"))
                sample_status_md = gr.Markdown(_sample_job_status())

                gr.Markdown(f"### {t('section_sample_manual')}")
                sample_lora_path = gr.Textbox(label=t("sample_lora_path"), value="", placeholder="/path/to/lora.safetensors")
                with gr.Row():
                    use_latest_lora_btn = gr.Button(t("btn_use_latest_lora"))
                    run_sample_btn = gr.Button(t("btn_generate_sample"), variant="primary")
                    refresh_sample_status_btn = gr.Button(t("btn_refresh_sample_status"))

                gr.Markdown(f"### {t('section_sample_queue')}")
                with gr.Row():
                    refresh_sample_queue_btn = gr.Button(t("btn_refresh_sample_queue"))
                    clear_sample_queue_btn = gr.Button(t("btn_clear_sample_queue"), variant="secondary")
                    retry_failed_sample_btn = gr.Button(t("btn_retry_failed_sample"), variant="secondary")
                sample_queue_table = gr.Dataframe(
                    headers=SAMPLE_QUEUE_HEADERS,
                    value=refresh_sample_queue_table(),
                    interactive=False,
                )
                sample_gallery = gr.Gallery(label=t("sample_gallery"), value=refresh_sample_gallery(), columns=4, height=520)

            # ================================================================
            # TAB 5 — Models
            # ================================================================
            with gr.Tab(t("tab_models")):
                gr.Markdown(f"### {t('section_models')}")
                with gr.Row():
                    refresh_models_btn = gr.Button(t("btn_refresh_models"))
                    download_models_btn = gr.Button(t("btn_download_missing_models"), variant="primary")
                model_table = gr.Dataframe(
                    headers=MODEL_TABLE_HEADERS,
                    value=refresh_model_table(cfg.get("base_model", "anima-base-v1.0"), cfg.get("diffsynth_dir", "")),
                    interactive=False,
                )
                model_log_box = gr.Textbox(label=t("model_log"), lines=10, interactive=False, show_copy_button=True)

            # ================================================================
            # TAB 6 — Runs
            # ================================================================
            with gr.Tab(t("tab_history")):
                gr.Markdown(f"### {t('section_history')}")
                refresh_history_btn = gr.Button(t("btn_refresh_history"))
                history_table = gr.Dataframe(
                    headers=HISTORY_TABLE_HEADERS,
                    value=refresh_history_table(),
                    interactive=False,
                )

            # ================================================================
            # TAB 7 — Outputs
            # ================================================================
            with gr.Tab(t("tab_outputs")):
                gr.Markdown(f"### {t('section_outputs')}")
                outputs_dir_input = gr.Textbox(
                    label=t("outputs_dir"),
                    value=cfg.get("output_directory", ""),
                    placeholder="/path/to/output",
                )
                with gr.Row():
                    refresh_outputs_btn = gr.Button(t("btn_refresh_outputs"))
                    latest_output_btn = gr.Button(t("btn_latest_output_to_sample"))
                latest_output_md = gr.Markdown("")
                outputs_table = gr.Dataframe(
                    headers=OUTPUT_TABLE_HEADERS,
                    value=refresh_outputs_table(cfg.get("output_directory", "")),
                    interactive=False,
                )

        # ── Backend visibility toggling ──────────────────────────────────
        def _toggle_backend(backend_value: str):
            ds = backend_value == "diffsynth"
            return (
                gr.update(visible=ds),         # diffsynth_group
                gr.update(visible=not ds),     # network_alpha
                gr.update(visible=not ds),     # resolution
                gr.update(visible=not ds),     # repeats
                gr.update(visible=not ds),     # caption_dropout
                gr.update(visible=not ds),     # kohya_optimizer_group
                gr.update(visible=not ds),     # kohya_saving_group
                gr.update(visible=not ds),     # kohya_noise_group
                gr.update(visible=not ds),     # train_batch_size
                gr.update(visible=not ds),     # max_grad_norm
                gr.update(visible=not ds),     # vae_chunk_size
                gr.update(visible=not ds),     # cache_latents
                gr.update(visible=not ds),     # cache_text_encoder
                gr.update(visible=not ds),     # vae_disable_cache
            )

        backend_radio.change(
            fn=_toggle_backend,
            inputs=[backend_radio],
            outputs=[
                diffsynth_group, network_alpha, resolution, repeats, caption_dropout,
                kohya_optimizer_group, kohya_saving_group, kohya_noise_group,
                train_batch_size, max_grad_norm, vae_chunk_size, cache_latents,
                cache_text_encoder_outputs, vae_disable_cache,
            ],
        )

        # ── Language save ────────────────────────────────────────────────
        def _save_language(lang: str):
            if lang in SUPPORTED_LANGS:
                set_lang(lang)
                return t("language_saved", lang=lang)
            return ""

        save_lang_btn.click(fn=_save_language, inputs=[language_dd], outputs=[language_status])

        # ── Input groups ─────────────────────────────────────────────────
        adv_inputs = [
            optimizer_type, lr_scheduler, lr_scheduler_num_cycles, lr_warmup_steps,
            train_batch_size, gradient_accumulation_steps, max_grad_norm,
            save_every_n_epochs, save_last_n_epochs, mixed_precision,
            gradient_checkpointing, seed, noise_offset, multires_noise_discount,
            timestep_sampling, discrete_flow_shift,
            cache_latents, cache_text_encoder_outputs, vae_chunk_size, vae_disable_cache,
            num_cpu_threads, log_tail_lines,
        ]
        basic_inputs = [
            project_name, base_model_dropdown, image_directory, output_directory,
            network_dim, network_alpha, learning_rate, max_train_epochs,
            resolution, repeats, caption_dropout, gpu_dropdown,
        ]
        diffsynth_inputs = [lora_target_modules_in, dataset_repeat_in, max_pixels_in, save_steps_ds_in]
        tb_inputs = [use_tb_chk, tb_logdir_in, tb_port_in]

        restore_outputs = [
            language_dd, backend_radio,
            project_name, gpu_dropdown, base_model_dropdown, image_directory, output_directory,
            network_dim, network_alpha, learning_rate, max_train_epochs,
            resolution, repeats, caption_dropout,
            custom_config_input, status_box, stop_status_md, log_box,
            diffsynth_group, lora_target_modules_in, dataset_repeat_in, max_pixels_in, save_steps_ds_in,
            diffsynth_dir_in, diffsynth_param_table,
            kohya_optimizer_group, optimizer_type, lr_scheduler, lr_scheduler_num_cycles, lr_warmup_steps,
            train_batch_size, gradient_accumulation_steps, max_grad_norm,
            kohya_saving_group, save_every_n_epochs, save_last_n_epochs,
            mixed_precision, gradient_checkpointing, seed, noise_offset, multires_noise_discount,
            timestep_sampling, discrete_flow_shift,
            cache_latents, cache_text_encoder_outputs, vae_chunk_size, vae_disable_cache,
            kohya_noise_group,
            resume_lora_path_in, num_cpu_threads, log_tail_lines,
            use_tb_chk, tb_logdir_in, tb_port_in, ngrok_enable_chk, ngrok_token_in,
            sample_enabled, sample_every_n_epochs, sample_low_vram,
            sample_prompt, sample_negative_prompt, sample_width, sample_height,
            sample_steps, sample_cfg_scale, sample_seed, sample_status_md,
            sample_lora_path, sample_queue_table, sample_gallery,
            model_table, model_log_box, history_table,
            outputs_dir_input, latest_output_md, outputs_table,
            last_train_cfg, last_dataset_cfg, last_diffsynth_args, last_tb_logdir_state,
        ]

        def _restore_ui_state():
            latest_cfg = load_config()
            ds = latest_cfg.get("backend", "kohya") == "diffsynth"
            status, log, sample_status, sample_queue, gallery = refresh_training_ui()
            return (
                get_lang(), latest_cfg.get("backend", "kohya"),
                latest_cfg["project_name"], gpu_choice_from_index(latest_cfg.get("gpu_index", "0")),
                latest_cfg.get("base_model", "anima-base-v1.0"),
                latest_cfg["image_directory"], latest_cfg["output_directory"],
                latest_cfg["network_dim"], gr.update(value=latest_cfg["network_alpha"], visible=not ds),
                latest_cfg["learning_rate"], latest_cfg["max_train_epochs"],
                gr.update(value=latest_cfg["resolution"], visible=not ds),
                gr.update(value=latest_cfg["repeats"], visible=not ds),
                gr.update(value=latest_cfg["caption_dropout"], visible=not ds),
                "", status, "", log,
                gr.update(visible=ds), latest_cfg["lora_target_modules"], latest_cfg["dataset_repeat"],
                latest_cfg["max_pixels"], latest_cfg["save_steps_ds"], latest_cfg["diffsynth_dir"],
                refresh_diffsynth_param_table(latest_cfg.get("last_diffsynth_args", "")),
                gr.update(visible=not ds), latest_cfg["optimizer_type"], latest_cfg["lr_scheduler"],
                latest_cfg["lr_scheduler_num_cycles"], latest_cfg["lr_warmup_steps"],
                gr.update(value=latest_cfg["train_batch_size"], visible=not ds),
                latest_cfg["gradient_accumulation_steps"],
                gr.update(value=latest_cfg["max_grad_norm"], visible=not ds),
                gr.update(visible=not ds), latest_cfg["save_every_n_epochs"], latest_cfg["save_last_n_epochs"],
                latest_cfg["mixed_precision"], latest_cfg["gradient_checkpointing"], latest_cfg["seed"],
                latest_cfg["noise_offset"], latest_cfg["multires_noise_discount"],
                latest_cfg["timestep_sampling"], latest_cfg["discrete_flow_shift"],
                gr.update(value=latest_cfg["cache_latents"], visible=not ds),
                gr.update(value=latest_cfg["cache_text_encoder_outputs"], visible=not ds),
                gr.update(value=latest_cfg["vae_chunk_size"], visible=not ds),
                gr.update(value=latest_cfg["vae_disable_cache"], visible=not ds),
                gr.update(visible=not ds),
                latest_cfg.get("resume_lora_path", ""), latest_cfg["num_cpu_threads_per_process"],
                latest_cfg["log_tail_lines"],
                latest_cfg["use_tensorboard"], latest_cfg.get("last_tb_logdir", "") or latest_cfg.get("tb_logdir", ""),
                latest_cfg["tb_port"], latest_cfg.get("ngrok_enable", False), latest_cfg.get("ngrok_token", ""),
                latest_cfg["sample_enabled"], latest_cfg["sample_every_n_epochs"], latest_cfg["sample_low_vram"],
                latest_cfg["sample_prompt"], latest_cfg["sample_negative_prompt"], latest_cfg["sample_width"],
                latest_cfg["sample_height"], latest_cfg["sample_steps"], latest_cfg["sample_cfg_scale"],
                latest_cfg["sample_seed"], sample_status,
                "", sample_queue, gallery,
                refresh_model_table(latest_cfg.get("base_model", "anima-base-v1.0"), latest_cfg.get("diffsynth_dir", "")),
                "", refresh_history_table(),
                latest_cfg.get("output_directory", ""), "", refresh_outputs_table(latest_cfg.get("output_directory", "")),
                latest_cfg.get("last_train_config", ""), latest_cfg.get("last_dataset_config", ""),
                latest_cfg.get("last_diffsynth_args", ""), latest_cfg.get("last_tb_logdir", ""),
            )

        # ── Configure Training event ─────────────────────────────────────
        def _configure_training_ui(*args):
            result = configure_training(*args)
            return (*result, refresh_diffsynth_param_table(result[3]))

        configure_btn.click(
            fn=_configure_training_ui,
            inputs=[backend_radio, diffsynth_dir_in] + basic_inputs + adv_inputs + diffsynth_inputs + tb_inputs + [resume_lora_path_in],
            outputs=[status_box, last_train_cfg, last_dataset_cfg, last_diffsynth_args, last_tb_logdir_state, diffsynth_param_table],
        )
        refresh_diffsynth_params_btn.click(fn=refresh_diffsynth_param_table, inputs=[last_diffsynth_args], outputs=[diffsynth_param_table])

        # ── Start Training event ─────────────────────────────────────────
        train_btn.click(
            fn=start_training,
            inputs=[backend_radio, diffsynth_dir_in, custom_config_input, gpu_dropdown, num_cpu_threads, base_model_dropdown, use_tb_chk],
            outputs=[log_box],
        )
        stop_train_btn.click(fn=stop_training, inputs=[], outputs=[stop_status_md])
        refresh_training_btn.click(
            fn=refresh_training_ui,
            inputs=[],
            outputs=[status_box, log_box, sample_status_md, sample_queue_table, sample_gallery],
            show_progress="hidden",
            queue=False,
        )

        # ── Sample controls ──────────────────────────────────────────────
        sample_inputs = [
            sample_enabled, sample_every_n_epochs, sample_prompt, sample_negative_prompt,
            sample_width, sample_height, sample_steps, sample_cfg_scale, sample_seed, sample_low_vram,
        ]
        save_sample_btn.click(
            fn=save_sample_settings,
            inputs=sample_inputs,
            outputs=[sample_status_md],
        )
        refresh_samples_btn.click(fn=refresh_sample_queue_ui, inputs=[], outputs=[sample_status_md, sample_queue_table, sample_gallery])
        refresh_sample_status_btn.click(fn=refresh_sample_queue_ui, inputs=[], outputs=[sample_status_md, sample_queue_table, sample_gallery])
        refresh_sample_queue_btn.click(fn=refresh_sample_queue_ui, inputs=[], outputs=[sample_status_md, sample_queue_table, sample_gallery])
        clear_sample_queue_btn.click(fn=clear_finished_sample_jobs, inputs=[], outputs=[sample_status_md, sample_queue_table, sample_gallery])
        retry_failed_sample_btn.click(fn=retry_latest_failed_sample, inputs=[], outputs=[sample_status_md, sample_queue_table, sample_gallery])
        use_latest_lora_btn.click(fn=latest_output_path, inputs=[outputs_dir_input], outputs=[sample_lora_path])
        run_sample_btn.click(
            fn=launch_sample_and_wait,
            inputs=[
                sample_lora_path, sample_prompt, sample_negative_prompt,
                sample_width, sample_height, sample_steps, sample_cfg_scale,
                sample_seed, sample_low_vram, base_model_dropdown, diffsynth_dir_in,
            ],
            outputs=[sample_status_md, sample_queue_table, sample_gallery],
        )

        # ── Model / history / output management ─────────────────────────
        refresh_models_btn.click(fn=refresh_model_table, inputs=[base_model_dropdown, diffsynth_dir_in], outputs=[model_table])
        download_models_btn.click(fn=download_missing_models, inputs=[base_model_dropdown, diffsynth_dir_in], outputs=[model_table, model_log_box])
        refresh_history_btn.click(fn=refresh_history_table, inputs=[], outputs=[history_table])
        refresh_outputs_btn.click(fn=refresh_outputs_table, inputs=[outputs_dir_input], outputs=[outputs_table])

        def _latest_output_for_ui(output_dir: str):
            latest = latest_output_path(output_dir)
            return latest, (t("latest_output_path", path=latest) if latest else t("sample_no_lora"))

        latest_output_btn.click(
            fn=_latest_output_for_ui,
            inputs=[outputs_dir_input],
            outputs=[sample_lora_path, latest_output_md],
        )

        # ── TensorBoard control ──────────────────────────────────────────
        def _start_tb_handler(logdir, port, state_logdir, use_ngrok, ngrok_token):
            # Prefer explicit input, else fall back to last run's logdir
            effective = (logdir or "").strip() or (state_logdir or "")
            # Persist ngrok prefs so the user doesn't need to retype the token
            save_config({
                "ngrok_enable": bool(use_ngrok),
                "ngrok_token": (ngrok_token or "").strip(),
                "tb_port": int(port) if port else 6006,
            })
            return start_tensorboard(effective, port, use_ngrok=bool(use_ngrok), ngrok_token=ngrok_token or "")

        start_tb_btn.click(
            fn=_start_tb_handler,
            inputs=[tb_logdir_in, tb_port_in, last_tb_logdir_state, ngrok_enable_chk, ngrok_token_in],
            outputs=[tb_status_md, tb_iframe],
        )
        stop_tb_btn.click(fn=stop_tensorboard, inputs=[], outputs=[tb_status_md, tb_iframe])

        demo.load(fn=_restore_ui_state, inputs=[], outputs=restore_outputs, show_progress="hidden", queue=False)
        if hasattr(gr, "Timer"):
            try:
                training_timer = gr.Timer(value=TRAINING_POLL_SECONDS, active=True)
            except TypeError:
                training_timer = gr.Timer(value=TRAINING_POLL_SECONDS)
            training_timer.tick(
                fn=refresh_training_ui,
                inputs=[],
                outputs=[status_box, log_box, sample_status_md, sample_queue_table, sample_gallery],
                show_progress="hidden",
                queue=False,
            )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo = build_ui()
    share_requested = IS_COLAB or os.environ.get("GRADIO_SHARE", "").lower() in ("1", "true", "yes")
    launch_kwargs = {
        "server_name": "0.0.0.0" if share_requested else "127.0.0.1",
        "server_port": 7860,
        "show_error": True,
    }
    if share_requested:
        launch_kwargs["share"] = True
        print(t("info_colab_detected") if IS_COLAB else t("info_gradio_share"))
    demo.launch(**launch_kwargs)
