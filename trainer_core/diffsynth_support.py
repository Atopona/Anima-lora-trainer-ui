from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DiffSynthSupportSpec:
    label: str
    model_id: str
    relative_path: Path
    download_relative_path: Path
    allow_patterns: tuple[str, ...]


SUPPORT_SPECS: tuple[DiffSynthSupportSpec, ...] = (
    DiffSynthSupportSpec(
        label="DiffSynth:Qwen tokenizer",
        model_id="Qwen/Qwen3-0.6B",
        relative_path=Path("models") / "Qwen" / "Qwen3-0.6B",
        download_relative_path=Path("models") / "Qwen" / "Qwen3-0.6B",
        allow_patterns=(
            "*.json",
            "*.model",
            "tokenizer*",
            "vocab*",
            "merges.txt",
            "special_tokens_map.json",
        ),
    ),
    DiffSynthSupportSpec(
        label="DiffSynth:SD3.5 tokenizer_3",
        model_id="stabilityai/stable-diffusion-3.5-large",
        relative_path=Path("models") / "stabilityai" / "stable-diffusion-3.5-large" / "tokenizer_3",
        download_relative_path=Path("models") / "stabilityai" / "stable-diffusion-3.5-large",
        allow_patterns=("tokenizer_3/*",),
    ),
)


def support_path(spec: DiffSynthSupportSpec, diffsynth_dir: str | Path) -> Path:
    return Path(diffsynth_dir) / spec.relative_path


def support_download_dir(spec: DiffSynthSupportSpec, diffsynth_dir: str | Path) -> Path:
    return Path(diffsynth_dir) / spec.download_relative_path


def path_size_bytes(path: str | Path) -> int:
    item = Path(path)
    if not item.exists():
        return 0
    if item.is_file():
        return item.stat().st_size
    total = 0
    for child in item.rglob("*"):
        if child.is_file():
            try:
                total += child.stat().st_size
            except OSError:
                pass
    return total


def directory_has_files(path: str | Path) -> bool:
    item = Path(path)
    if not item.exists():
        return False
    if item.is_file():
        return True
    return any(child.is_file() for child in item.rglob("*"))


def is_support_ready(spec: DiffSynthSupportSpec, diffsynth_dir: str | Path) -> bool:
    return directory_has_files(support_path(spec, diffsynth_dir))


def download_support_model(spec: DiffSynthSupportSpec, diffsynth_dir: str | Path) -> Path:
    try:
        from modelscope.hub.snapshot_download import snapshot_download
    except Exception:
        from modelscope import snapshot_download

    local_dir = support_download_dir(spec, diffsynth_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    patterns = list(spec.allow_patterns)
    try:
        snapshot_download(
            spec.model_id,
            local_dir=str(local_dir),
            allow_patterns=patterns,
        )
    except TypeError:
        snapshot_download(
            spec.model_id,
            local_dir=str(local_dir),
            allow_file_pattern=patterns,
        )
    return support_path(spec, diffsynth_dir)
