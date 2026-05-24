from __future__ import annotations

import csv
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}


def validate_dataset(image_dir: str) -> tuple[int, list[str], list[str]]:
    p = Path(image_dir)
    if not p.exists():
        raise FileNotFoundError(f"Directory not found: {image_dir}")
    if not p.is_dir():
        raise NotADirectoryError(f"Not a directory: {image_dir}")

    all_files = list(p.iterdir())
    image_files = [f for f in all_files if f.suffix.lower() in IMAGE_EXTS and f.is_file()]
    txt_basenames = {f.stem for f in all_files if f.suffix.lower() == ".txt" and f.is_file()}

    missing = [f.name for f in image_files if f.stem not in txt_basenames]
    warnings: list[str] = []
    empty_captions: list[str] = []
    unreadable_captions: list[str] = []

    for img in image_files:
        txt = img.with_suffix(".txt")
        if not txt.exists():
            continue
        try:
            if not txt.read_text(encoding="utf-8").strip():
                empty_captions.append(txt.name)
        except UnicodeDecodeError:
            unreadable_captions.append(txt.name)
        except OSError:
            unreadable_captions.append(txt.name)

    if not image_files:
        warnings.append("No image files found in directory.")
    if missing:
        warnings.append(f"{len(missing)} image(s) are missing caption (.txt) files.")
    if empty_captions:
        warnings.append(f"{len(empty_captions)} caption file(s) are empty.")
    if unreadable_captions:
        warnings.append(f"{len(unreadable_captions)} caption file(s) could not be read as UTF-8.")
    return len(image_files), missing, warnings


def generate_diffsynth_metadata(image_dir: str, output_path: Path) -> tuple[Path, int]:
    """Scan a kohya-style flat dir and write Anima DiffSynth metadata.csv."""
    rows = []
    for img in sorted(Path(image_dir).iterdir()):
        if not img.is_file() or img.suffix.lower() not in IMAGE_EXTS:
            continue
        txt = img.with_suffix(".txt")
        caption = ""
        if txt.exists():
            try:
                caption = txt.read_text(encoding="utf-8").strip().replace("\n", " ")
            except Exception:
                caption = ""
        rows.append({"image": img.name, "prompt": caption})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "prompt"])
        writer.writeheader()
        writer.writerows(rows)
    return output_path, len(rows)


def migrate_diffsynth_metadata_for_anima(metadata_path: Path) -> Path:
    """Convert legacy DiffSynth metadata file_name/text columns to image/prompt."""
    if not metadata_path.exists():
        return metadata_path

    with open(metadata_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    if "image" in fieldnames and "prompt" in fieldnames:
        return metadata_path

    image_key = "image" if "image" in fieldnames else "file_name"
    prompt_key = "prompt" if "prompt" in fieldnames else "text"
    if image_key not in fieldnames or prompt_key not in fieldnames:
        return metadata_path

    migrated_path = metadata_path.with_name(f"{metadata_path.stem}_anima.csv")
    with open(migrated_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "prompt"])
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "image": row.get(image_key, ""),
                "prompt": row.get(prompt_key, ""),
            })
    return migrated_path

