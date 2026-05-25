from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trainer_core import diffsynth_support


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download DiffSynth Anima tokenizer/support files.")
    parser.add_argument(
        "--diffsynth-dir",
        default="DiffSynth-Studio",
        help="DiffSynth-Studio checkout directory. Defaults to ./DiffSynth-Studio.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    diffsynth_dir = Path(args.diffsynth_dir).resolve()
    if not diffsynth_dir.exists():
        raise FileNotFoundError(f"DiffSynth-Studio directory not found: {diffsynth_dir}")

    for spec in diffsynth_support.SUPPORT_SPECS:
        path = diffsynth_support.support_path(spec, diffsynth_dir)
        if diffsynth_support.is_support_ready(spec, diffsynth_dir):
            print(f"{spec.label}: ready at {path}")
            continue
        print(f"{spec.label}: downloading {spec.model_id} -> {path}")
        diffsynth_support.download_support_model(spec, diffsynth_dir)
        if not diffsynth_support.is_support_ready(spec, diffsynth_dir):
            raise RuntimeError(f"{spec.label}: download finished but support files are still missing at {path}")
        print(f"{spec.label}: done")


if __name__ == "__main__":
    main()
