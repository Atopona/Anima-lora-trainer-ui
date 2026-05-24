from __future__ import annotations

import math


def estimate_steps(
    *,
    backend: str,
    n_images: int,
    repeats: int,
    dataset_repeat: int,
    epochs: int,
    train_batch_size: int,
    gradient_accumulation_steps: int,
) -> dict[str, int]:
    batch = max(int(train_batch_size), 1)
    grad = max(int(gradient_accumulation_steps), 1)
    ep = max(int(epochs), 1)
    if backend == "diffsynth":
        progress_per_epoch = int(n_images) * int(dataset_repeat)
        optimizer_per_epoch = math.ceil(progress_per_epoch / grad)
        effective_repeats = int(dataset_repeat)
    else:
        progress_per_epoch = math.ceil((int(n_images) * int(repeats)) / (batch * grad))
        optimizer_per_epoch = progress_per_epoch
        effective_repeats = int(repeats)
    return {
        "effective_repeats": effective_repeats,
        "progress_per_epoch": progress_per_epoch,
        "progress_total": progress_per_epoch * ep,
        "optimizer_per_epoch": optimizer_per_epoch,
        "optimizer_total": optimizer_per_epoch * ep,
    }

