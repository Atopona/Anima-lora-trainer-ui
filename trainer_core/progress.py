from __future__ import annotations

import re


_TQDM_RE = re.compile(r"(?P<current>\d+)\s*/\s*(?P<total>\d+)")
_LOSS_RE = re.compile(r"loss[=:\s]+([0-9]*\.?[0-9]+(?:[eE][+-]?\d+)?)")


def parse_tqdm_progress(line: str) -> tuple[int, int] | None:
    match = _TQDM_RE.search(line or "")
    if not match:
        return None
    current = int(match.group("current"))
    total = int(match.group("total"))
    if total <= 0:
        return None
    return current, total


def parse_loss(line: str) -> float | None:
    match = _LOSS_RE.search(line or "")
    if not match:
        return None
    try:
        value = float(match.group(1))
    except ValueError:
        return None
    if not 0.0 < value < 1e6:
        return None
    return value


class ProgressTracker:
    def __init__(self, expected_total: int = 0):
        self.expected_total = max(int(expected_total or 0), 0)
        self.completed_offset = 0
        self.last_current = 0
        self.last_total = 0
        self.last_display_current = -1

    def feed(self, line: str) -> str | None:
        parsed = parse_tqdm_progress(line)
        if not parsed:
            return None
        current, total = parsed
        if self.last_total and current < self.last_current:
            self.completed_offset += self.last_total

        self.last_current = current
        self.last_total = total
        display_current = self.completed_offset + current
        display_total = self.expected_total if self.expected_total else self.completed_offset + total
        if display_total < display_current:
            display_total = display_current
        if display_current == self.last_display_current:
            return None
        self.last_display_current = display_current
        percent = (display_current / display_total) * 100 if display_total else 0.0
        return f"[progress] {display_current}/{display_total} ({percent:.1f}%)"
