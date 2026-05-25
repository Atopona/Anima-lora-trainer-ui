from __future__ import annotations

import re
import json


_TQDM_RE = re.compile(r"(?P<current>\d+)\s*/\s*(?P<total>\d+)")
_LOSS_RE = re.compile(r"loss[=:\s]+([0-9]*\.?[0-9]+(?:[eE][+-]?\d+)?)")
STRUCTURED_PROGRESS_PREFIX = "__ANIMA_PROGRESS__"


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


def parse_structured_progress(line: str) -> dict | None:
    text = line or ""
    if not text.startswith(STRUCTURED_PROGRESS_PREFIX):
        return None
    payload = text[len(STRUCTURED_PROGRESS_PREFIX):].strip()
    if not payload:
        return None
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    try:
        step = int(data.get("step", 0))
    except (TypeError, ValueError):
        return None
    if step <= 0:
        return None
    return data


class ProgressTracker:
    def __init__(self, expected_total: int = 0, expected_epoch_total: int = 0):
        self.expected_total = max(int(expected_total or 0), 0)
        self.expected_epoch_total = max(int(expected_epoch_total or 0), 0)
        self.completed_offset = 0
        self.last_current = 0
        self.last_total = 0
        self.last_display_current = -1
        self.structured_seen = False

    def accepts(self, current: int, total: int) -> bool:
        if total <= 0:
            return False
        if not self.expected_epoch_total:
            return True
        return total == self.expected_epoch_total or total == self.expected_total

    def feed(self, line: str) -> str | None:
        event = parse_structured_progress(line)
        if event:
            return self.feed_structured(event)
        if self.structured_seen:
            return None
        parsed = parse_tqdm_progress(line)
        if not parsed:
            return None
        current, total = parsed
        if not self.accepts(current, total):
            return None
        if self.last_total and current < self.last_current:
            self.completed_offset += self.expected_epoch_total or self.last_total

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
        if self.expected_epoch_total:
            total_epochs = max((display_total + self.expected_epoch_total - 1) // self.expected_epoch_total, 1)
            epoch_index = min(self.completed_offset // self.expected_epoch_total + 1, total_epochs)
            epoch_percent = (current / total) * 100 if total else 0.0
            return (
                f"[progress] epoch {epoch_index}/{total_epochs}: "
                f"{current}/{total} ({epoch_percent:.1f}%), "
                f"total {display_current}/{display_total} ({percent:.1f}%)"
            )
        return f"[progress] {display_current}/{display_total} ({percent:.1f}%)"

    def feed_structured(self, event: dict) -> str | None:
        self.structured_seen = True
        try:
            display_current = int(event.get("step", 0))
        except (TypeError, ValueError):
            return None
        if display_current <= 0:
            return None

        try:
            event_total = int(event.get("total", 0))
        except (TypeError, ValueError):
            event_total = 0
        display_total = event_total or self.expected_total or display_current
        if display_total < display_current:
            display_total = display_current

        if display_current == self.last_display_current:
            return None
        self.last_display_current = display_current

        source = str(event.get("source") or "structured")
        percent = (display_current / display_total) * 100 if display_total else 0.0

        try:
            epoch_total = int(event.get("epoch_total", 0)) or self.expected_epoch_total
        except (TypeError, ValueError):
            epoch_total = self.expected_epoch_total
        if epoch_total:
            try:
                epoch_current = int(event.get("epoch_step", 0))
            except (TypeError, ValueError):
                epoch_current = 0
            if epoch_current <= 0:
                epoch_current = ((display_current - 1) % epoch_total) + 1
            try:
                epoch_index = int(event.get("epoch", 0))
            except (TypeError, ValueError):
                epoch_index = 0
            if epoch_index <= 0:
                epoch_index = ((display_current - 1) // epoch_total) + 1
            try:
                total_epochs = int(event.get("epochs", 0))
            except (TypeError, ValueError):
                total_epochs = 0
            if total_epochs <= 0:
                total_epochs = max((display_total + epoch_total - 1) // epoch_total, 1)
            epoch_percent = (epoch_current / epoch_total) * 100 if epoch_total else 0.0
            return (
                f"[progress:{source}] epoch {epoch_index}/{total_epochs}: "
                f"{epoch_current}/{epoch_total} ({epoch_percent:.1f}%), "
                f"total {display_current}/{display_total} ({percent:.1f}%)"
            )
        return f"[progress:{source}] {display_current}/{display_total} ({percent:.1f}%)"
