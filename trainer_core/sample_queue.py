def is_terminal_sample_status(status: str) -> bool:
    normalized = (status or "").lower()
    return (
        normalized == "done"
        or normalized in {"cancelled", "canceled"}
        or normalized.startswith("failed")
    )


def format_sample_elapsed(seconds: float | int | None) -> str:
    if seconds is None:
        return ""
    try:
        total = max(int(seconds), 0)
    except (TypeError, ValueError):
        return ""
    minutes, sec = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {sec}s"
    if minutes:
        return f"{minutes}m {sec}s"
    return f"{sec}s"
