"""Timecode <-> seconds helpers for subtitle formats."""

from __future__ import annotations


def seconds_to_ts(seconds: float, sep: str = ".") -> str:
    """Seconds -> HH:MM:SS.mmm (VTT uses '.', SRT uses ',')."""
    if seconds < 0:
        seconds = 0.0
    ms_total = round(seconds * 1000)
    h, rem = divmod(ms_total, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def ts_to_seconds(ts: str) -> float:
    """HH:MM:SS.mmm or HH:MM:SS,mmm -> seconds. Also accepts MM:SS.mmm."""
    ts = ts.strip().replace(",", ".")
    parts = ts.split(":")
    if len(parts) == 3:
        h, m, s = parts
    elif len(parts) == 2:
        h, m, s = "0", parts[0], parts[1]
    else:
        raise ValueError(f"bad timestamp: {ts!r}")
    return int(h) * 3600 + int(m) * 60 + float(s)
