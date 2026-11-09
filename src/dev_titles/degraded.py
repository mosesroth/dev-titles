"""Steps a run skipped after an error, so a degraded run can't read as a clean one.

Eighteen steps catch their own exception and carry on (a failed optional step must never sink an
hours-long run). They used to only print "X skipped (TypeError)": review-and-fix was skipped that
way on two videos, and both runs still reported PASS in REVIEW.md, status.json and the queue
summary. Each such step now calls `note`, and the run reports the list.
"""

from __future__ import annotations

_STEPS: list[str] = []


def note(step: str, exc: BaseException) -> str:
    """Record that `step` was skipped because of `exc`; returns the message to log."""
    msg = f"{step} skipped ({type(exc).__name__})"
    _STEPS.append(msg)
    return msg


def steps() -> list[str]:
    return list(dict.fromkeys(_STEPS))


def reset() -> None:
    _STEPS.clear()


def banner() -> str:
    """'' for a clean run, else the line every report leads with."""
    s = steps()
    return f"DEGRADED RUN — {len(s)} step(s) skipped after an error: {'; '.join(s)}" if s else ""
