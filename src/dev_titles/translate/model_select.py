"""Model selection: auto-pick the newest model in a family so the pipeline
upgrades itself as new Claude models ship, instead of pinning an ID in code.

There is no server-side "latest" alias on the Messages API, so `resolve_model`
lists the account's available models and picks the highest version in a family
(default the Opus family). Falls back to a pinned constant if the Models API is
unavailable. `pick_latest` / `parse_version` are pure and unit-tested.
"""

from __future__ import annotations

# Fallback when the Models API can't be reached. Bump when a newer default ships.
DEFAULT_MODEL = "claude-opus-5"


def parse_version(model_id: str, family: str) -> tuple[int, ...] | None:
    """Version tuple for a model id within `family`, or None if it doesn't match.

    'claude-opus-5' -> (5,); 'claude-opus-4-8' -> (4, 8). Non-numeric / dated
    snapshot suffixes stop the parse (so 'claude-opus-4-5-20251101' -> (4, 5)).
    """
    prefix = family.rstrip("-") + "-"
    if not model_id.startswith(prefix):
        return None
    rest = model_id[len(prefix) :]
    parts: list[int] = []
    for chunk in rest.split("-"):
        # Version components are short (1-4 digits); a long run like a date
        # snapshot (e.g. 20251101) marks the end of the version.
        if chunk.isdigit() and len(chunk) <= 4:
            parts.append(int(chunk))
        else:
            break
    return tuple(parts) or None


def pick_latest(model_ids, family: str = "claude-opus") -> str | None:
    """Return the highest-version id in `family`, or None if none match."""
    best_id, best_ver = None, None
    for mid in model_ids:
        ver = parse_version(mid, family)
        if ver is not None and (best_ver is None or ver > best_ver):
            best_id, best_ver = mid, ver
    return best_id


def resolve_model(
    client,
    requested: str | None = "auto",
    *,
    family: str = "claude-opus",
    fallback: str = DEFAULT_MODEL,
) -> str:
    """Resolve the model id to use. An explicit id passes through; "auto"/None
    queries the Models API and picks the newest in `family`, falling back to
    `fallback` on any error or empty match."""
    if requested and requested != "auto":
        return requested
    try:
        ids = [m.id for m in client.models.list()]
        return pick_latest(ids, family) or fallback
    except Exception:
        return fallback
