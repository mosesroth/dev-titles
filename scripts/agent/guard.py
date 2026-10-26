#!/usr/bin/env python3
"""Deliverables guard for agent runs: `snapshot` hashes (and copies) every .srt/.vtt under
output/FINAL_SUBTITLES, output/UPLOAD_*, output/ticket; `verify` reports any file changed, added or
removed since, and restores changed/removed ones from the copy (an added file is moved aside).

  python3 scripts/agent/guard.py snapshot <guard dir>
  python3 scripts/agent/guard.py verify <guard dir>     # exit 1 if anything was touched
"""

import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

OUTPUT = Path(
    os.environ.get("DEV_TITLES_OUTPUT", Path(__file__).resolve().parents[2] / "output")
).resolve()


def files():
    for root in [OUTPUT / "FINAL_SUBTITLES", OUTPUT / "ticket", *OUTPUT.glob("UPLOAD_*")]:
        if root.is_dir():
            yield from (p for p in root.rglob("*") if p.suffix in (".srt", ".vtt") and p.is_file())


def h(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    cmd, guard = sys.argv[1], Path(sys.argv[2])
    manifest = guard / "manifest.json"
    if cmd == "snapshot":
        guard.mkdir(parents=True, exist_ok=True)
        m = {}
        for p in files():
            rel = str(p.relative_to(OUTPUT))
            m[rel] = h(p)
            dst = guard / "copy" / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dst)
        manifest.write_text(json.dumps(m), encoding="utf-8")
        print(f"guard: {len(m)} deliverable file(s) snapshotted")
        return 0
    m = json.loads(manifest.read_text(encoding="utf-8"))
    now = {str(p.relative_to(OUTPUT)): p for p in files()}
    changed = [r for r, x in m.items() if r in now and h(now[r]) != x]
    removed = [r for r in m if r not in now]
    added = [r for r in now if r not in m]
    for r in changed + removed:
        (OUTPUT / r).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(guard / "copy" / r, OUTPUT / r)
    for r in added:
        dst = guard / "added" / r
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(OUTPUT / r), dst)
    if changed or removed or added:
        print("DELIVERABLES TOUCHED by the agent — restored:")
        for kind, rs in (
            ("changed", changed),
            ("removed", removed),
            ("added (moved aside)", added),
        ):
            for r in rs:
                print(f"  {kind}: {r}")
        return 1
    print("guard: no deliverable was touched")
    return 0


if __name__ == "__main__":
    sys.exit(main())
