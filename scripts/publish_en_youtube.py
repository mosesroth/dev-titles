#!/usr/bin/env python3
"""Publish improved EN subtitles to your YouTube channel.

Input layout (override with --subs-dir / --meta-dir):
  <subs-dir>/<name>.en.srt                     the subtitle to publish
  <meta-dir>/<name>/SOURCE-URL.txt             any file containing the video URL (…?v=VIDEO_ID),
                                               or a <name>/_ingest/<VIDEO_ID>.* downloaded file
Alternatively pass --video-id name=VIDEO_ID (repeatable) to map names to IDs directly.

For each video: find its existing uploaded EN track and REPLACE its content
(captions.update), or insert one if there is none. Never creates a duplicate track; never touches
YouTube's auto-generated (ASR) track.

SAFE BY DEFAULT: without --apply this is a DRY RUN — it only reads the channel (captions.list) and
prints what it would do. Progress is recorded in a JSONL ledger (--ledger), so a quota stop
(default 10,000 units/day; an update is ~450 + a list ~50) resumes the next day without redoing work.

Credentials (never stored in the repo): a Google OAuth *Desktop* client JSON for a Cloud project with
"YouTube Data API v3" enabled, authorised by the channel OWNER (invited Studio users can't use
the API; see docs/YOUTUBE-API-SETUP.md). The token is cached at
~/.config/dev_titles/youtube_token.json.

  pip install -e '.[youtube]'
  PYTHONPATH=src python3 scripts/publish_en_youtube.py --client-secret ~/Downloads/client_secret.json        # dry run
  PYTHONPATH=src python3 scripts/publish_en_youtube.py --client-secret ~/Downloads/client_secret.json --apply
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

from dev_titles.publish.youtube import YouTubePublisher

SCOPES = ["https://www.googleapis.com/auth/youtube.force-ssl"]
TOKEN = Path.home() / ".config" / "dev_titles" / "youtube_token.json"
LEDGER = Path("output/youtube_publish_log.jsonl")
FINAL = Path("output/FINAL_SUBTITLES")  # default --subs-dir
TICKET = Path("output")  # default --meta-dir
UNITS = {"list": 50, "update": 450, "insert": 400}


def video_id(name: str, meta_dir: Path = TICKET) -> str | None:
    d = meta_dir / name
    for f in (d / "SOURCE-URL.txt", d / "TICKET-DRAFT.md"):
        if f.exists():
            m = re.search(r"v=([A-Za-z0-9_-]{11})", f.read_text(encoding="utf-8"))
            if m:
                return m.group(1)
    if (d / "_ingest").is_dir():
        for f in (d / "_ingest").iterdir():
            m = re.match(r"([A-Za-z0-9_-]{11})\.", f.name)
            if m:
                return m.group(1)
    return None


def credentials(client_secret: str):
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow

    creds = Credentials.from_authorized_user_file(str(TOKEN), SCOPES) if TOKEN.exists() else None
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
    if not creds or not creds.valid:
        creds = InstalledAppFlow.from_client_secrets_file(client_secret, SCOPES).run_local_server(
            port=0
        )
        TOKEN.parent.mkdir(parents=True, exist_ok=True)
        TOKEN.write_text(creds.to_json(), encoding="utf-8")
    return creds


def done_today(ledger: Path = LEDGER) -> set[str]:
    if not ledger.exists():
        return set()
    return {
        json.loads(l)["video"]
        for l in ledger.read_text().splitlines()
        if l.strip() and json.loads(l).get("applied")
    }


def artifact_errors(path: Path) -> list:
    from dev_titles.cli import _read_any
    from dev_titles.validate import validate_subtitle

    return [
        v
        for v in validate_subtitle(_read_any(str(path), "en"))
        if v.rule in ("sound_tag", "caption_artifact")
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--client-secret", required=True)
    ap.add_argument(
        "--apply", action="store_true", help="actually write to YouTube (default: dry run)"
    )
    ap.add_argument("--quota", type=int, default=10000, help="units to spend this run")
    ap.add_argument("--only", nargs="*", help="limit to these video names")
    ap.add_argument("--subs-dir", type=Path, default=FINAL, help="dir of <name>.en.srt files")
    ap.add_argument(
        "--meta-dir", type=Path, default=TICKET, help="dir of <name>/ folders holding the video URL"
    )
    ap.add_argument(
        "--video-id",
        action="append",
        default=[],
        metavar="NAME=ID",
        help="map a subtitle name to its YouTube video ID (repeatable; overrides --meta-dir)",
    )
    ap.add_argument("--ledger", type=Path, default=LEDGER, help="JSONL progress ledger")
    a = ap.parse_args()
    ids = dict(kv.split("=", 1) for kv in a.video_id)

    pub = YouTubePublisher(credentials=credentials(a.client_secret))
    already = done_today(a.ledger) if a.apply else set()
    spent = 0
    for srt in sorted(a.subs_dir.glob("*.en.srt")):
        name = srt.name[: -len(".en.srt")]
        if a.only and name not in a.only:
            continue
        if name in already:
            print(f"  {name:<24} already published — skip")
            continue
        vid = ids.get(name) or video_id(name, a.meta_dir)
        if not vid:
            print(f"  {name:<24} NO VIDEO ID — skip")
            continue
        bad = artifact_errors(srt)
        if bad:  # never publish a track with sound tags / '>>' markers (scripts/check_upload.py)
            print(f"  {name:<24} REFUSED — {len(bad)} cue(s) with {sorted({v.rule for v in bad})}")
            continue
        plan = pub.upsert_caption(vid, "en", str(srt), name="English", dry_run=True)
        spent += UNITS["list"]
        cost = UNITS[plan["action"]]
        if not a.apply:
            print(
                f"  {name:<24} {vid}  would {plan['action'].upper():<6}"
                + (
                    f" existing track '{plan['existing_name']}' ({plan['id']})"
                    if plan["id"]
                    else ""
                )
            )
            continue
        if spent + cost > a.quota:
            print(f"  quota budget reached ({spent} units) — stopping; re-run tomorrow to continue")
            break
        res = pub.upsert_caption(vid, "en", str(srt), name="English")
        spent += cost
        rec = {
            "video": name,
            "id": vid,
            **res,
            "applied": True,
            "at": time.strftime("%Y-%m-%d %H:%M"),
        }
        a.ledger.parent.mkdir(parents=True, exist_ok=True)
        with a.ledger.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"  {name:<24} {vid}  {res['action']}d  (caption {res['id']})")
    print(f"{'APPLIED' if a.apply else 'DRY RUN'} — ~{spent} quota units")


if __name__ == "__main__":
    sys.exit(main())
