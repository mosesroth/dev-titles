"""YouTube caption publisher (optional publish path, requirement R7).

One of two delivery paths (the other is the docs-team review bundle,
dev_titles/review.py) — both are available. If the docs team runs the tool and
wants captions live on YouTube, this uploads one caption track per language via
the YouTube Data API v3 `captions.insert`. WebVTT and SRT are both accepted.

Needs channel OAuth credentials (scope `youtube.force-ssl`) and
`pip install 'dev-titles[youtube]'`. Both the google client import and the
service build are lazy so the harness/tests run without the dependency.

Quota note: `captions.insert` costs ~400 units; default daily quota 10,000
(~24 tracks/day) — batch and request more for the back-catalog.
"""

from __future__ import annotations

from pathlib import Path


class YouTubePublisher:
    def __init__(self, *, credentials=None):
        self.credentials = credentials
        self._service = None

    def _svc(self):
        if self.credentials is None:
            raise RuntimeError(
                "YouTubePublisher needs OAuth credentials (scope "
                "'youtube.force-ssl'). Pass credentials=."
            )
        if self._service is None:
            try:
                from googleapiclient.discovery import build
            except ImportError as e:  # pragma: no cover
                raise RuntimeError("Install YouTube deps: pip install 'dev-titles[youtube]'") from e
            self._service = build("youtube", "v3", credentials=self.credentials)
        return self._service

    def insert_caption(
        self,
        video_id: str,
        language: str,
        file_path: str,
        *,
        name: str = "",
        is_draft: bool = False,
    ) -> str:
        """Upload one caption track; returns the created caption resource id."""
        svc = self._svc()  # guards credentials + lazily builds the client first
        if not Path(file_path).is_file():
            raise FileNotFoundError(file_path)
        from googleapiclient.http import MediaFileUpload  # lazy

        body = {
            "snippet": {
                "videoId": video_id,
                "language": language,
                "name": name or language,
                "isDraft": is_draft,
            }
        }
        media = MediaFileUpload(file_path, mimetype="application/octet-stream", resumable=False)
        resp = svc.captions().insert(part="snippet", body=body, media_body=media).execute()
        return resp["id"]

    def list_captions(self, video_id: str) -> list[dict]:
        """Existing caption tracks: [{id, language, name, trackKind, isDraft}] (~50 quota units)."""
        resp = self._svc().captions().list(part="snippet", videoId=video_id).execute()
        return [{"id": it["id"], **it["snippet"]} for it in resp.get("items", [])]

    @staticmethod
    def pick_existing(tracks: list[dict], language: str) -> dict | None:
        """The uploaded (non-ASR) track to REPLACE for `language`, if any. YouTube's auto-generated
        track (trackKind 'asr') is never touched. With several matches, prefer the non-draft one."""
        lang = language.lower()
        cands = [
            t
            for t in tracks
            if (t.get("language") or "").lower().split("-")[0] == lang.split("-")[0]
            and (t.get("trackKind") or "").lower() != "asr"
        ]
        cands.sort(key=lambda t: bool(t.get("isDraft")))
        return cands[0] if cands else None

    def upsert_caption(
        self, video_id: str, language: str, file_path: str, *, name: str = "", dry_run: bool = False
    ) -> dict:
        """Replace the existing uploaded `language` track's content (captions.update, ~450 units),
        else insert a new one (~400). Never creates a duplicate track. Returns {action, id}."""
        if not Path(file_path).is_file():
            raise FileNotFoundError(file_path)
        existing = self.pick_existing(self.list_captions(video_id), language)
        if dry_run:
            return {
                "action": "update" if existing else "insert",
                "id": existing and existing["id"],
                "existing_name": existing and existing.get("name"),
            }
        from googleapiclient.http import MediaFileUpload  # lazy

        media = MediaFileUpload(file_path, mimetype="application/octet-stream", resumable=False)
        if existing:
            body = {"id": existing["id"], "snippet": {"isDraft": False}}
            resp = (
                self._svc().captions().update(part="snippet", body=body, media_body=media).execute()
            )
            return {"action": "update", "id": resp["id"]}
        return {
            "action": "insert",
            "id": self.insert_caption(video_id, language, file_path, name=name),
        }

    def publish_outputs(self, video_id: str, outputs, *, is_draft: bool = False) -> dict[str, str]:
        """Insert a caption track per (language, file_path) pair. Returns
        {language: caption_id}. Accepts (lang, path) tuples or objects with
        `.language` and `.file_path`."""
        result: dict[str, str] = {}
        for item in outputs:
            lang, path = item if isinstance(item, tuple) else (item.language, item.file_path)
            result[lang] = self.insert_caption(video_id, lang, path, is_draft=is_draft)
        return result
