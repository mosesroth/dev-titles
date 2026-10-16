"""Frame.io V4 ingest + review-asset publish (requirement R7).

STUB. Notes from research:
  - Ingest source media via V4 API (or a Custom Action / webhook trigger).
  - Frame.io has NO caption-track API — captions go back as FILE ASSETS uploaded
    into the project for review, plus a comment linking them.
  - Server-to-server auth uses Adobe IMS and requires the account be managed via
    the Adobe Admin Console (open access item for the user).
"""

from __future__ import annotations


class FrameioClient:
    def __init__(self, *, token=None, account_id: str | None = None):
        self.token = token
        self.account_id = account_id

    def download_asset(self, asset_id: str, dest_path: str) -> str:
        raise NotImplementedError("FrameioClient stub — V4 asset download.")

    def upload_review_asset(self, folder_id: str, file_path: str) -> str:
        raise NotImplementedError(
            "FrameioClient stub — upload captions as review file assets (no "
            "caption-track API in Frame.io)."
        )
