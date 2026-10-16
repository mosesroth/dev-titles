"""AWS Transcribe adapter (primary ASR — procurement-clean, R1).

Two layers:
  - `from_transcribe_json()` — PURE parser: AWS Transcribe output JSON ->
    Transcript with word timing. Testable without AWS.
  - `AwsTranscriber.transcribe()` — live path: upload media to S3, start an async
    transcription job (with optional custom vocabulary for jargon), poll, fetch
    the result JSON, and parse it. Lazy boto3 import; needs AWS creds + a region +
    an S3 bucket. Not exercised in unit tests.
"""

from __future__ import annotations

import json
import time
from typing import Any

from ..model import Transcript, Word
from .base import Transcriber


def from_transcribe_json(data: dict[str, Any], language: str = "en") -> Transcript:
    """Parse AWS Transcribe output JSON into a Transcript.

    Punctuation items (no timing) are appended to the preceding word's text so
    the cue builder / segmenter see sentence-final marks.
    """
    items = data.get("results", {}).get("items", [])
    words: list[Word] = []
    for it in items:
        alt = (it.get("alternatives") or [{}])[0]
        content = alt.get("content", "")
        if not content:
            continue
        if it.get("type") == "punctuation":
            if words:
                words[-1].text += content
            continue
        conf = alt.get("confidence")
        words.append(
            Word(
                text=content,
                start=float(it["start_time"]),
                end=float(it["end_time"]),
                confidence=float(conf) if conf not in (None, "") else None,
            )
        )
    return Transcript(language=language, words=words)


class AwsTranscriber(Transcriber):
    def __init__(
        self,
        *,
        region: str | None = None,
        output_bucket: str | None = None,
        poll_seconds: float = 10.0,
    ):
        self.region = region
        self.output_bucket = output_bucket
        self.poll_seconds = poll_seconds

    def transcribe(self, media_path, *, language="en", custom_vocabulary=None) -> Transcript:
        # Lazy import so the harness/tests don't require boto3.
        import uuid

        try:
            import boto3
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("Install boto3: pip install 'dev-titles[aws]'") from e
        if not self.output_bucket:
            raise RuntimeError("output_bucket is required for AWS Transcribe jobs")

        s3 = boto3.client("s3", region_name=self.region)
        transcribe = boto3.client("transcribe", region_name=self.region)

        key = f"dev-titles/input/{uuid.uuid4().hex}/{media_path.split('/')[-1]}"
        s3.upload_file(media_path, self.output_bucket, key)
        job = f"dev-titles-{uuid.uuid4().hex}"
        params: dict[str, Any] = {
            "TranscriptionJobName": job,
            "LanguageCode": language if "-" in language else f"{language}-US",
            "Media": {"MediaFileUri": f"s3://{self.output_bucket}/{key}"},
            "OutputBucketName": self.output_bucket,
        }
        if custom_vocabulary:
            params["Settings"] = (
                {"VocabularyName": custom_vocabulary} if isinstance(custom_vocabulary, str) else {}
            )
        transcribe.start_transcription_job(**params)

        while True:
            resp = transcribe.get_transcription_job(TranscriptionJobName=job)
            status = resp["TranscriptionJob"]["TranscriptionJobStatus"]
            if status in ("COMPLETED", "FAILED"):
                break
            time.sleep(self.poll_seconds)
        if status == "FAILED":
            raise RuntimeError(
                f"Transcribe job failed: {resp['TranscriptionJob'].get('FailureReason')}"
            )

        out_key = f"{job}.json"
        body = s3.get_object(Bucket=self.output_bucket, Key=out_key)["Body"].read()
        return from_transcribe_json(json.loads(body), language=language)
