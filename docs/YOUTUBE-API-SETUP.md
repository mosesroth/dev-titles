# YouTube caption publishing: one-time setup for the channel owner

Goal: let the subtitle pipeline replace or add caption tracks (en, ja, zh-Hans) on your YouTube
channel directly, instead of uploading each file by hand in YouTube Studio.

## Why the channel OWNER has to do step 4

YouTube's channel-permissions page lists "YouTube APIs" among the things invited users (Manager,
Editor, Subtitle Editor, and so on) **cannot** access (support.google.com/youtube/answer/9481328).
So the API authorization must come from the channel **owner** account. Being a Manager in
YouTube Studio is not enough. If the channel is still a legacy Brand Account, a Brand Account
owner or manager can authorize instead.

## What the owner does (about 15 minutes, once)

Steps 1–3 can be done from any Google account that is allowed to create Google Cloud projects.
Only step 4 needs the channel owner.

1. **Create a project and enable the API**
   - Go to https://console.cloud.google.com/ → project picker → **New project** (e.g. "Subtitle Publisher").
   - **APIs & Services → Library** → search **YouTube Data API v3** → **Enable**.

2. **Configure the OAuth consent ("Google Auth Platform")**
   - **APIs & Services → OAuth consent screen** (this opens Google Auth Platform).
   - **Branding:** an app name (e.g. "Subtitle Publisher") and a support email.
   - **Audience:** choose user type *Internal* if the account is in a Google Workspace org,
     otherwise *External*. For External, click **Publish app** so the status is **In production**.
     Apps left in *Testing* get refresh tokens that expire after 7 days, which forces a sign-in
     every week. An unverified app is fine for private use; Google only shows an "unverified app"
     warning at sign-in.
   - **Data Access:** add the scope `https://www.googleapis.com/auth/youtube.force-ssl`, which the
     captions API requires for insert/update.

3. **Create the client credentials**
   - **Clients → Create client → Application type: Desktop app** → name it → **Create**.
   - **Download JSON** (`client_secret_….json`). If someone else runs the tool, send it to them
     over a secure channel. The file identifies the app only and grants no access to the channel
     on its own.

4. **Authorize once, as the channel owner** (in person or over screen-share if someone else runs
   the tool)
   - The operator runs the publisher in dry-run mode on their machine, and a browser window opens.
   - The **owner** signs in with the channel-owner Google account, picks the channel, and clicks
     **Allow**. To get past the "unverified app" notice, use *Advanced → Go to …*.
   - The token is saved only on the operator's machine
     (`~/.config/dev_titles/youtube_token.json`, never in the repo), so nothing needs to be sent back.

5. *(Optional)* **Quota:** the default is 10,000 units/day. A caption update costs about 450 units
   and a list about 50, so that allows roughly 20 track updates per day. For bulk re-uploads,
   request more under **APIs & Services → YouTube Data API v3 → Quotas**. The publisher stops
   cleanly when it reaches the budget and resumes the next day.

## What the access allows, and the safeguards

- `youtube.force-ssl` is a broad scope, because Google has no captions-only scope. The tool uses
  only `captions.list`, `captions.update` and `captions.insert`.
- **Dry run by default:** it prints what it would change, and `--apply` is required to write.
- It **replaces** the existing uploaded track for a language, or adds one if none exists. It
  never creates duplicates and never touches YouTube's auto-generated (ASR) track.
- Before replacing a track, compare it with your file first. Reviewers sometimes edit tracks
  directly on YouTube, and an overwrite would lose those edits.
- The owner can revoke access at any time: https://myaccount.google.com/permissions → the app → Remove access.

## Operator commands (after setup)

```
pip install -e '.[youtube]'
PYTHONPATH=src python3 scripts/publish_en_youtube.py --client-secret ~/.config/dev_titles/client_secret.json            # dry run
PYTHONPATH=src python3 scripts/publish_en_youtube.py --client-secret ~/.config/dev_titles/client_secret.json --apply    # write
```

By default the script publishes every `<name>.en.srt` in `--subs-dir` (default
`output/FINAL_SUBTITLES`). It finds each video's ID either from `--video-id NAME=ID` or from a
file containing the video URL under `--meta-dir/<name>/` (default `output`). Progress
goes to `--ledger` (default `output/youtube_publish_log.jsonl`). For example:

```
PYTHONPATH=src python3 scripts/publish_en_youtube.py --client-secret client_secret.json \
    --subs-dir bundle/ --video-id talk=VIDEO_ID
```
