"""Sync check: a translated track that runs a cue or two behind the speech is flagged."""

from dev_titles.model import Cue
from dev_titles.validate.sync import sync_report

NAMES = [
    "Acme",
    "Priya",
    "iPhone",
    "Peak Design",
    "Postgres",
    "Claude",
    "Gemini",
    "Redis",
    "Kotlin",
    "React",
    "Expo",
    "Tauri",
    "Vercel",
    "Hugging",
    "Rock",
    "Maestro",
    "Storybook",
    "Webpack",
    "Sentry",
    "Netlify",
]


def _en():
    return [
        Cue(i * 5.0, i * 5.0 + 4.5, [f"We talked about {n} today."]) for i, n in enumerate(NAMES)
    ]


def test_in_step_track_is_not_flagged():
    zh = [Cue(c.start, c.end, [f"我们今天聊了{n}"]) for c, n in zip(_en(), NAMES)]
    r = sync_report(zh, _en())
    assert r["drifted"] is False and r["median"] == 0


def test_track_two_cues_behind_is_flagged():
    en = _en()
    zh = [
        Cue(en[i + 2].start, en[i + 2].end, [f"我们今天聊了{n}"]) for i, n in enumerate(NAMES[:-2])
    ]
    r = sync_report(zh, en)
    assert r["drifted"] is True and r["median"] >= 10


def test_too_few_anchors_gives_no_verdict():
    zh = [Cue(0, 4.5, ["没有拉丁字母"])]
    assert sync_report(zh, _en())["drifted"] is None
