"""sync_delivered: fixes made on delivered files are carried back into the run folder."""

import json

from dev_titles import sync_delivered as sd
from dev_titles.cli import _read_any
from dev_titles.formats import write_srt, write_vtt
from dev_titles.model import Cue, Subtitle


def _sub(lang, *cues):
    return Subtitle(lang, [Cue(float(k), k + 1.0, list(c)) for k, c in enumerate(cues)]).renumber()


def _rows(*txts):
    return [{"lang": "ja", "i": k, "src": f"en {k}", "txt": t} for k, t in enumerate(txts)]


def test_a_name_fix_lands_in_its_one_cache_line_with_its_spacing():
    old = _sub(
        "ja",
        ["本日は ガゼル・バルディオ・ウィルさんを"],
        ["お迎えしています"],
        ["ガゼルさん ご参加ありがとう"],
    )
    new = _sub(
        "ja",
        ["本日は Ghizlan Bardiowilさんを"],
        ["お迎えしています"],
        ["Ghizlanさん ご参加ありがとう"],
    )
    rows = _rows(
        "本日は ガゼル・バルディオ・ウィルさんをお迎えしています", "ガゼルさん ご参加ありがとう"
    )
    done, failed = sd.sync_track(rows, "ja", old, new)
    assert failed == [] and len(done) == 2
    assert rows[0]["txt"] == "本日は Ghizlan Bardiowilさんをお迎えしています" and rows[0]["locked"]
    assert rows[1]["txt"] == "Ghizlanさん ご参加ありがとう"
    # a second sync finds it carried: nothing changes, nothing reported
    assert sd.sync_track(rows, "ja", old, new) == ([], [])


def test_a_latin_name_change_is_one_whole_word_edit():
    old, new = _sub("zh-Hans", ["Cutter，感谢您的邀请"]), _sub("zh-Hans", ["Chris，感谢您的邀请"])
    spans = sd.changed_spans(old, new)
    assert len(spans) == 1
    rows = [{"lang": "zh-Hans", "i": 0, "src": "", "txt": "Cutter，感谢您的邀请"}]
    sd.sync_track(rows, "zh-Hans", old, new)
    assert rows[0]["txt"] == "Chris，感谢您的邀请"
    assert sd.sync_track(rows, "zh-Hans", old, new) == ([], [])  # never 'Chrisis'


def test_layout_only_changes_are_skipped_and_ambiguous_spans_reported():
    assert sd.changed_spans(_sub("ja", ["転換", "点に"]), _sub("ja", ["転換点に"])) == []
    old = _sub("ja", ["同じ言葉です"], ["同じ言葉です"])
    new = _sub("ja", ["違う言葉です"], ["同じ言葉です"])
    rows = _rows("同じ言葉です", "同じ言葉です")
    done, failed = sd.sync_track(rows, "ja", old, new)
    assert done == [] and len(failed) == 1 and "not in exactly one cache line" in failed[0]


def test_sync_folder_en_and_tracks_and_the_reviewed_track_cap(tmp_path, monkeypatch):
    out, lib = tmp_path / "ticket" / "V", tmp_path / "FINAL_SUBTITLES"
    out.mkdir(parents=True)
    lib.mkdir()
    (out / "en.vtt").write_text(write_vtt(_sub("en", ["Thanks, Cutter."])), encoding="utf-8")
    (lib / "V.en.srt").write_text(write_srt(_sub("en", ["Thanks, Chris."])), encoding="utf-8")
    (out / "ja.srt").write_text(write_srt(_sub("ja", ["カカルさん ありがとう"])), encoding="utf-8")
    (lib / "V.ja.srt").write_text(write_srt(_sub("ja", ["Chrisさん ありがとう"])), encoding="utf-8")
    (out / "cache.jsonl").write_text(
        json.dumps(_rows("カカルさん ありがとう")[0], ensure_ascii=False) + "\n", encoding="utf-8"
    )
    rep = sd.sync(out, lib, _read_any, write_vtt, user="me")
    assert rep["en"] is True and rep["ja"] == {"locked": 1, "failed": []}
    assert "Thanks, Chris." in (out / "en.vtt").read_text() and list(out.glob("en.pre_sync_*.vtt"))
    assert "Chrisさん" in (out / "cache.jsonl").read_text()
    edit = json.loads((out / "locked_edits.jsonl").read_text().splitlines()[0])
    assert edit["note"] == "sync-delivered" and edit["by"] == "me" and edit["lang"] == "ja"
    assert sd.sync(out, lib, _read_any, write_vtt)["en"] is False  # already in sync
    monkeypatch.setattr(sd, "MAX_SPANS", 0)
    (lib / "V.ja.srt").write_text(write_srt(_sub("ja", ["全く別の訳です"])), encoding="utf-8")
    rep = sd.sync(out, lib, _read_any, write_vtt)
    assert rep["ja"]["locked"] == 0 and "reviewed track" in rep["ja"]["failed"][0]


def test_cli_sync_delivered_reports_and_fails_on_uncarried(tmp_path, monkeypatch, capsys):
    from types import SimpleNamespace as NS

    from dev_titles import cli

    monkeypatch.setattr(
        sd, "sync", lambda *a, **k: {"en": True, "ja": {"locked": 2, "failed": ["x"]}}
    )
    assert cli.cmd_sync_delivered(NS(out=str(tmp_path / "V"), library="", all=False)) == 1
    o = capsys.readouterr().out
    assert "approve-en" in o and "ja: 2 line(s)" in o and "NOT carried: x" in o
    monkeypatch.setattr(
        sd, "sync", lambda *a, **k: {"en": False, "ja": {"locked": 0, "failed": ["y"]}}
    )
    assert cli.cmd_sync_delivered(NS(out=str(tmp_path / "V"), library="", all=False)) == 1
    assert "nothing to carry" not in capsys.readouterr().out
    monkeypatch.setattr(sd, "sync", lambda *a, **k: {"en": False})
    assert cli.cmd_sync_delivered(NS(out=str(tmp_path / "V"), library="", all=False)) == 0
    assert "nothing to carry" in capsys.readouterr().out


def test_a_short_new_fragment_found_elsewhere_is_not_taken_as_carried():
    # deleting '不要な' leaves only short context: finding it elsewhere is no proof the fix landed
    old = _sub("ja", ["これは不要な言葉です"])
    new = _sub("ja", ["これは言葉です"])
    rows = _rows("これは不要な言葉です", "それは言葉です")
    done, failed = sd.sync_track(rows, "ja", old, new)
    assert done or failed  # applied or reported — never silently skipped


def test_the_sync_delivered_command_is_registered():
    from dev_titles import cli

    a = cli.build_parser().parse_args(["sync-delivered", "out/V", "--all"])
    assert a.func is cli.cmd_sync_delivered and a.all and a.library == ""


def test_nearby_edits_are_one_edit_and_context_never_crosses_into_the_next():
    """A deleted '今は' and an inserted 'こうしたメモ化を' a Latin word apart were applied
    separately and cut a ja line to 'React Compilerが こうド時に…'."""
    old = _sub("ja", ["今はReact Compilerがビルド時にメモ化を全て自動で行います"])
    new = _sub("ja", ["React Compilerが こうしたメモ化を", "ビルド時にすべて自動で行います"])
    rows = _rows("今はReact Compilerがビルド時にメモ化を全て自動で行います")
    done, failed = sd.sync_track(rows, "ja", old, new)
    txt = rows[0]["txt"]
    assert failed == [] and "こうド時" not in txt  # one edit, never a cut-up line
    assert sd._compact(txt) == sd._compact(
        "React Compilerがこうしたメモ化をビルド時にすべて自動で行います"
    )


def test_lines_matching_the_delivered_track_are_locked_too():
    old = _sub("ja", ["カカルさん ありがとう"], ["次へ進みます"])
    new = _sub("ja", ["Chrisさん ありがとう"], ["次へ進みます"])
    rows = _rows("カカルさん ありがとう", "次へ進みます", "まだ届いていない行")
    sd.sync_track(rows, "ja", old, new)
    assert rows[0]["locked"] and rows[1]["locked"]  # changed and unchanged delivered lines
    assert not rows[2].get("locked")  # a line not in the delivered track stays free


def test_a_line_differing_only_by_a_line_break_comma_is_locked():
    old = _sub("zh-Hans", ["非常感谢大家观看，迫不及待"])
    new = _sub("zh-Hans", ["非常感谢大家观看", "迫不及待"])
    rows = [{"lang": "zh-Hans", "i": 0, "src": "", "txt": "非常感谢大家观看，迫不及待"}]
    sd.sync_track(rows, "zh-Hans", old, new)
    assert rows[0]["locked"]


def test_a_second_sync_never_repeats_an_insertion_and_short_lines_are_not_locked_by_accident():
    old = _sub("ja", ["2つ目のメリットHermesV1です"])
    new = _sub("ja", ["2つ目のメリットがHermesV1です"])
    rows = _rows("2つ目のメリットHermesV1です", "はい")
    sd.sync_track(rows, "ja", old, new)
    sd.sync_track(rows, "ja", old, new)  # again, before the run folder is rebuilt
    assert rows[0]["txt"] == "2つ目のメリットがHermesV1です"  # never 'がが'
    assert not rows[1].get("locked")  # 'はい' isn't in the delivered track as a line


def test_a_sync_that_only_locks_lines_still_saves_the_cache(tmp_path):
    out, lib = tmp_path / "ticket" / "V", tmp_path / "FINAL_SUBTITLES"
    out.mkdir(parents=True)
    lib.mkdir()
    (out / "ja.srt").write_text(write_srt(_sub("ja", ["次へ進みましょう"])), encoding="utf-8")
    (lib / "V.ja.srt").write_text(write_srt(_sub("ja", ["次へ進みましょう"])), encoding="utf-8")
    (out / "cache.jsonl").write_text(
        json.dumps(_rows("次へ進みましょう")[0], ensure_ascii=False) + "\n", encoding="utf-8"
    )
    sd.sync(out, lib, _read_any, write_vtt)
    assert json.loads((out / "cache.jsonl").read_text().splitlines()[0])["locked"] is True
