"""zh second-person referent pass (zh_pronoun)."""

import json

from dev_titles.localize import SentenceRecord
from dev_titles.zh_pronoun import _policy_ok, guard_ok, pronoun_pass


def test_guard_only_pronoun_changes():
    assert guard_ok("您目前只支持iOS", "你们目前只支持iOS")
    assert guard_ok("将您的应用打包为框架", "将应用打包为框架")
    assert not guard_ok("您目前只支持iOS", "你们现在只做iOS")  # rewording slipped in
    assert not guard_ok("您好", "您好")  # no change


def test_policy_generic_only_possessive():
    assert _policy_ok("GROUP", "您目前只支持iOS", "你们目前只支持iOS")
    assert _policy_ok("AUDIENCE", "建议您试试", "建议大家试试")
    assert _policy_ok("GENERIC", "将您的应用打包", "将应用打包")
    assert not _policy_ok("GENERIC", "如果您想试试", "如果想试试")  # generic subject: kept


def test_pronoun_pass_applies_and_is_idempotent(tmp_path):
    recs = [
        SentenceRecord("zh-Hans", 0, "Can you tell us about it?", "您能聊聊吗？", "", []),
        SentenceRecord("zh-Hans", 1, "Right now you're just doing iOS.", "您目前只支持iOS", "", []),
        SentenceRecord("zh-Hans", 2, "We ship it.", "我们发布它", "", []),
    ]
    calls = []

    def invoke(prompt):
        calls.append(prompt)
        assert "We ship it." not in prompt  # no "you": never sent
        return json.dumps([{"id": 1, "referent": "GROUP", "zh": "你们目前只支持iOS"}])

    state = tmp_path / "pronoun_state.zh-Hans.json"
    chg = pronoun_pass(recs, invoke, state, logger=lambda m: None)
    assert [c["i"] for c in chg] == [1] and recs[1].txt == "你们目前只支持iOS"
    assert recs[0].txt == "您能聊聊吗？"
    assert pronoun_pass(recs, invoke, state, logger=lambda m: None) == [] and len(calls) == 1
