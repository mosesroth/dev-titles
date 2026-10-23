"""Shared test helpers + common imports for the split test suite."""

from dev_titles.translate import (
    Translator,
)


def _transcribe_json(words):
    """Build minimal AWS Transcribe output JSON from (text, start, end) tuples."""
    items = []
    for text, start, end in words:
        items.append(
            {
                "type": "pronunciation",
                "start_time": str(start),
                "end_time": str(end),
                "alternatives": [{"confidence": "0.99", "content": text}],
            }
        )
    return {"results": {"items": items}}


def _mqm_by_target_marker(prompt):
    """Fake MQM judge: flags a 'major' on any TARGET containing 'BAD', else no-error.
    Parses the numbered batch prompt that `_judge_batch` builds."""
    import re as _re

    out = []
    for m in _re.finditer(r"\[(\d+)\] SOURCE:.*?TARGET: ([^\n]*)", prompt, _re.S):
        n, tgt = m.group(1), m.group(2)
        out.append(f"[{n}] " + ("major: mistranslation" if "BAD" in tgt else "no-error"))
    return "\n".join(out)


class _FakeTranslator(Translator):
    """Records the prompts it receives and returns a scripted draft/refined pair."""

    def __init__(self, draft, refined):
        self.draft, self.refined = draft, refined
        self.reqs = []

    def translate(self, req):
        self.reqs.append(req)
        return self.draft

    def reflect_and_refine(self, req, draft):
        return self.refined
