"""Shared test helpers + common imports for the split test suite."""

from dev_titles.config import spec_for
from dev_titles.cue_builder import build_en_cues
from dev_titles.eval import (
    GembaMqmScorer,
    QeResult,
    QeScorer,
    below_threshold_segments,
    parse_gemba_mqm,
    rank_engines,
)
from dev_titles.formats import read_vtt, write_srt, write_vtt
from dev_titles.glossary import load
from dev_titles.linguistics import get_joiner, is_repeated_text
from dev_titles.model import Cue, Subtitle
from dev_titles.segment import apply_kinsoku, break_lines, normalize_zh_punct
from dev_titles.textwidth import display_width
from dev_titles.transcribe import from_transcribe_json
from dev_titles.translate import (
    TranslationRequest,
    Translator,
    build_translate_prompt,
    translate_segment,
)
from dev_titles.validate import validate_subtitle
from dev_titles.validate.rules import validity_report


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
