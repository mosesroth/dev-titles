#!/usr/bin/env python3
"""Re-time a zh track whose text has drifted from the speech, WITHOUT changing its text.

For tracks the sync check (dev_titles/validate/sync.py) flags. Cause on record: the retired 1:1
"reflow" step left some tracks one or two cues behind the English; vendor reviews kept that timing.

  1. join the track's cue texts; a break between two CJK characters is marked "|"
  2. an LLM assigns the text, verbatim and in order, to the EN sentences — verified: the concatenation
     must equal the input exactly, else retry / abort
  3. each "|" is judged "join" (inside a word: 手机|壳) or "space" (a pause: 应用|我会在)
  4. cues are rebuilt on the EN sentence timing with build_target_cues (free segmentation)
Only change on top of the text: a ，/、 that lands at a line end becomes the line break (house rule).

  PYTHONPATH=src python3 scripts/retime_drifted.py <zh.srt> <en.vtt|srt> <out base path>
"""
import json, re, sys
from dev_titles.cli import _read_any
from dev_titles.config import spec_for
from dev_titles.formats import write_srt, write_vtt
from dev_titles.glossary import load
from dev_titles.model import Subtitle
from dev_titles.resegment import build_target_cues, merge_en_to_sentences
from dev_titles import llm
from dev_titles.validate.rules import validity_report
inv = llm.invoke()
zh_path, en_path, out_base = sys.argv[1:4]
spec = spec_for("zh-Hans"); g = load("glossary/glossary.yaml")
keep = [t.translations.get("zh-Hans") for t in g.terms if t.translations.get("zh-Hans")] + list(g.identifiers)
en = _read_any(en_path, "en"); sents = merge_en_to_sentences(en)
def _join(parts):
    """Join cue texts; a break between two CJK characters is marked "|" and decided after alignment."""
    out = parts[0]
    for p in parts[1:]:
        a, b = out.rstrip(), p.lstrip()
        both_cjk = a and b and re.match(r"[\u4e00-\u9fff]", a[-1]) and re.match(r"[\u4e00-\u9fff]", b[0])
        out = a + ("|" if both_cjk else " ") + b
    return out


zh_text = _join([c.text.replace("\n", "") for c in _read_any(zh_path, "zh-Hans").cues])
N = lambda s: re.sub(r"\s", "", s)
P = """Below are numbered English sentences from a video, and the Chinese subtitle TEXT that translates them,
in order. Split the Chinese text into consecutive pieces, one per English sentence id, so each piece is the
translation of that sentence. Use the Chinese EXACTLY as given: do not add, drop, reorder or change a single
character; every character (including the "|" marks) must land in exactly one piece, in order. A piece may be "" if the Chinese
merged that sentence into a neighbour. The Chinese may continue past the last sentence: stop at the last id
and leave the rest.
ENGLISH:
{en}
CHINESE TEXT:
{zh}
Return ONLY JSON: {{"pieces": [{{"id": n, "zh": "..."}}]}}"""
texts = [""] * len(sents); pos = 0; rest = zh_text; i = 0; step = 20
while i < len(sents):
    j = min(i + step, len(sents))
    en_block = "\n".join(f"{k} | {sents[k].text}" for k in range(i, j))
    budget = int(sum(len(sents[k].text) for k in range(i, j)) * 0.9) + 120
    zh_block = rest[:budget] if j < len(sents) else rest
    ok = False
    for attempt in range(3):
        m = re.search(r"\{.*\}", inv(P.format(en=en_block, zh=zh_block)) or "", re.S)
        try: pieces = {int(p["id"]): p["zh"] for p in json.loads(m.group(0))["pieces"]}
        except Exception: continue
        got = "".join(pieces.get(k, "") for k in range(i, j))
        if N(zh_block).startswith(N(got)) and (j < len(sents) or N(got) == N(zh_block)):
            ok = True; break
    if not ok:
        if step > 5:  # a hard stretch: retry it in smaller pieces
            step //= 2
            continue
        raise SystemExit(f"alignment failed at sentences {i}-{j}")
    step = 20
    for k in range(i, j): texts[k] = pieces.get(k, "").strip()
    # advance `rest` past the consumed characters (whitespace-insensitive)
    need = len(N(got)); cut = 0; seen = 0
    while seen < need: seen += (not rest[cut].isspace()); cut += 1
    rest = rest[cut:]; i = j
assert N("".join(texts)) == N(zh_text), "text changed!"
B = """Each Chinese subtitle line below contains "|" marks where an old caption break fell. For each mark decide:
"join" if it falls inside a word or tight phrase (e.g. 手机|壳, 帮|助), or "space" if it falls at a pause between
clauses or sentences (e.g. 应用|我会在). Lines (id | text):
{rows}
Return ONLY JSON: {{"<id>": ["join"|"space", ...one per mark, in order]}}"""
idx = [k for k, t in enumerate(texts) if "|" in t]
for c0 in range(0, len(idx), 40):
    chunk = idx[c0:c0 + 40]
    m = re.search(r"\{.*\}", inv(B.format(rows="\n".join(f"{k} | {texts[k]}" for k in chunk))) or "", re.S)
    dec = json.loads(m.group(0)) if m else {}
    for k in chunk:
        ch = dec.get(str(k), [])
        parts = texts[k].split("|"); t = parts[0]
        for q, p in enumerate(parts[1:]):
            t += ("" if q < len(ch) and ch[q] == "join" else " ") + p
        texts[k] = t
texts = [t.replace("|", "") for t in texts]
cues = build_target_cues(en, sents, texts, spec, keep_together=keep, free_segment=True)
s = Subtitle("zh-Hans", cues).renumber()
open(out_base + ".srt", "w", encoding="utf-8").write(write_srt(s)); open(out_base + ".vtt", "w", encoding="utf-8").write(write_vtt(s))
json.dump(texts, open(out_base + ".alignment.json", "w"), ensure_ascii=False, indent=0)
r = validity_report(s, False); print("cues", len(cues), "err-cues", r["cues_with_errors"], r["violations_by_rule"])
