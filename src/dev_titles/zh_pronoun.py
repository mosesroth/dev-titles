"""zh second-person REFERENT check: who does each English "you" mean?

The zh house register uses 您 for the person being spoken to (native-reviewer verdict, see
localize._normalizer), but the translator also writes 您 where "you" is NOT one person:

  - generic "you" (anyone, developers in general), possessive:     omit 您的
      "package your app as a framework"          将您的应用打包为框架 -> 将应用打包为框架
    (a generic SUBJECT 您 is left as is — see _policy_ok)
  - the addressee's team / company (plural):   你们
      "right now you're just doing iOS" (said to an engineer about their team's library)  您目前只支持iOS -> 你们目前只支持iOS
  - the viewers:                                 大家

The style notes already say this (GENERIC "YOU", AUDIENCE "you"), but a per-sentence prompt rule
did not land reliably (an A/B against a human-reviewed final). This pass looks only at
sentences whose zh has a singular 您/你 and whose English has "you/your", asks one batched LLM call
for the referent, and applies the pronoun change only. GUARD: apart from the pronoun (and a
的 / comma / 就 next to it) the text must be unchanged, so a rewrite can't slip in.
IDEMPOTENT: a sidecar state file remembers each (source, target) pair it has checked.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

_YOU = re.compile(r"\byou(?:r|rs|rself|'re|'ve|'d|'ll)?\b", re.I)
_SINGULAR = re.compile(r"您|你(?!们)")
# "you all" / "you guys" / "both of you": rendered 大家, which is right for the VIEWERS but wrong when
# it's said to the people in the conversation ("You all build in public." -> 你们)
_YOU_ALL = re.compile(
    r"\byou (?:all|guys|both|two)\b|\b(?:all|both|each|either) of you\b|\byou folks\b", re.I
)
# characters a pronoun fix may add or remove; everything else must match exactly
_PRONOUN_CHARS = set("您你们大家的就，, ")


def _policy_ok(referent: str, old: str, new: str) -> bool:
    """Which referent fixes are APPLIED. GROUP (你们) and AUDIENCE (大家) are errors and always apply.
    GENERIC applies only to the possessive 您的+noun (将您的应用 -> 将应用, the house "avoid 您的+object"
    rule): a generic SUBJECT 您 ("如果您想试试…") is how the human final usually kept it (9 of 12 on the
    A/B against the human final), so dropping it is a reviewer-split style call, not a fix."""
    if referent in ("GROUP", "AUDIENCE"):
        return True
    return referent == "GENERIC" and "您的" in old and old.replace("您的", "") == new


PROMPT = """You are checking second-person pronouns in Simplified Chinese subtitles of an English video ({kind}).

House rule: 您 is ONLY for one specific person being spoken to (the guest or the host, e.g. a question
put to them, "for you, Mike", "I love the work you do"). For every other "you":
  - GENERIC  — "you" means anyone / developers in general / how a tool works ("you can run them",
               "it lets you build…", "package your app"): OMIT the pronoun (drop 您 / 您的; keep the rest).
  - GROUP    — "you" means the addressee's team, company or several people ("right now you're just doing
               iOS" said to an engineer about their company's library): use 你们.
  - AUDIENCE — "you" means the viewers watching the video: use 大家.
  (A "you all / you guys" said to the people IN the conversation — the guests — is GROUP: 你们, even
   where the Chinese currently has 大家; 大家 is only for the viewers.)
  - PERSON   — one specific person being spoken to: KEEP 您.
When unsure, it is PERSON (keep). Change ONLY the pronoun (and a directly attached 的 or comma);
never reword anything else.

For each item: the preceding English lines (context), the English sentence, and the current Chinese.
Return a JSON array with ONLY the items that need a change:
[{{"id": <id>, "referent": "GENERIC|GROUP|AUDIENCE", "zh": "<corrected Chinese>"}}]
Return [] if nothing needs changing.

{items}
"""


def _h(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:16]


def guard_ok(old: str, new: str) -> bool:
    """Only pronoun-related characters may differ, and something must actually change."""
    if not new or new == old:
        return False
    strip = lambda s: "".join(c for c in s if c not in _PRONOUN_CHARS)  # noqa: E731
    return strip(old) == strip(new)


def _parse(text: str) -> list:
    m = re.search(r"\[.*\]", text or "", re.S)
    if not m:
        return []
    try:
        out = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    return [x for x in out if isinstance(x, dict)]


def pronoun_pass(
    records,
    invoke,
    state_path=None,
    *,
    normalize=None,
    conversational: bool = False,
    logger=print,
    batch: int = 40,
) -> list[dict]:
    """Fix the zh pronoun of records whose English "you" is generic / plural / the audience.
    Mutates `records` in place; returns the applied changes [{i, referent, old, new}]."""
    sp = Path(state_path) if state_path else None
    state = json.loads(sp.read_text(encoding="utf-8")) if sp and sp.exists() else {}
    srcs = [r.src for r in records]
    todo = [
        r
        for r in records
        if (
            (_YOU.search(r.src) and _SINGULAR.search(r.txt))
            or (_YOU_ALL.search(r.src) and "大家" in r.txt)
        )
        and not state.get(_h(r.src, r.txt))
    ]
    kind = "interview / conversation" if conversational else "talk or tutorial"
    applied: list[dict] = []
    for b in range(0, len(todo), batch):
        chunk = todo[b : b + batch]
        items = "\n".join(
            json.dumps(
                {
                    "id": r.i,
                    "context": " ".join(srcs[max(0, r.i - 2) : r.i]),
                    "en": r.src,
                    "zh": r.txt,
                },
                ensure_ascii=False,
            )
            for r in chunk
        )
        try:
            out = _parse(invoke(PROMPT.format(kind=kind, items=items)))
        except Exception as e:  # never sink a run on it
            logger(f"zh-Hans: pronoun check skipped ({type(e).__name__})")
            return applied
        by_id = {r.i: r for r in chunk}
        for x in out:
            r = by_id.get(x.get("id"))
            new = (x.get("zh") or "").strip()
            if r is None or x.get("referent") not in ("GENERIC", "GROUP", "AUDIENCE"):
                continue
            if normalize is not None:
                new = normalize(new, r.src)
            if not guard_ok(r.txt, new):
                logger(f"zh-Hans: pronoun fix #{r.i} rejected by the guard: {new!r}")
                continue
            if not _policy_ok(x["referent"], r.txt, new):
                continue
            applied.append({"i": r.i, "referent": x["referent"], "old": r.txt, "new": new})
            r.txt = new
        for r in chunk:
            state[_h(r.src, r.txt)] = 1
        if sp:
            sp.write_text(json.dumps(state), encoding="utf-8")
    return applied
