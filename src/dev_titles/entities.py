"""NAMES POLICY per video: partner company names stay Latin; products named with ordinary words
are translated, one rendering per video.

What native reviewers do: company names stay in Latin (never localized), while a partner's product
named with an ordinary word is translated, consistently (a 'Concierge' product -> ja コンシェルジュ,
zh 礼宾服务). Without this the harness localized company names (from a Wikipedia lookup and term
pins) and mixed Latin / translated forms of one product (case-sensitive contradictory pins).
The glossary's own brands still follow the glossary.

One LLM pass over the brief + transcript (cached in <out>/entities.json):
  companies:          ["Acme Air", ...]                 -> kept Latin (added to the video's names)
  ordinary_products:  [{"name": "Concierge", "ja": "コンシェルジュ", "zh-Hans": "礼宾服务"}]
                      -> one per-video rendering per language (every case variant replaced)
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

PROMPT = """You help subtitle translators keep names consistent in a developer video.

VIDEO BRIEF: {brief}

CAPITALIZED NAMES IN THE TRANSCRIPT (count): {caps}

TRANSCRIPT (excerpt): {text}

List:
1. "companies": every company, organization or brand named (partners, customers, employers,
   competitors), spelled as in the transcript. These keep their Latin spelling in subtitles.
2. "ordinary_products": product or feature names that are ORDINARY English words used as a name
   (e.g. a partner's "Concierge", a "Studio", an "Assistant"), which subtitles translate with the
   standard local word, consistently. Give ja (standard term or katakana loanword) and zh-Hans —
   the natural FULL noun a native subtitler would write for the product (not a clipped one-word
   form that only works as a modifier).
Exclude: people's names; the channel owner's own products and brands (they follow the glossary);
coined product names that are not ordinary words (e.g. Kubernetes, Figma) — those stay Latin.
Return ONLY JSON: {{"companies": [...], "ordinary_products": [{{"name": "...", "ja": "...", "zh-Hans": "..."}}]}}"""

_OWN_BRANDS = re.compile(r"^$")  # set your own brands here; the glossary covers them otherwise


def _caps(text: str, n: int = 40) -> str:
    c = Counter(m.group(0) for m in re.finditer(r"\b[A-Z][a-zA-Z0-9]+(?: [A-Z][a-zA-Z0-9]+)*\b", text))
    return ", ".join(f"{k} ({v})" for k, v in c.most_common(n) if v >= 2)


def names_policy(en_text: str, invoke, cache_path=None, brief: str = "") -> dict:
    """{'companies': [...], 'ordinary_products': [{'name', 'ja', 'zh-Hans'}]} — {} on failure."""
    key = hashlib.sha1((brief + "\x00" + en_text).encode("utf-8"), usedforsecurity=False).hexdigest()[:16]
    p = Path(cache_path) if cache_path else None
    if p and p.exists():
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            if d.get("key") == key:
                return d.get("policy", {})
        except (json.JSONDecodeError, OSError):
            pass
    try:
        out = invoke(PROMPT.format(brief=brief or "(none)", caps=_caps(en_text) or "(none)", text=en_text[:6000])) or ""
        m = re.search(r"\{.*\}", out, re.S)
        pol = json.loads(m.group(0)) if m else {}
    except Exception:
        pol = {}
    companies = [c.strip() for c in pol.get("companies", []) if isinstance(c, str) and c.strip()
                 and not _OWN_BRANDS.match(c.strip())]
    products = [x for x in pol.get("ordinary_products", []) if isinstance(x, dict) and x.get("name")
                and not _OWN_BRANDS.match(str(x["name"]))]
    pol = {"companies": companies, "ordinary_products": products}
    if p:
        p.write_text(json.dumps({"key": key, "policy": pol}, ensure_ascii=False, indent=1), encoding="utf-8")
    return pol


def apply_policy(video_vocab: dict | None, policy: dict, langs=("ja", "zh-Hans")) -> dict:
    """Merge the policy into the per-video vocab: companies join the keep-Latin names and lose any
    localized rendering; each ordinary-word product gets ONE rendering per language, replacing every
    case variant ('concierge' / 'Concierge'). Returns a new dict."""
    vv = json.loads(json.dumps(video_vocab or {}))  # deep copy
    companies = policy.get("companies") or []
    products = policy.get("ordinary_products") or []
    if not companies and not products:
        return vv
    names = list(vv.get("names") or [])
    vv["names"] = names + [c for c in companies if c not in names]
    rend = vv.setdefault("renderings", {})
    comp_l = {c.lower() for c in companies}
    for lang in set(rend) | set(langs):
        r = rend.setdefault(lang, {})
        for src in [s for s in r if s.lower() in comp_l]:
            del r[src]  # a company name is never localized
        for prod in products:
            tgt = (prod.get(lang) or "").strip()
            if not tgt:
                continue
            name = str(prod["name"]).strip()
            for src in [s for s in r if s.lower() == name.lower()]:
                del r[src]
            r[name] = {"canonical": tgt, "avoid": [name]}
    return vv
