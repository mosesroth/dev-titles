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
Return ONLY JSON: {{"companies": [...], "ordinary_products": [{{"name": "...", "company": "<whose product>",
"ja": "...", "zh-Hans": "..."}}]}}"""

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


# ---- Official renderings from the partner's own website, locked in glossary/partners.yaml ----

PARTNERS_PATH = "glossary/partners.yaml"
_LANG_HINT = {"ja": ("日本語", "Japanese", ("/ja", "ja-jp", "/jp", "jp/", "ja_jp")),
              "zh-Hans": ("中文", "Simplified Chinese", ("/zh", "zh-cn", "/cn", "cn/", "zh_cn", "zh-hans"))}
FIND = """This text is from {url}, a page on {company}'s own website in {L}. How does this page write the
{company} product called "{product}" in {L}? Use ONLY this page: if it does not mention that product,
answer null. Answer ONLY JSON: {{"rendering": "<exactly as written on the page>" | null, "quote": "<the exact
sentence or phrase from the page that contains it>"}}

PAGE TEXT:
{text}"""


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def load_partners(path=PARTNERS_PATH) -> dict:
    import yaml

    p = Path(path)
    if not p.exists():
        return {}
    d = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return d.get("partners", {}) or {}


def save_partners(partners: dict, path=PARTNERS_PATH) -> None:
    import yaml

    head = ("# Partner product names with their OFFICIAL ja / zh-Hans renderings, found on the partner's own\n"
            "# website by the pre-run names step (entities.official_rendering) and locked here. Company names\n"
            "# themselves stay Latin. Edit by hand to correct; an entry here is never looked up again.\n")
    Path(path).write_text(head + yaml.safe_dump({"partners": partners}, allow_unicode=True, sort_keys=True),
                          encoding="utf-8")


def official_rendering(company, product, lang, *, web_search, web_fetch, invoke, max_pages=4):
    """(rendering, url) from the company's OWN site in `lang`, or (None, None). A page counts only if its
    domain contains the company's name, and the rendering only if the quoted text is really on the page."""
    hint, L, paths = _LANG_HINT[lang]
    slug = _slug(company)
    urls = []
    for q in (f'"{company}" "{product}" {hint}', f"{company} {product} {hint}", f"{company} {hint}"):
        try:
            urls += [u for u in web_search(q) if u not in urls]
        except Exception:
            continue
    own = [u for u in urls if slug and slug[:12] in _slug(re.sub(r"^https?://([^/]+).*", r"\1", u))]
    own.sort(key=lambda u: 0 if any(t in u.lower() for t in paths) else 1)
    for u in own[:max_pages]:
        try:
            text = web_fetch(u)
        except Exception:
            continue
        if not text or (lang == "ja" and not re.search(r"[ぁ-んァ-ン]", text)) or (
            lang == "zh-Hans" and (not re.search(r"[一-鿿]", text) or re.search(r"[ぁ-ん]", text))
        ):
            continue  # not a page in that language
        try:
            m = re.search(r"\{.*\}", invoke(FIND.format(url=u, company=company, L=L, product=product,
                                                       text=text[:12000])) or "", re.S)
            d = json.loads(m.group(0)) if m else {}
        except Exception:
            d = {}
        r, quote = (d.get("rendering") or "").strip(), (d.get("quote") or "").strip()
        if r and quote and r in quote and re.sub(r"\s", "", quote) in re.sub(r"\s", "", text):
            return r, u
    return None, None


def resolve_official(policy, *, web_search, web_fetch, invoke, langs=("ja", "zh-Hans"),
                     path=PARTNERS_PATH, logger=print) -> dict:
    """Replace the policy's model-chosen product renderings with LOCKED official ones: from
    partners.yaml if known, else looked up on the partner's site and locked there. Products with no
    official rendering keep the model's choice, marked unverified."""
    partners = load_partners(path)
    changed = False
    for prod in policy.get("ordinary_products", []):
        company = (prod.get("company") or "").strip()
        name = str(prod["name"]).strip()
        if not company:
            continue
        entry = partners.setdefault(company, {}).setdefault("products", {}).setdefault(name, {})
        for lang in langs:
            locked = entry.get(lang)
            if isinstance(locked, dict) and locked.get("rendering"):
                prod[lang] = locked["rendering"]
                prod.setdefault("source", {})[lang] = locked.get("source", "partners.yaml")
                continue
            r, url = official_rendering(company, name, lang, web_search=web_search, web_fetch=web_fetch,
                                        invoke=invoke)
            if r:
                prod[lang] = r
                prod.setdefault("source", {})[lang] = url
                entry[lang] = {"rendering": r, "source": url}
                changed = True
                logger(f"names policy: official {lang} rendering of {company} '{name}' = {r} ({url}) — locked")
            else:
                prod.setdefault("source", {})[lang] = "unverified (no official page found)"
        if not entry:
            del partners[company]["products"][name]
    if changed:
        save_partners({k: v for k, v in partners.items() if v.get("products")}, path)
    return policy
