"""Per-video vocabulary pre-ingestion.

Every new video carries domain vocab the static glossary doesn't have (product
names, third-party tools, code/API identifiers). Instead of patching the glossary
after the fact, this builds a per-video supplement UP FRONT: scan the video's seed
text (title + transcript/captions) for the things worth looking up, pull the
relevant docs (the product's developer docs via a docs-search MCP; third-party feature
home docs via web fetch), and harvest the canonical spellings + code identifiers.

Doc access (MCP / web fetch) is only available to the orchestrating agent, so the
lookups are passed in as callables — `docs_search(query)->text` and (optional)
`web_fetch(url)->text`. This module stays a pure, testable transformer; the agent
(or the subtitle-localizer skill) wires the real MCP/fetch and persists the result.
The pipeline then just loads the per-video vocab and merges it with the glossary —
feeding BOTH transcription correction (spellings/identifiers) and translation (DNT).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

# Code/API identifiers in doc text: camelCase, PascalCase runs, dotted (a.b),
# and scope-resolved (a::b). This is what ASR splits into words and translation
# must keep Latin.
_IDENT = re.compile(
    r"\b[A-Za-z][A-Za-z0-9]*(?:[.:]{1,2}[A-Za-z][A-Za-z0-9]*)+\b"  # dotted / a::b
    r"|\b[a-z]+[A-Z][A-Za-z0-9]*\b"  # camelCase
    r"|\b[A-Z][a-z]+[A-Z][A-Za-z0-9]*\b"
)  # PascalCase (2+ humps)

# Proper-noun / product-name candidates in the seed to look up (multi-word Caps
# runs, and standalone acronyms). We look these up; the docs give us the exact
# spelling + the API vocab around them.
_PROPER = re.compile(r"\b(?:[A-Z][a-zA-Z0-9]+(?:[ .][A-Z][a-zA-Z0-9.]+)+)\b")
_ACRONYM = re.compile(r"\b[A-Z]{2,6}(?:\.js)?\b")

# Everyday capitalized words that are not domain vocab (avoid looking these up).
# Includes sentence-initial words that otherwise start a bogus multi-word proper
# run ("With React", "And Lazy", "Whereas ...") — a term must not BEGIN with one.
_STOP = frozenset(
    "The This That These Those With From When While If For And But So "
    "You Your We Our They It Its Here There Now Today First Second Third "
    "A An As At By In On Of To Or Nor Yet Whereas Whether Because "
    "Once Just Like Also Then Than Thank Thanks Hello Hi Hey Let Lets "
    "Next Finally However Additionally Unlike Both Each Every".split()
)


# Common English words (fillers, discourse markers, interjections, generic adjectives/
# adverbs/verbs) that get capitalized at sentence start and pollute the single-word term
# scan — pinning a rendering for these forces one translation of an everyday word and even
# hallucinates (a live run proposed 'Actually'->哀しみの天使, 'Very'->とても, 'Sure'->もちろん).
# A per-video TERM must be domain/proper-noun, never one of these.
_COMMON_WORDS = frozenset("""
very sure cool nice good great well okay yeah yes maybe really quite pretty actually
exactly basically literally honestly absolutely definitely totally probably certainly
obviously essentially especially usually simply clearly truly right wrong true false
what when where which while about above after again always another because before being
between both could doing down during each even every first from gonna wanna going have
here into just know like look made make many more most much need next only other over
part some such than that them then there these they thing things think this those time
want were will with would your huge super lots kind sort thing stuff nice awesome amazing
incredible fantastic wonderful brilliant perfect happy sorry hello welcome thanks thank
today tomorrow yesterday people something anything everything nothing someone everyone
plus minus okay okey guys folks stuff thanks welcome
press choose check click select enter type open close install navigate launch start stop
play pause hit tap save copy paste download upload verify provide declare build deploy
""".split())


def _is_common_word(term: str) -> bool:
    return " " not in term and term.lower() in _COMMON_WORDS


def _is_acronym_form(term: str) -> bool:
    """All-caps alphabetic, 2-5 chars (AI, API, SDK, LLM) — kept Latin, never translated."""
    return term.isupper() and term.isalpha() and 2 <= len(term) <= 5


def extract_lookup_terms(seed_text: str, max_terms: int = 40) -> list[str]:
    """Distinct product/proper-noun/acronym candidates worth a doc lookup / rendering pin,
    most frequent first. Multi-word proper phrases are kept freely; single capitalized words
    are kept unless they are a common English word (the noise/hallucination source — 'Very',
    'Actually', 'Sure' — while genuine single-word proper nouns like 'Perfetto' stay)."""
    from collections import Counter

    c: Counter = Counter()
    # split on sentence-final marks first so a proper-noun run can't span "…App. Hello"
    for sent in re.split(r"[.!?]\s+", seed_text):
        for m in _PROPER.finditer(sent):
            phrase = m.group(0).strip()
            # drop a run that BEGINS with a stopword (sentence-initial capital):
            # "With React", "And Lazy", "The Developer Portal or" -> noise.
            if phrase.split()[0] in _STOP:
                continue
            c[phrase] += 1
    for m in _ACRONYM.finditer(seed_text):
        t = m.group(0)
        if t not in _STOP:
            c[t] += 1
    # single Capitalized words (Perfetto, Shaka) — drop common English words that pollute
    # the scan ('Very', 'Actually', 'Sure') and hallucinate renderings. Beyond the denylist
    # (which can't be exhaustive), require POSITIVE proper-noun evidence: a genuine proper noun
    # is (almost) always capitalized, while a common noun only gets a capital at a sentence
    # start or in a title. So keep a word only if its capitalized count DOMINATES its lowercase
    # count — 'Customers' (2 cap / 14 lower) is dropped, 'Perfetto'/'Nickel' (never lowercased)
    # are kept. Denylist-independent; catches the common nouns the curated list misses.
    low_form: Counter = Counter(
        m.group(0).lower()
        for m in re.finditer(r"\b[A-Za-z]{3,}\b", seed_text)
        if m.group(0).islower()
    )
    cap_form: Counter = Counter(re.findall(r"\b[A-Z][a-z]{3,}\b", seed_text))
    # STANDALONE evidence: a capitalized word that only ever appears INSIDE a longer proper run
    # ('World' in 'Hello World', 'Studio', 'Code') is not its own term — pinning it gave
    # World→世界 / Code→符号. Count only outside-a-run uses.
    # (Sentence-initial tutorial imperatives — Press/Choose/Check — are in _COMMON_WORDS.)
    _runs = [m.span() for sent in [seed_text] for m in _PROPER.finditer(sent)]
    def _standalone(w: str) -> int:
        n = 0
        for m in re.finditer(rf"\b{re.escape(w)}\b", seed_text):
            a, b = m.span()
            if any(ra <= a and b <= rb and (ra, rb) != (a, b) for ra, rb in _runs):
                continue  # inside a multi-word proper run
            n += 1
        return n
    for w, ccount in cap_form.items():
        if w in _STOP or _is_common_word(w):
            continue
        if not _standalone(w):
            continue
        if low_form.get(w.lower(), 0) == 0 or ccount > low_form[w.lower()]:
            c[w] += ccount
    ranked = [t for t, _ in c.most_common() if t.lower() not in {"the", "a", "an"}]
    return ranked[:max_terms]


def prune_terms_to_text(vocab: dict, text: str) -> list[str]:
    """Drop harvested `terms` (and their renderings) whose source no longer appears in the
    CORRECTED transcript, returning the dropped list. The vocab is harvested from the RAW ASR
    (before clean-read), so an ASR mishear ('Chief Devilist' <- 'Chief Evangelist') becomes a
    bogus term; once clean-read fixes the transcript the garble is gone from the captions and
    must leave the vocab too — otherwise it's treated as a keep-Latin DNT term and forced,
    untranslated, into the output. Identifiers and speaker names are NOT pruned: an identifier
    (shaka.Player.configure) or a name legitimately differs from the spoken/caption form."""
    low = (text or "").lower()

    def present(t: str) -> bool:
        return bool(t) and t.lower() in low

    dropped = [t for t in vocab.get("terms", []) if not present(t)]
    if dropped:
        vocab["terms"] = [t for t in vocab.get("terms", []) if present(t)]
        for _lang, m in (vocab.get("renderings") or {}).items():
            for k in list(m):
                if not present(k):
                    del m[k]
    return dropped


def harvest_identifiers(doc_text: str) -> list[str]:
    """Code/API identifiers found in fetched documentation text (deduped)."""
    seen, out = set(), []
    for m in _IDENT.finditer(doc_text):
        tok = m.group(0)
        low = tok.lower()
        if low in seen:
            continue
        if low.endswith((".html", ".png", ".com", ".dev", ".org", ".net", ".io", ".gov")):
            continue  # drop file/URL noise
        seen.add(low)
        out.append(tok)
    return out


def build_video_vocab(
    seed_text: str,
    docs_search,
    *,
    web_fetch=None,
    third_party: dict[str, str] | None = None,
    max_terms: int = 25,
    translate=None,
    langs=("ja", "zh-Hans"),
    metadata: str = "",
    known=(),
) -> dict:
    """Harvest a per-video vocab supplement.

    - `docs_search(query) -> text`: product developer-doc search (e.g. wrap a
      docs-search MCP server). Called for each lookup term; identifiers are harvested
      from the returned snippets/pages.
    - `web_fetch(url) -> text` (optional): for `third_party` features not in the
      product docs — a {name: home_doc_url} map (e.g. {"Shaka Player": "https://.../docs"}).
    - `translate(prompt) -> str` (optional): when given, also pins ONE canonical
      target rendering per term per language (see `propose_renderings`) so the
      translation stage stops flip-flopping video-specific terms.
    - `metadata` (optional): written video metadata (e.g. a YouTube description) —
      the reliable source for speaker names ("Presented by X"), harvested into
      `names` alongside self-introductions in the seed.
    - `known` (optional): source forms the CURATED glossary already owns. They (and single words
      of curated multi-word names) get no per-video rendering — the glossary decides, and the
      LLM/Wikipedia guess was the source of Orbit→轨道, Nova→新星, Orbit TV→Acme Orbit TV.
    Returns {"terms", "identifiers", "renderings", "names"}.
    """
    terms = extract_lookup_terms(seed_text, max_terms=max_terms)
    _known = {k.lower() for k in known or ()}
    _known_words = {w for k in _known if " " in k for w in k.split()}

    def _curated(t: str) -> bool:
        tl = t.lower()
        if tl in _known or (" " not in tl and tl in _known_words):
            return True
        # a sub-phrase of a curated name ('Orbit Virtual' ⊂ 'Orbit Virtual Device')
        return " " in tl and any(f" {tl} " in f" {k} " for k in _known if " " in k)
    idents: list[str] = []
    for t in terms:
        try:
            idents += harvest_identifiers(docs_search(t) or "")
        except Exception:
            continue
    if web_fetch and third_party:
        for name, url in third_party.items():
            try:
                idents += harvest_identifiers(web_fetch(url) or "")
            except Exception:
                continue
    # Acronyms (AI/API/LLM/SDK) are DNT — kept Latin via the identifiers list, NOT given a
    # translated rendering pin (a live run mis-pinned 'AI'->人工知能). Fold them into idents.
    idents += [t for t in terms if _is_acronym_form(t)]
    # dedupe identifiers, keep terms (proper names) separately
    seen, uidents = set(), []
    for i in idents:
        if i.lower() not in seen:
            seen.add(i.lower())
            uidents.append(i)
    names = harvest_names(seed_text, metadata)
    vocab: dict = {"terms": terms, "identifiers": uidents, "renderings": {}, "names": names}
    if translate is not None:
        # skip=names so speaker names aren't given translated rendering pins (they're DNT,
        # and the propose step ran on the pre-correction transcript with misheard spellings).
        _rterms = [t for t in terms if not _curated(t)]
        vocab["renderings"] = propose_renderings(
            _rterms, translate, langs=langs, context=seed_text, skip=names
        )
        # AUTHORITATIVE entity names (brand/company/product) override the LLM guess:
        # look up each entity's official localized name and pin it. Only when a real
        # web_fetch is wired (the pure default lambda returns "").
        if web_fetch is not None:
            try:
                # Same guardrails as propose: entity lookup is for BRAND/PRODUCT names, not
                # concepts/acronyms — 'AI' has a Wikipedia langlink (→人工知能) but must stay
                # Latin; common words + speaker names are excluded too.
                _skipn = {n.lower() for n in names}
                # multi-word only: a single word's Wikipedia article is usually ANOTHER sense
                # (Nova -> the astronomical 新星, Orbit -> 轨道); curated names are the glossary's call.
                entities = [
                    t
                    for t in _rterms
                    if " " in t.strip()
                    and not _is_common_word(t)
                    and not _is_acronym_form(t)
                    and t.lower() not in _skipn
                ]
                auth = resolve_entity_renderings(
                    entities, web_fetch=web_fetch, invoke=translate, langs=langs
                )
                for lang, m in auth.items():
                    vocab["renderings"].setdefault(lang, {}).update(m)
            except Exception:
                pass  # authoritative-rendering lookup is best-effort; keep the base vocab
    return vocab


def identify_subject(transcript_text: str, invoke) -> dict:
    """From a first-pass transcript (NOT metadata — live there is none), identify the
    video's subject and authoritative documentation to pre-ingest vocab from. Returns
    {"subject", "products": [...], "doc_urls": [...], "confident": bool}. When it
    cannot confidently name the subject/docs, `confident` is False — the caller must
    then ASK the operator for a docs URL rather than silently proceeding."""
    import json as _json

    prompt = (
        "You are prepping vocabulary for subtitling a technical developer video. From the "
        "transcript excerpt, identify: the main SUBJECT (one line), the PRODUCTS/frameworks/"
        "libraries named, and authoritative DOCUMENTATION URLs whose pages would define this "
        "video's vocabulary (the product's official developer docs, or third-party "
        "framework docs e.g. reactnative.dev, docs.expo.dev, react.dev). Only give a URL you "
        "are confident is real and relevant. If you CANNOT confidently identify the subject or "
        "any real doc URL, set confident=false. Output ONLY JSON: "
        '{"subject":"...","products":["..."],"doc_urls":["..."],"confident":true|false}.\n\n'
        "TRANSCRIPT:\n" + transcript_text[:3000]
    )
    try:
        raw = invoke(prompt) or ""
        s, e = raw.find("{"), raw.rfind("}")
        d = _json.loads(raw[s : e + 1]) if s != -1 else {}
    except Exception:
        d = {}
    return {
        "subject": d.get("subject", ""),
        "products": d.get("products", []),
        "doc_urls": [
            u for u in d.get("doc_urls", []) if isinstance(u, str) and u.startswith("http")
        ],
        "confident": bool(d.get("confident")),
    }


def prep_video_vocab(
    transcript_text: str,
    invoke,
    *,
    docs_search=None,
    web_fetch=None,
    extra_doc_urls=(),
    out_path=None,
) -> dict:
    """One call for the agent/skill LIVE flow (no YouTube metadata): identify the
    subject from the transcript, gather docs, build+save per-video vocab. Returns
    {"vocab", "subject", "needs_input"}. `needs_input` is True when nothing could be
    identified/fetched — the caller must then ASK the operator for a docs URL
    (AskUserQuestion) rather than proceed. `docs_search(query)->text` wraps a
    docs-search MCP (product docs); `web_fetch(url)->text` defaults to `http_fetch`."""
    subj = identify_subject(transcript_text, invoke)
    urls = [u for u in (list(extra_doc_urls) + subj["doc_urls"]) if u]
    have_source = bool(urls or docs_search)
    if not have_source:
        return {
            "vocab": {"terms": [], "identifiers": [], "renderings": {}, "names": []},
            "subject": subj,
            "needs_input": not subj["confident"],
        }
    vocab = build_video_vocab(
        transcript_text,
        docs_search=docs_search or (lambda q: ""),
        web_fetch=web_fetch or http_fetch,
        third_party={f"d{i}": u for i, u in enumerate(urls)},
        translate=invoke,
    )
    vocab["subject"] = subj["subject"]
    if out_path:
        save_video_vocab(out_path, vocab)
    return {"vocab": vocab, "subject": subj, "needs_input": False}


def curated_forms(glossary_path) -> list[str]:
    """All source forms (incl. aliases) of the curated glossary + term base, for `known=`."""
    try:
        from .glossary import load

        g = load(glossary_path)
        return [f for t in g.terms for f in t.all_source_forms] + list(g.identifiers)
    except Exception:
        return []


def http_fetch(url: str, timeout: int = 20) -> str:
    """Fetch a public doc URL and return visible text (tags/scripts stripped). For
    INTERNAL/authenticated docs the agent/skill should use a docs MCP instead."""
    import urllib.request

    if not url.lower().startswith(("https://", "http://")):
        raise ValueError(f"refusing to fetch non-http(s) URL: {url[:80]}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 dev-titles"})
    # scheme restricted to http(s) above; Bandit can't see that check
    with urllib.request.urlopen(req, timeout=timeout) as r:  # nosec B310
        html = r.read().decode("utf-8", "ignore")
    html = re.sub(r"(?is)<(script|style|noscript).*?</\1>", " ", html)
    return re.sub(r"\s+", " ", re.sub(r"(?s)<[^>]+>", " ", html)).strip()


def wikipedia_langlink_title(entity: str, lang_code: str, web_fetch) -> str | None:
    """The Wikipedia article title for `entity` in `lang_code`, via the English
    Wikipedia langlinks API — the AUTHORITATIVE established localized name for a brand/
    company/product (e.g. 'Northwind Airways' -> ja 'ノースウィンド航空').
    Returns None if the entity has no article or no localized version (then it stays
    Latin / gets flagged, rather than an LLM guessing a name). `web_fetch(url)->text`
    fetches the API JSON. `lang_code` is a Wikipedia code ('ja', 'zh')."""
    import json as _json
    import urllib.parse

    url = (
        "https://en.wikipedia.org/w/api.php?action=query&format=json&prop=langlinks"
        f"&titles={urllib.parse.quote(entity)}&lllang={lang_code}&lllimit=1&redirects=1"
    )
    try:
        data = _json.loads(web_fetch(url) or "{}")
        for p in data.get("query", {}).get("pages", {}).values():
            for ll in p.get("langlinks", []):
                title = ll.get("*") or ll.get("title")
                if title:
                    return title.strip()
    except Exception:
        return None
    return None


def resolve_entity_renderings(entities, *, web_fetch, invoke=None, langs=("ja", "zh-Hans")) -> dict:
    """Resolve OFFICIAL localized names for named entities (brands/companies/products)
    from an authoritative source, per video — the 'check the company's canonical
    translation' step the reviewer asked for, distinct from `propose_renderings`
    (which LLM-GUESSES). For each entity, look up its Wikipedia language-edition title
    (established usage) and pin it, with the English name as an `avoid` form so a cue
    that left it in English gets normalized. Entities with no localized article are
    OMITTED — the translator keeps them Latin and the reviewer decides (no guessing).

    zh titles are stored in a Chinese variant that may be Traditional; when `invoke`
    is given it converts the verified name to Simplified (grounded conversion, not a
    free guess). Returns {lang: {entity: {"canonical", "avoid"}}}.
    """
    wiki = {"ja": "ja", "zh-Hans": "zh", "zh": "zh"}
    out: dict[str, dict] = {}
    for ent in entities:
        for lang in langs:
            code = wiki.get(lang)
            if not code:
                continue
            title = wikipedia_langlink_title(ent, code, web_fetch)
            if not title:
                continue
            canonical = title
            if lang in ("zh-Hans", "zh") and invoke is not None:
                canonical = _to_simplified(title, invoke) or title
            out.setdefault(lang, {})[ent] = {"canonical": canonical, "avoid": [ent]}
    return out


def _to_simplified(text: str, invoke) -> str | None:
    """Convert a verified Chinese name to Simplified characters (Wikipedia stores some
    titles in Traditional). This is a grounded conversion of an authoritative string,
    not a translation — low risk. Returns None on failure (caller keeps the original)."""
    try:
        raw = (
            invoke(
                "Convert this Chinese text to Simplified Chinese characters. Output "
                "ONLY the converted text, nothing else:\n" + text
            )
            or ""
        )
        out = raw.strip().splitlines()[0].strip() if raw.strip() else ""
        return out or None
    except Exception:
        return None


def propose_renderings(
    terms: list[str], translate, *, langs=("ja", "zh-Hans"), context: str = "", skip=()
) -> dict:
    """Pin ONE canonical target rendering per key term, per language — so a
    video-specific concept ('concurrent rendering', 'strict mode') is not rendered
    a different way in every cue (the flip-flop a native reviewer reads as MT).

    `translate(prompt) -> str` is an LLM invoke (one call PER LANGUAGE, not per
    term). The model decides DNT-vs-translate itself: product/API/code names keep
    their Latin form (canonical == the term); descriptive terms get a translation.
    It also lists a few plausible WRONG renderings as `avoid`, so enforcement can
    deterministically normalize them. Returns
    {lang: {term: {"canonical": str, "avoid": [str, ...]}}}. Best-effort: a term
    that can't be parsed is skipped rather than failing the whole prep.

    GUARDRAILS (a live run pinned 'AI'->人工知能, 'Actually'->哀しみの天使, 'Sure'->もちろん):
    common English words are dropped (never a per-video term), acronyms (AI/API/LLM)
    are dropped from RENDERINGS — they stay Latin via the identifiers/DNT list, not a
    translated pin — and `skip` (e.g. speaker names) is excluded.
    """
    import json as _json

    out: dict[str, dict] = {}
    _skip = {s.lower() for s in skip}
    terms = [
        t
        for t in terms
        if not _is_common_word(t) and not _is_acronym_form(t) and t.lower() not in _skip
    ]
    if not terms:
        return out
    lang_name = {"ja": "Japanese", "zh-Hans": "Simplified Chinese"}
    for lang in langs:
        name = lang_name.get(lang, lang)
        prompt = (
            f"For each English term below, give the ONE {name} rendering to use "
            f"CONSISTENTLY in subtitles for this video, plus 1-3 alternative "
            f"renderings a translator might WRONGLY use instead (so we can normalize "
            f"them). RULES: if the term is a product name, API/component name, or code "
            f'identifier, KEEP IT IN LATIN — set "canonical" to the term unchanged '
            f'and "avoid" to []. Otherwise translate it. Output ONLY a JSON object '
            f'mapping each term to {{"canonical": "...", "avoid": ["..."]}}.\n'
            + (f"\nVideo context: {context[:600]}\n" if context else "")
            + "\nTERMS:\n"
            + "\n".join(f"- {t}" for t in terms)
        )
        try:
            raw = translate(prompt) or ""
            s, e = raw.find("{"), raw.rfind("}")
            data = _json.loads(raw[s : e + 1]) if s != -1 and e != -1 else {}
        except Exception:
            data = {}
        clean: dict[str, dict] = {}
        for t in terms:
            v = data.get(t)
            if isinstance(v, dict) and v.get("canonical"):
                av = [a for a in (v.get("avoid") or []) if a and a != v["canonical"]]
                clean[t] = {"canonical": str(v["canonical"]).strip(), "avoid": av}
        if clean:
            out[lang] = clean
    return out


def _is_latin_name(s: str) -> bool:
    return bool(s) and s.isascii() and any(c.isalpha() for c in s)


def renderings_to_terms(renderings: dict, skip_sources=(), curated_renderings=None):
    """Ephemeral glossary Terms from a per-video `renderings` map, so both prompt
    injection (retrieve) and deterministic enforce apply them.

    - `skip_sources`: source forms already in the CURATED static glossary — those
      ALWAYS win (per-video renderings are LLM-guessed and only fill gaps; a live
      run had the model pick 並行レンダリング while the glossary pins コンカレント, which
      would make enforce ping-pong). Compared case-insensitively.
    - A per-video term that CONTAINS a curated word is dropped when its rendering CONFLICTS
      with the curated one (render function→レンダー関数 vs render→レンダリング); with
      `curated_renderings` {form_lower: {lang: canonical}} a CONSISTENT longer term is kept
      (Developer Options→开发者选项 ⊃ Developer→开发者) — dropping it let the soft one-word term
      half-translate the UI label (开发者Options).
    - A per-video term that is a WORD INSIDE a curated multi-word name is dropped: the
      fragment is never its own term (e.g. Orbit→轨道 with avoid 'Orbit',
      which enforce applies target-wide → 'My 轨道 TV'). Same for ASCII avoid-forms that are
      words of a curated name.
    - DNT terms (canonical == source, no avoid) are skipped — the identifiers list
      already keeps them Latin.
    """
    from .glossary import Term

    skip = {s.lower() for s in skip_sources}
    curated = {k.lower(): v for k, v in (curated_renderings or {}).items()}
    # curated multi-char forms — a per-video term that CONTAINS one as a whole word
    # (e.g. 'render function' ⊃ 'render') can fight the curated term's rendering.
    skip_words = [re.escape(s) for s in skip if len(s) >= 4]
    skip_re = re.compile(r"\b(" + "|".join(skip_words) + r")\b", re.I) if skip_words else None
    curated_words = {w for s in skip if " " in s for w in re.findall(r"[a-z0-9][a-z0-9.+#-]*", s)}

    def _conflicts(src, lang, canon):
        for m in skip_re.finditer(src) if skip_re else ():
            if m.group(0).lower() not in curated:
                return True  # no rendering info (legacy caller) — keep the conservative drop
            cr = (curated[m.group(0).lower()] or {}).get(lang, "")
            if cr and cr not in canon:
                return True
        return False

    by_source: dict[str, Term] = {}
    for lang, m in (renderings or {}).items():
        for src, spec in m.items():
            if src.lower() in skip:
                continue  # curated glossary owns this term
            if " " not in src.strip() and src.strip().lower() in curated_words:
                continue  # a fragment of a curated multi-word name (Orbit ⊂ Orbit TV)
            canon = (spec or {}).get("canonical", "").strip()
            if canon.isascii() and _is_latin_name(src) and canon.lower() != src.strip().lower():
                continue  # a Latin name re-spelled as other Latin (VS Code -> Visual Studio Code,
                # avoid 'VS Code'): rewrites the speaker's words; Latin names stay as spoken
            if _conflicts(src, lang, canon):
                continue  # contains a curated word but renders it differently
            if " " not in src.strip() and any(
                a.strip().lower() == src.strip().lower() for a in (spec or {}).get("avoid", [])
            ):
                continue  # legacy single-word Wikipedia entity pin (Rally -> ラリー, Jordan ->
                # エマーソン): the lookup now skips single words; ignore ones already in vocab.json
            # DROP dangerous avoid-forms: a SINGLE-char CJK form is almost always a
            # substring of unrelated words, so normalizing it corrupts text (a live run
            # pinned 'Native'->ネイティブ avoid '生', turning 発生 into 発ネイティブ). Keep ASCII
            # and 2+char CJK forms — those are source-gated in enforce, so limited blast —
            # except an ASCII form that is a word of a curated name (applied target-wide).
            avoid = [
                a
                for a in (spec or {}).get("avoid", [])
                if a
                and (a.isascii() or len(a) >= 2)
                and not (a.isascii() and a.lower() in curated_words | skip)
                and a.strip().lower() not in (src.strip().lower(), canon.lower())
            ]
            if not canon or (canon == src and not avoid):
                continue  # pure DNT — nothing to pin/normalize
            term = by_source.get(src) or Term(source=src, enforce="soft")
            term.translations[lang] = canon
            if avoid:
                term.avoid.setdefault(lang, []).extend(avoid)
            by_source[src] = term
    return list(by_source.values())


def _yaml_term_block(source: str, per_lang: dict, note: str) -> str:
    """One glossary `terms:` list entry as YAML text (2-space indent). Text-append
    keeps the hand-written glossary's comments/formatting intact (a round-trip
    load/dump would strip them)."""

    def q(s: str) -> str:
        return '"' + str(s).replace("\\", "\\\\").replace('"', '\\"') + '"'

    lines = [f"  - source: {q(source)}"]
    for lang in ("zh-Hans", "ja"):
        if per_lang.get(lang, {}).get("canonical"):
            lines.append(f"    {lang}: {q(per_lang[lang]['canonical'])}")
    lines.append("    enforce: soft")
    avoids = {
        lang: per_lang[lang]["avoid"]
        for lang in ("zh-Hans", "ja")
        if per_lang.get(lang, {}).get("avoid")
    }
    if avoids:
        lines.append("    avoid:")
        for lang, av in avoids.items():
            lines.append(f"      {lang}: [{', '.join(q(a) for a in av)}]")
    lines.append(f"    notes: {q(note)}")
    return "\n".join(lines)


def promote_renderings_to_glossary(
    renderings: dict,
    glossary_path: str | Path,
    *,
    source_video: str = "",
    only_approved: bool = False,
    date: str = "",
) -> list[str]:
    """Append reviewer-approved per-video renderings into the SHARED glossary so
    every future video renders those terms the SAME way (cross-video consistency).

    This is the promotion half of the propose→approve→promote loop: vocab-prep
    proposes per video, the reviewer approves, and accepted renderings are added
    here ONCE and reused everywhere. Terms already in the glossary (by source, or
    whose source contains a curated word) are skipped — the shared decision wins,
    nothing is re-decided. `only_approved` limits promotion to renderings flagged
    {"approved": true}. Returns the list of promoted source terms.
    """
    from .glossary import load as _load

    path = Path(glossary_path)
    text = path.read_text(encoding="utf-8")
    g = _load(glossary_path)
    existing = {f.lower() for t in g.terms for f in t.all_source_forms}
    words = [re.escape(s) for s in existing if len(s) >= 4]
    word_re = re.compile(r"\b(" + "|".join(words) + r")\b", re.I) if words else None

    # group {source: {lang: {canonical, avoid}}}
    by_source: dict[str, dict] = {}
    for lang, m in (renderings or {}).items():
        for src, spec in (m or {}).items():
            if only_approved and not (spec or {}).get("approved"):
                continue
            canon = (spec or {}).get("canonical", "").strip()
            if not canon or canon == src:  # empty or pure DNT -> nothing to pin
                continue
            by_source.setdefault(src, {})[lang] = {
                "canonical": canon,
                "avoid": [a for a in (spec or {}).get("avoid", []) if a and a != canon],
            }

    blocks, promoted = [], []
    note_tail = (
        (
            f" (promoted from {source_video} vocab-prep"
            if source_video
            else " (promoted from vocab-prep"
        )
        + (f", {date}" if date else "")
        + ")"
    )
    for src, per_lang in by_source.items():
        if src.lower() in existing or (word_re and word_re.search(src)):
            continue  # shared glossary already owns it
        blocks.append(_yaml_term_block(src, per_lang, "reviewer-approved" + note_tail))
        promoted.append(src)

    if not blocks:
        return []
    addition = "\n" + "\n".join(blocks) + "\n"
    idx = text.find("\nstyle:")  # insert at the end of the terms list
    if idx == -1:
        new_text = text.rstrip() + "\n" + addition
    else:
        new_text = text[:idx] + addition + text[idx:]
    path.write_text(new_text, encoding="utf-8")
    return promoted


def save_video_vocab(path: str | Path, vocab: dict) -> None:
    Path(path).write_text(json.dumps(vocab, ensure_ascii=False, indent=2), encoding="utf-8")


def load_video_vocab(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        return {"terms": [], "identifiers": [], "renderings": {}, "names": []}
    v = json.loads(p.read_text(encoding="utf-8"))
    v.setdefault("terms", [])
    v.setdefault("identifiers", [])
    v.setdefault("renderings", {})
    v.setdefault("names", [])  # speaker / proper names (DNT; correct mishears to these)
    return v


# Self-introduction patterns ("Hi, my name is Dana Whitfield", "I'm Priya and…").
# (?i:...) makes only the intro PHRASE case-insensitive; the captured name stays
# case-sensitive ([A-Z][a-z]+) so we don't grab lowercased words.
_INTRO = re.compile(
    r"(?i:\b(?:my name is|i['’]?m|i am|this is|here['’]?s|hello,?\s+(?:i['’]?m|my name is))\s+)"
    r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})"
)


def extract_name_candidates(transcript_text: str) -> list[str]:
    """Propose speaker names from self-introductions in the transcript. ASR often
    MISHEARS the name itself (Whitfield->Whitby), so these are CANDIDATES for the
    operator to confirm/correct, not ground truth — but they seed the `names` list
    that then corrects every later mention to the confirmed spelling."""
    return _dedupe_names(m.group(1).strip() for m in _INTRO.finditer(transcript_text))


# Presenter patterns in video metadata / descriptions (more reliable than ASR:
# the text is written, not transcribed). A trigger ("Presented by", "Speakers:",
# "with") followed by a LIST of names joined by ,/&/and — captured whole, then split.
_NAME = r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}"
_META_NAME = re.compile(
    r"(?i:\b(?:presented by|speakers?:?|presenters?:?|hosted by|featuring|join|meet|by|with)\s+)"
    r"(" + _NAME + r"(?:(?:\s*(?:,|&|and)\s*)+" + _NAME + r")*)"
)  # ', and' = two seps
_NAME_SPLIT = re.compile(r"\s*(?:,|&|\band\b)\s*", re.I)

# Job-title + name ("Senior Solutions Architect Dana Whitfield", "Sr. Developer
# Evangelist Marco Bellini"): consume a run of role words, capture the name.
_ROLE = (
    r"Sr\.?|Jr\.?|Senior|Junior|Principal|Staff|Lead|Chief|Head|VP|CTO|CEO|CFO|COO|"
    r"Founder|Co-?founder|Cofounder|"
    r"Solutions?|Solution|Developer|Software|Technical|Product|Program|Engineering|"
    r"Engineer|Architect|Evangelist|Advocate|Manager|Director|Specialist|Consultant|"
    r"Designer|Scientist|Analyst|Relations|Team"
)
_ROLE_NAME = re.compile(r"(?:\b(?:" + _ROLE + r")\b\.?\s+){1,4}(" + _NAME + r")")


_ROLE_WORDS = frozenset(w.replace(r"\.?", "").replace("?", "").lower() for w in _ROLE.split("|"))


_TRIGGERS = frozenset(
    "join meet presented presenter presenters speaker speakers "
    "hosted host featuring by with and".split()
)
# Common capitalized words that leak from descriptions but aren't person names.
_NON_NAME = frozenset(
    "tools tool team teams solutions solution portal sdk api toolkit "
    "kit platform apps games app game overview guide".split()
)


def _clean_name(name: str):
    """Reduce a raw capture to the actual person name: strip leading trigger words
    ('Join', 'with') and role/title words ('Senior Solutions Architect'), so
    'Join Senior Solutions Architect Dana Whitfield' -> 'Dana Whitfield'. Returns None if
    nothing name-like remains (a title-only or junk capture like 'Tools')."""
    toks = name.split()
    while toks and (toks[0].lower() in _TRIGGERS or toks[0].strip(".").lower() in _ROLE_WORDS):
        toks = toks[1:]
    if not toks or all(t.strip(".").lower() in _ROLE_WORDS for t in toks):
        return None
    if len(toks) == 1 and toks[0].lower() in _NON_NAME:
        return None
    return " ".join(toks)


def _dedupe_names(names) -> list[str]:
    seen, out = set(), []
    for raw in names:
        name = _clean_name(raw) if raw else None
        if not name or name.split()[0] in _STOP or name.lower() in seen:
            continue
        seen.add(name.lower())
        out.append(name)
    return out


def reconcile_names(reliable: list[str], candidates: list[str]) -> list[str]:
    """Merge authoritative `reliable` names (from written metadata) with `candidates`
    (from ASR self-intros, which may be MIS-HEARD). Drop any candidate that collides
    with a reliable name — shares a name token or is a close string match — because
    it's almost certainly a mishearing of it ('Dana Whitby' vs metadata 'Dana
    Whitfield'). Keeping the mishear as a valid name defeats the correction (it saw both
    'Whitfield' and 'Whitby' as correct and left Whitby). Genuinely different speakers
    (no shared token, low similarity) are kept."""
    import difflib

    out = list(reliable)
    rel_tokens = {t.lower() for n in reliable for t in n.split()}
    for c in candidates:
        toks = {t.lower() for t in c.split()}
        collides = (toks & rel_tokens) or any(
            difflib.SequenceMatcher(None, c.lower(), r.lower()).ratio() > 0.6 for r in reliable
        )
        if not collides and c not in out:
            out.append(c)
    return out


def load_name_registry(path="glossary/names.yaml") -> list[str]:
    """Canonical person names from the channel registry (glossary/names.yaml) — hosts,
    guests, speakers harvested from video descriptions. Empty list if absent."""
    p = Path(path)
    if not p.exists():
        return []
    try:
        import yaml

        doc = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except Exception:
        return []
    return [
        e["name"].strip()
        for e in doc.get("people", [])
        if isinstance(e, dict) and (e.get("name") or "").strip()
    ]


# First names that are also common English words — never match these by first name alone
# (the verb "will" false-matched "Will Jones"; likewise Mark/Bill/Art/May/Grant/Chase...).
_COMMON_WORD_NAMES = frozenset(
    (
        "will mark bill art may grant chase drew rich guy hope rose joy dean miles jack "
        "gene barry buddy dawn faith mercy sunny hunter carol holly ray"
    ).split()
)


def _word_present(term: str, hay: str) -> bool:
    """`term` as WHOLE words in `hay` (avoids 'gio' matching inside 'giovanni')."""
    return re.search(rf"(?<![a-z0-9]){re.escape(term.lower())}(?![a-z0-9])", hay) is not None


def match_registry_names(
    text: str, registry: list[str], *, first_name_context: str = ""
) -> list[str]:
    """Canonical registry names referenced in a video, so ASR mishears get corrected to the
    KNOWN spelling instead of guessed. Whole-word matching throughout. A FULL name (>=2
    tokens) matches anywhere in `text` (transcript+description). A first name matches only in
    `first_name_context` (the reliable description, not the transcript full of 'will'/'mark'),
    only when UNIQUE to one registry person, and never for a common-English-word name. A
    SINGLE-token registry entry (Marco, Bart) matches only in the description. A bare
    first-name hit is promoted to the unique fuller canonical (Marco -> Marco Bellini)."""
    from collections import Counter

    low, ctx = text.lower(), (first_name_context or "").lower()
    fulls = [n for n in registry if len(n.split()) >= 2]
    # Uniqueness is measured among FULL names only: a bare first-name-only registry entry
    # ('Jordan') must NOT block the full canonical ('Jordan Avery') from resolving via its
    # first name — that redundant entry is what left the mangled 'Jordan Avers' uncorrected.
    full_first_counts = Counter(n.split()[0].lower() for n in fulls)
    hits: set[str] = set()
    for name in registry:
        toks = name.split()
        if len(toks) >= 2:
            if _word_present(name, low):
                hits.add(name)
                continue
            first = toks[0].lower()
            # First-name hook: surface the full canonical when its first name appears and is
            # UNIQUE among full registry names and not a common word. An uncommon first name
            # >=4 chars is reliable in the TRANSCRIPT too (not just the description), so a
            # MANGLED SURNAME ('Jordan Avers' <- registry 'Jordan Avery', which can't whole-
            # word match) still resolves to the canonical for ASR correction; shorter/ambiguous
            # first names stay description-only (avoid transcript 'will'/'mark' noise).
            if first not in _COMMON_WORD_NAMES and full_first_counts[first] == 1:
                scope = low if len(first) >= 4 else ctx
                if _word_present(first, scope):
                    hits.add(name)
        else:  # single-token name: description-only, not a common word
            if name.lower() not in _COMMON_WORD_NAMES and _word_present(name, ctx):
                hits.add(name)
    # promote a single-token hit to the unique fuller canonical sharing its first token
    promoted: set[str] = set()
    for n in hits:
        if len(n.split()) == 1:
            longer = [f for f in fulls if f.split()[0].lower() == n.lower()]
            promoted.add(longer[0] if len(longer) == 1 else n)
        else:
            promoted.add(n)
    # drop a name wholly contained (as whole words) in another retained name
    out = [n for n in promoted if not any(n != m and _word_present(n, m.lower()) for m in promoted)]
    return sorted(out)


def harvest_names(seed_text: str, metadata: str = "") -> list[str]:
    """Speaker/proper names for the per-video vocab: from written video metadata
    (description — reliable) plus self-introductions in the seed transcript
    (candidates). Metadata names come first (higher confidence)."""
    meta = [
        n.strip()
        for m in _META_NAME.finditer(metadata or "")
        for n in _NAME_SPLIT.split(m.group(1))
        if n.strip()
    ]  # split X, Y and Z
    meta += [m.group(1).strip() for m in _ROLE_NAME.finditer(metadata or "")]  # role-title + name
    # Metadata names must be a FULL name (>=2 tokens after cleaning): single capitalized
    # words from a description are almost always junk (Co, Expo, Tools, a company name),
    # not a person. Self-intros ('I'm Priya') may be a single given name.
    meta = [c for n in meta if (c := _clean_name(n)) and len(c.split()) >= 2]
    intro = [m.group(1).strip() for m in _INTRO.finditer(seed_text or "")]
    return _dedupe_names([*meta, *intro])
