#!/usr/bin/env python3
"""Build glossary/en_terms.yaml: the approved English spelling of product and technology names.

The English review checks every video's script against this list and fixes case or spacing variants
("Android tv", "AndroidTV", "react native", "visual studio code") to the approved form. Sources, merged:
  - glossary.yaml: keep-Latin identifiers and terms whose rendering is the English name itself;
  - an optional term base (glossary/termbase.yaml): its English source terms;
  - the documentation names the harness harvests per video (<output>/ticket/*/vocab.json), names
    found in at least two videos;
  - glossary/partners.yaml: partner companies and their products.
Only real names are kept: a term base full of title-cased UI labels ("Marketing", "Admin", "Catalog
Ingestion") would otherwise capitalize ordinary speech. A one-word name that is also an English word
("cursor", "expo", "discord") is never enforced. The `extra`, `exclude` and `aliases` sections of an
existing en_terms.yaml are kept across rebuilds (hand curation).

  PYTHONPATH=src python3 scripts/build_en_terms.py
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "glossary" / "en_terms.yaml"
# brand words that make a multi-word phrase a name ("Android Studio"); add your own under `brands:` in
# glossary/en_terms.yaml
BRANDS = {
    "React",
    "Android",
    "Google",
    "Apple",
    "Microsoft",
    "Chrome",
    "GitHub",
    "Visual",
    "Claude",
    "Gemini",
}


def _words() -> set[str]:
    p = Path("/usr/share/dict/words")
    return (
        {w.strip().lower() for w in p.read_text(encoding="utf-8", errors="ignore").split()}
        if p.exists()
        else set()
    )


WORDS = _words()


def _brandish(tok: str) -> bool:
    return (
        tok in BRANDS
        or bool(re.search(r"[a-z][A-Z]", tok))  # DevTools, TypeScript, iOS, macOS
        or (len(tok) >= 2 and tok.isupper() and any(c.isalpha() for c in tok))  # SDK, TV, MCP
        or bool(re.search(r"\d", tok))  # 4K, S3
        or bool(re.match(r"^[A-Z][A-Za-z]*\.js$", tok))  # Next.js, Node.js
    )


# a term base / docs name ending in a generic word would capitalize ordinary speech ("Android TV apps"
# -> "Android TV Apps", "a reward URL" -> "Reward URL"): such names stay out unless curated
GENERIC_TAIL = {
    "apps",
    "app",
    "applications",
    "apis",
    "api",
    "id",
    "ids",
    "url",
    "urls",
    "program",
    "programs",
    "store",
    "calculator",
    "report",
    "reports",
    "aspects",
    "preference",
    "preferences",
    "account",
    "accounts",
    "plugin",
    "plugins",
    "modules",
    "module",
    "sandbox",
    "channel",
    "login",
    "settings",
}


# acronyms that are ordinary vocabulary: they don't make a phrase a name ("smart TV", "gen AI")
GENERIC_ACRONYMS = {
    "TV",
    "AI",
    "UI",
    "UX",
    "API",
    "APIs",
    "SDK",
    "OS",
    "ID",
    "URL",
    "IDE",
    "CLI",
    "LLM",
    "HD",
}


def is_name(term: str, curated: bool) -> bool:
    toks = term.split()
    if not curated and len(toks) > 1 and toks[-1].lower() in GENERIC_TAIL:
        return False
    if not toks or len(term) < 2 or not re.search(r"[A-Za-z]", term):
        return False
    if re.match(r"^[a-z]", term) and not re.match(
        r"^(i|mac|tv)[A-Z]", term
    ):  # code identifiers, plain words
        return False
    if len(toks) == 1:
        t = toks[0]
        if t.lower() in WORDS and not (
            curated and t.isupper()
        ):  # "Cursor", "Expo"; keep curated EMBER
            return False
        return curated or _brandish(t)
    if not all(re.match(r"^[A-Z0-9]", x) or re.match(r"^(i|mac)[A-Z]", x) for x in toks):
        return False  # "Kids reading apps", "Day Light Savings Time" style labels stay out unless all-caps words
    return curated or any(_brandish(x) and x not in GENERIC_ACRONYMS for x in toks)


def main() -> int:
    prev = yaml.safe_load(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    prev = prev or {}
    BRANDS.update(prev.get("brands") or [])
    exclude = set(prev.get("exclude") or [])
    found: dict[str, str] = {}

    def add(term, source, curated=False):
        term = " ".join(str(term).split())
        if term and term not in exclude and term not in found and is_name(term, curated):
            found[term] = source

    g = yaml.safe_load((ROOT / "glossary" / "glossary.yaml").read_text(encoding="utf-8"))
    for ident in g.get("identifiers") or []:
        add(ident, "glossary identifier", curated=True)
    for t in g.get("terms") or []:
        if t.get("ja") == t["source"] or t.get("zh-Hans") == t["source"]:
            add(t["source"], "glossary (kept Latin)", curated=True)
    tbp = ROOT / "glossary" / "termbase.yaml"
    tb = yaml.safe_load(tbp.read_text(encoding="utf-8")) if tbp.exists() else {}
    for t in (tb or {}).get("terms") or []:
        add(t["source"], "term base")
    seen: dict[str, int] = {}
    for p in (Path(os.environ.get("DEV_TITLES_OUTPUT", "output")) / "ticket").glob("*/vocab.json"):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for x in set(
            (d.get("identifiers") or []) + [y for y in d.get("terms") or [] if isinstance(y, str)]
        ):
            seen[x] = seen.get(x, 0) + 1
    for x, n in sorted(seen.items()):
        if n >= 2:
            add(x, "official docs (harvested per video)")
    pp = ROOT / "glossary" / "partners.yaml"
    if pp.exists():
        partners = (yaml.safe_load(pp.read_text(encoding="utf-8")) or {}).get("partners") or {}
        for company, d in partners.items():
            add(company, "partner", curated=True)
            for product in d.get("products") or {}:
                add(product, "partner product", curated=False)
    for x in prev.get("extra") or []:
        found[x] = "hand-added"
    doc = {
        "terms": [
            {"term": k, "source": v} for k, v in sorted(found.items(), key=lambda kv: kv[0].lower())
        ],
        "aliases": prev.get("aliases") or {},
        "brands": prev.get("brands") or [],
        "extra": prev.get("extra") or [],
        "exclude": sorted(exclude),
    }
    head = (
        "# GENERATED by scripts/build_en_terms.py — approved English spelling of product and technology names.\n"
        "# The English review fixes case/spacing variants to these. Edit `extra` (add), `exclude` (drop) and\n"
        "# `aliases` (variant -> approved) by hand; they survive a rebuild.\n"
    )
    OUT.write_text(
        head + yaml.safe_dump(doc, allow_unicode=True, sort_keys=False, width=110), encoding="utf-8"
    )
    by = {}
    for v in found.values():
        by[v] = by.get(v, 0) + 1
    print(f"{len(found)} approved names -> {OUT}  {by}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
