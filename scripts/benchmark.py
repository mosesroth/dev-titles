#!/usr/bin/env python3
"""Quality BENCHMARK: grade machine subtitles against human-reviewed ones, OLD vs NEW, paired.

For each Video:lang, takes the OLD machine sentence records (<old-dir>/<video>/cache.jsonl) and the
NEW ones (<new-dir>/<video>/cache.jsonl), and the human-reviewed subtitles
(output/FINAL_SUBTITLES/<video>.<lang>.srt, grabbed by English sentence windows). Reports
character overlap (biased toward whichever draft the reviewer edited — never rank on it), length
and reading speed, and a paired native-viewer grading (OK / NOTICEABLE / MEANING) of the same
sampled sentences for OLD and NEW. Target for "reviewer-ready": NOTICEABLE < 2%, MEANING 0%.

  PYTHONPATH=src python3 scripts/benchmark.py --spec VideoA:ja --spec VideoA:zh-Hans \\
      --old-dir output/ticket --new-dir output/rerun_eval --out output/benchmark
"""
import json,re,random,difflib,statistics,sys
SEV_PROMPT = '''You are a native {L} speaker and professional subtitle reviewer. For each item you get the English line, a MACHINE subtitle, and the HUMAN-REVIEWED subtitle a translator produced from it. Judge the MACHINE version as if it were published as-is. It is a WHOLE SENTENCE shown across one or more subtitle cues, so do NOT mark it down for its total length (reading speed is measured separately); ignore line breaks/spacing; the human version is just a reference — differences in style or concision alone are NOT errors).
Classify:
- OK: a native viewer would find it fine; the human edit is preference/condensation.
- NOTICEABLE: understandable and correct in meaning, but a native viewer would notice it (unnatural/literal phrasing, awkward word order, wrong register, odd term, too long to read comfortably).
- MEANING: a native viewer would misunderstand, miss, or be misled about what the speaker said (mistranslation, wrong/negated meaning, garbled, key content missing, wrong name/number).
Return ONLY a JSON array: {{"id": n, "level": "OK|NOTICEABLE|MEANING", "why": "<max 18 words, English>"}}.

{body}'''
from dev_titles.cli import _read_any
from dev_titles.formats.vtt import read_vtt
from dev_titles.resegment import merge_en_to_sentences
from dev_titles import llm
inv=llm.invoke()
import argparse
ap=argparse.ArgumentParser(); ap.add_argument("--spec",action="append",required=True,help="Video:lang"); ap.add_argument("--old-dir",default="output/ticket"); ap.add_argument("--new-dir",default="output/rerun_eval"); ap.add_argument("--out",default="output/benchmark"); ap.add_argument("--n",type=int,default=40); A=ap.parse_args()
SPECS=[tuple(x.split(":")) for x in A.spec]
import os; os.makedirs(A.out,exist_ok=True)
P=SEV_PROMPT
def cache(p,lang): return {r["src"].strip():r["txt"] for r in map(json.loads,open(p)) if r.get("lang")==lang}
N=lambda s: re.sub(r"\s","",s or "")
rows=[];stats={}
random.seed(11)
for v,lang in SPECS:
    sents=merge_en_to_sentences(read_vtt(f"{A.new_dir}/{v}/en.vtt","en"))
    old=cache(f"{A.old_dir}/{v}/cache.jsonl",lang); new=cache(f"{A.new_dir}/{v}/cache.jsonl",lang)
    hum=_read_any(f"output/FINAL_SUBTITLES/{v}.{lang}.srt",lang).cues
    grab=lambda s,e: "".join(N(c.text) for c in hum if s<=(c.start+c.end)/2<e)
    S=[]
    for i,s in enumerate(sents):
        o,n,h=N(old.get(s.text.strip())),N(new.get(s.text.strip())),grab(s.start,s.end)
        if not h or not o or not n: continue
        S.append(dict(v=v,lang=lang,i=i,t=s.start,dur=max(s.end-s.start,0.5),en=s.text,old=o,new=n,human=h))
    sim=lambda a,b: difflib.SequenceMatcher(None,a,b,autojunk=False).ratio()
    st={}
    for k in ("old","new"):
        sims=[sim(x[k],x["human"]) for x in S]
        st[k]=dict(sim=statistics.mean(sims),same=sum(r==1 for r in sims)/len(S),ge80=sum(r>=.8 for r in sims)/len(S),
                   len_ratio=sum(len(x[k]) for x in S)/sum(len(x["human"]) for x in S),
                   over10=sum(len(x[k])/x["dur"]>10 for x in S)/len(S))
    st["human"]=dict(over10=sum(len(x["human"])/x["dur"]>10 for x in S)/len(S))
    st["n"]=len(S); stats[f"{v}/{lang}"]=st
    cand=[x for x in S if x["old"]!=x["human"] or x["new"]!=x["human"]]
    rows+=random.sample(cand,min(A.n,len(cand)))
    print(v,lang,json.dumps(st),flush=True)
json.dump(stats,open(f"{A.out}/stats.json","w"),indent=1)
res=[]
for k in ("old","new"):
    for lang in ("ja","zh-Hans"):
        R=[r for r in rows if r["lang"]==lang]
        for b in range(0,len(R),10):
            ch=R[b:b+10]
            body="\n\n".join(f'{j+1}. EN: {r["en"]}\n   MACHINE: {r[k]}\n   HUMAN: {r["human"]}' for j,r in enumerate(ch))
            try:
                out=inv(P.format(L="Japanese" if lang=="ja" else "Simplified Chinese",body=body)) or ""
                m=re.search(r"\[.*\]",out,re.S); js=json.loads(m.group(0)) if m else []
            except Exception: js=[]
            d={x.get("id"):x for x in js if isinstance(x,dict)}
            for j,r in enumerate(ch): res.append(dict(which=k,v=r["v"],lang=lang,i=r["i"],t=r["t"],en=r["en"],machine=r[k],human=r["human"],**{kk:d.get(j+1,{}).get(kk) for kk in ("level","why")}))
            print(k,lang,b,len(R),flush=True)
            json.dump(res,open(f"{A.out}/severity.json","w"),ensure_ascii=False,indent=1)
import collections
lines=["| lang | version | n | OK | NOTICEABLE | MEANING |","|---|---|---|---|---|---|"]
for lang in sorted({r["lang"] for r in res}):
    for k in ("old","new"):
        S=[x for x in res if x["lang"]==lang and x["which"]==k and x.get("level")]
        if not S: continue
        c=collections.Counter(x["level"] for x in S)
        lines.append(f"| {lang} | {k} | {len(S)} | {c['OK']/len(S):.0%} | {c['NOTICEABLE']/len(S):.0%} | {c['MEANING']/len(S):.0%} |")
open(f"{A.out}/summary.md","w").write("\n".join(lines)+"\n")
print("\n".join(lines))
print("DONE")
