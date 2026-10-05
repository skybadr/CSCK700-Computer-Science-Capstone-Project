"""Look up candidate references on Crossref; save metadata for Harvard entries."""
import json, sys, time, urllib.parse, urllib.request, difflib
sys.stdout.reconfigure(encoding="utf-8")
Q = [l.strip() for l in open(sys.argv[1], encoding="utf-8") if l.strip()]
out = {}
for q in Q:
    url = ("https://api.crossref.org/works?rows=3&select=title,author,container-title,"
           "issued,page,DOI,volume,issue,type&query.bibliographic=" + urllib.parse.quote(q))
    req = urllib.request.Request(url, headers={"User-Agent": "capstone-ref-check (mailto:B.A.E.Abouabdou@liverpool.ac.uk)"})
    try:
        items = json.load(urllib.request.urlopen(req, timeout=30))["message"]["items"]
    except Exception as e:
        print("ERR", q, e); continue
    best = max(items, key=lambda it: difflib.SequenceMatcher(None, q.lower(), (it.get("title") or [""])[0].lower()).ratio())
    t = (best.get("title") or [""])[0]
    sim = difflib.SequenceMatcher(None, q.lower(), t.lower()).ratio()
    au = "; ".join(f"{a.get('family','')}, {a.get('given','')}" for a in best.get("author", []))
    yr = best.get("issued", {}).get("date-parts", [[None]])[0][0]
    ct = (best.get("container-title") or [""])[0]
    print(f"{'OK ' if sim > .9 else '?? '}{sim:.2f} | {q}\n    -> {t} | {yr} | {ct} | pp {best.get('page')} | vol {best.get('volume')} | {best.get('DOI')}\n    {au}")
    out[q] = dict(best, match=sim)
    time.sleep(0.4)
json.dump(out, open(sys.argv[2], "w", encoding="utf-8"), ensure_ascii=False, indent=1)
