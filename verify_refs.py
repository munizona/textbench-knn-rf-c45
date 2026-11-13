"""Check every DOI in references.json against Crossref (run on a machine with internet access).
   python verify_refs.py references.json
Prints title/year/first author returned by Crossref so you can compare with the manuscript."""
import json, sys, urllib.request, urllib.parse

refs = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "references.json", encoding="utf8"))
for r in refs:
    doi = r.get("doi")
    if not doi:
        print(f"[no DOI] {r['id']}: {r['title'][:70]} -> {r.get('url')}"); continue
    try:
        with urllib.request.urlopen("https://api.crossref.org/works/" + urllib.parse.quote(doi), timeout=30) as f:
            m = json.load(f)["message"]
        yr = (m.get("issued", {}).get("date-parts") or [[None]])[0][0]
        au = (m.get("author") or [{}])[0].get("family")
        print(f"[ok] {r['id']}: {m['title'][0][:70]} | {yr} | {au}")
    except Exception as e:
        print(f"[FAIL] {r['id']}: {doi} ({e})")
