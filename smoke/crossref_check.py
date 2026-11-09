"""Crossref lookup (authors, year, venue, volume, pages, update notices) for a list of DOIs. Usage: crossref_check.py doi [doi ...]"""
import sys, requests
for doi in sys.argv[1:]:
    r = requests.get(f"https://api.crossref.org/works/{doi}", timeout=60, headers={"User-Agent": "h2su-verify (mailto:none)"})
    if r.status_code != 200:
        print(doi, "HTTP", r.status_code); continue
    m = r.json()["message"]
    au = "; ".join(f"{a.get('family','')} {a.get('given','')}" for a in m.get("author", [])[:8])
    yr = (m.get("issued", {}).get("date-parts") or [[None]])[0][0]
    upd = [(u.get("type"), u.get("label")) for u in m.get("update-to", [])] + [("relation", k) for k in m.get("relation", {}) if "retract" in k.lower()]
    print(f"{doi}\n  title: {m.get('title',[''])[0]}\n  authors: {au}\n  year: {yr}; venue: {m.get('container-title',[''])[0]}; vol {m.get('volume')} issue {m.get('issue')} pages {m.get('page') or m.get('article-number')}; type {m.get('type')}\n  update/retraction notices: {upd or 'none'}")
