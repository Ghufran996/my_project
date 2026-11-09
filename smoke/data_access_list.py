"""List PANGAEA BSRN monthly 'Basic measurements of radiation' datasets per station (metadata only)."""
import requests, re, json, sys
ES = "https://ws.pangaea.de/es/pangaea/panmd/_search"
stations = ["Lindenberg","Cabauw","Carpentras","Gobabeb","Tamanrasset"]
rows = []
for st in stations:
    q = f'"measurements of radiation at station {st}"'
    frm = 0
    while True:
        r = requests.get(ES, params={"q": q, "size": 200, "from": frm}, timeout=90)
        r.raise_for_status()
        h = r.json()["hits"]
        for x in h["hits"]:
            s = x["_source"]
            m = re.search(r"<md:title>(.*?)</md:title>", s.get("xml-thumb",""))
            t = m.group(1) if m else ""
            if f"measurements of radiation at station {st}" not in t: continue
            rows.append(dict(station=st, title=t, uri=s["URI"], minDT=s.get("minDateTime"), maxDT=s.get("maxDateTime"),
                             login=s.get("sp-loginOption"), lat=s.get("meanPosition",{}).get("lat"), lon=s.get("meanPosition",{}).get("lon"),
                             status=s.get("sp-dataStatus"), techkw=[k for k in s.get("techKeyword",[]) if k.startswith("license")]))
        frm += 200
        if frm >= h["total"] or frm > 5000: break
json.dump(rows, open(sys.argv[1], "w"), indent=0)
print(len(rows))
import collections
print(collections.Counter(r["station"] for r in rows))
