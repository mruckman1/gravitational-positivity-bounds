"""Pool the j=40-PINNED fleet — the matched-scope headline comparison against
the LLM's 2.96514 (j_audit=40). Every eval here is j_audit=40 by construction,
so certified c values are directly comparable. Reports the valley mechanism:
how often the search certifies at all vs refuses/infeasible, and whether any
certified design reaches the LLM frontier."""
import json, glob, os
LLM = 2.96514
CERT = {"certified", "valid"}
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))

def cval(r):
    try: return float(r.get("c"))
    except (TypeError, ValueError): return None

seeds = {}
for f in sorted(glob.glob("ablation_v2_j40_*_s*.jsonl")):
    seeds[os.path.basename(f).replace("ablation_v2_j40_", "").replace(".jsonl", "")] = \
        [json.loads(l) for l in open(f)]

pool = [r for rows in seeds.values() for r in rows]
cert = [r for r in pool if r.get("status") in CERT and cval(r) is not None]
best = min((cval(r) for r in cert), default=None)
from collections import Counter
st = Counter(r.get("status") for r in pool)
n = len(pool)
print("=== j=40 PINNED FLEET (matched scope; LLM headline = %.5f) ===" % LLM)
print("seeds=%d  evals=%d  certified=%d (%.0f%%)" % (len(seeds), n, len(cert), 100*len(cert)/max(n,1)))
print("status breakdown:", dict(st))
print("POOLED BEST CERTIFIED @ j=40:  %s" % (("%.5f" % best) if best else "-- none certified yet"))
if best: print("   gap to LLM = %+.4f  (conventional %s LLM)" % (best-LLM, "BEATS" if best<LLM else "above"))
# valley signature: do towers appear in certified designs?
ck = [r for r in cert if (r.get("k_len") or 0) >= 2]
print("certified designs WITH C2-susy tower (k>=2): %d / %d certified  (best such: %s)"
      % (len(ck), len(cert), ("%.5f" % min(cval(r) for r in ck)) if ck else "none"))
kpop = sum(1 for r in pool if (r.get("k_len") or 0) >= 2)
print("designs that TRIED k>=2: %d / %d evals (%.0f%%)" % (kpop, n, 100*kpop/max(n,1)))
print("\n=== per seed ===")
for k in sorted(seeds):
    rows = seeds[k]; c = [r for r in rows if r.get("status") in CERT and cval(r) is not None]
    b = min((cval(r) for r in c), default=None)
    print("  %-14s evals=%3d cert=%3d best=%-9s kpop=%d ewma=%.0fs" %
          (k, len(rows), len(c), ("%.5f" % b) if b else "--",
           sum(1 for r in rows if (r.get("k_len") or 0) >= 2),
           rows[-1].get("ewma_s", 0) if rows else 0))
print("\n=== DECISION (matched scope) ===")
if best is None:
    print("  no j=40 certificate yet — search still in the valley (refusals).")
elif best <= 2.970:
    print("  !! conventional REACHES the LLM frontier at j=40 (best %.5f) -> retitle." % best)
elif best >= 2.99:
    print("  conventional STALLS above LLM at j=40 (best %.5f, gap %+.4f)." % (best, best-LLM))
else:
    print("  inconclusive (best %.5f in (2.970, 2.99))." % best)
