"""Pool the free-scope ablation_v2 fleet logs (secondary, narrow-scope result);
apply the pre-registered decision rule."""
import json, glob, os
LLM_BEST = 2.96514
CERT = {"certified", "valid"}
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "freej_logs"))

seeds = {}
for f in sorted(glob.glob("ablation_v2_*_s*.jsonl")):
    if "_s99." in f:      # skip the smoke test
        continue
    key = os.path.basename(f).replace("ablation_v2_", "").replace(".jsonl", "")
    rows = [json.loads(l) for l in open(f)]
    seeds[key] = rows

def cval(r):
    c = r.get("c")
    try:
        return float(c)
    except (TypeError, ValueError):
        return None

pool = [r for rows in seeds.values() for r in rows]
pool_cert = [r for r in pool if r.get("status") in CERT and cval(r) is not None]
pool_best = min((cval(r) for r in pool_cert), default=None)
n_tot = len(pool)
n_cert = len(pool_cert)
n_to = sum(1 for r in pool if r.get("timed_out"))
n_kpop = sum(1 for r in pool if (r.get("k_len") or 0) >= 2)
n_champlike = sum(1 for r in pool if (r.get("k_len") or 0) >= 2 and (r.get("e1_len") or 0) >= 3)

print("=== POOLED (%d seeds) ===" % len(seeds))
print("evals=%d  certified=%d (%.0f%%)  timeouts=%d (%.0f%%)"
      % (n_tot, n_cert, 100*n_cert/max(n_tot,1), n_to, 100*n_to/max(n_tot,1)))
print("k_powers populated (>=2): %d (%.0f%%)   champion-like (k>=2 & e1>=3): %d (%.0f%%)"
      % (n_kpop, 100*n_kpop/max(n_tot,1), n_champlike, 100*n_champlike/max(n_tot,1)))
print("POOLED BEST CERTIFIED c = %s   (LLM: %.5f)" % (pool_best, LLM_BEST))
if pool_best is not None:
    print("   gap to LLM = %+.4f" % (pool_best - LLM_BEST))

# best certified achieved BY a champion-like design?
champ_cert = [cval(r) for r in pool_cert if (r.get("k_len") or 0) >= 2]
print("best certified among k-populated designs: %s"
      % (min(champ_cert) if champ_cert else "none certified with k>=2 yet"))

print("\n=== PER SEED ===")
for key in sorted(seeds):
    rows = seeds[key]
    cert = [r for r in rows if r.get("status") in CERT and cval(r) is not None]
    best = min((cval(r) for r in cert), default=None)
    to = sum(1 for r in rows if r.get("timed_out"))
    kp = sum(1 for r in rows if (r.get("k_len") or 0) >= 2)
    ew = rows[-1].get("ewma_s") if rows else 0
    print("  %-12s evals=%3d cert=%3d best=%-9s timeouts=%d kpop=%d ewma=%.0fs"
          % (key, len(rows), len(cert), ("%.5f" % best) if best else "--",
             to, kp, ew or 0))

# decision-rule flags
print("\n=== DECISION FLAGS ===")
sub = [ (k, cval(r)) for k,rows in seeds.items() for r in rows
        if r.get("status") in CERT and cval(r) is not None and cval(r) <= 2.969 ]
if sub:
    print("!! RETITLE TRIGGER: conventional seed certified <= 2.969:", sub[:5])
elif pool_best is not None and pool_best <= 2.970:
    print("!! RETITLE TRIGGER: pooled best <= 2.970")
elif pool_best is not None and pool_best >= 2.99:
    print("   Consistent with 'Machine-Guided' (pooled best >= 2.99); needs "
          "budget>=2x462 & each seed>=120 evals & k explored to confirm.")
else:
    print("   Inconclusive so far (pooled best in (2.970, 2.99)).")
