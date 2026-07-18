"""Standalone re-audit of the tail-hardened certificate: fresh process, load the
persisted exact-rational functional, verify cG reproduces the recorded exact
value, and re-run the exact audit at every even J = 0..60 (62..120 were already
independently audited by the part-2 process)."""
import sys, os, json, time
REPO = "/Users/mruckman1/Desktop/dev/quantum_gravity"
sys.path.insert(0, REPO); os.chdir(REPO)
import sympy as sp
from qgse.verifiers.gravity_susy import SusyR4Verifier

ART = sys.argv[1] if len(sys.argv) > 1 else "results/positivity/artifacts/extu_tailhard_functional.json"
d = json.load(open(ART))
c = d["config"]
a = [sp.Rational(s) for s in d["a_rat"]]
V = SusyR4Verifier(f_powers=tuple(c["f_powers"]), h_powers=tuple(c["h_powers"]),
                   e_powers=tuple(c["e_powers"]), x4_powers=tuple(c["x4_powers"]),
                   x6_powers=tuple(c["x6_powers"]), k_powers=tuple(c["k_powers"]),
                   e1_powers=tuple(c["e1_powers"]), p_max=sp.Rational(c["p_max"]))
Pq = V.p_max
cG = sum(q * Pq**(n - 1) / (n - 1) for q, n in zip(a[:len(V.f_powers)], V.f_powers))
print("cG reproduces recorded exact:", str(cG) == d["c_cert_exact"], flush=True)
fails = []
t0 = time.time()
for J in range(0, 61, 2):
    ok, why = V.audit(a, J, 60)
    if not ok:
        fails.append((J, why)); print("  FAIL J=%d: %s" % (J, why), flush=True)
print("re-audit J=0..60: %s (%d spins, %.0fs)"
      % ("ALL PASS" if not fails else "FAILURES", 31, time.time() - t0), flush=True)
