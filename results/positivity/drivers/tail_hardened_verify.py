"""Stage-2 build, part 2: on the persisted TAIL-HARDENED functional,
 (a) extend the EXACT audit past J=40 (J = 42..j_hi, step 2)  -- theorems;
 (b) stable tail SCAN to J=200 (min over m of row(J,m).a)      -- measurement.
Writes results/positivity/artifacts/extu_tailhard_tail.json.
Usage: python tail_hardened_verify.py [j_hi_exact=60]
"""
import sys, os, json, time
REPO = "/Users/mruckman1/Desktop/dev/quantum_gravity"
sys.path.insert(0, REPO); os.chdir(REPO)
import numpy as np
import mpmath as mp
import sympy as sp
from qgse.verifiers.gravity_susy import SusyR4Verifier, D

J_HI = int(sys.argv[1]) if len(sys.argv) > 1 else 60
ART2 = sys.argv[2] if len(sys.argv) > 2 else \
    "results/positivity/artifacts/extu_tailhard_functional.json"
TAILOUT = ART2.replace("_functional.json", "_tail.json")
d = json.load(open(ART2)); c = d["config"]
a_rat = [sp.Rational(s) for s in d["a_rat"]]
a_f = np.array([float(q) for q in a_rat])
V = SusyR4Verifier(f_powers=tuple(c["f_powers"]), h_powers=tuple(c["h_powers"]),
                   e_powers=tuple(c["e_powers"]), x4_powers=tuple(c["x4_powers"]),
                   x6_powers=tuple(c["x6_powers"]), k_powers=tuple(c["k_powers"]),
                   e1_powers=tuple(c["e1_powers"]), p_max=sp.Rational(c["p_max"]))
print("tail-hardened functional: c_cert_exact head = %s..." % d["c_cert_exact"][:24],
      flush=True)

# ---- (b) FIRST (cheap): stable tail scan J = 42..200 ----------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fast_rows import fast_row, validate_fast
validate_fast(V, samples=((44, 2.0), (60, 20.0), (120, 60.0)),
              log=lambda m: print(m, flush=True))
pm = float(V.p_max)


def _scan_m2s(J):
    """J-adaptive: cover b >= 3 (m2 to (2J/3)^2) log-spaced + dense refill on
    the dip track b in [10, 20] (m2 ~ (2J/14)^2)."""
    hi = max(150.0, (2.0 * J / 3.0) ** 2)
    dip = (2.0 * J / 14.0) ** 2
    g = np.concatenate([np.geomspace(1.0, hi, 72),
                        np.linspace(0.4 * dip, 2.5 * dip, 32)])
    return np.unique(g[g >= 1.0])


scan = {}
print("\n[scan] min_m E_J (stable rows, J-adaptive m2 to b>=3; measurement "
      "not proof):", flush=True)
for J in list(range(42, 82, 2)) + list(range(84, 242, 6)):
    pg = np.linspace(1e-4, pm, max(240, int(2.2 * J)))
    m2s = _scan_m2s(J)
    vals = [float(fast_row(V, J, float(m2), pg) @ a_f) for m2 in m2s]
    mn = min(vals); m2min = float(m2s[int(np.argmin(vals))])
    scan[J] = {"min": mn, "at_m2": m2min, "at_b": 2.0 * J / np.sqrt(m2min)}
    tag = "NEGATIVE" if mn < -1e-9 else "ok"
    print("  J=%3d: min = %+.4e at m2=%8.2f (b=%.1f)  %s"
          % (J, mn, m2min, 2.0 * J / np.sqrt(m2min), tag), flush=True)
json.dump(scan, open(TAILOUT, "w"),
          indent=1)

# ---- (a) exact audit extension J = 42..J_HI --------------------------------
print("\n[audit] EXACT per-spin audit past the certificate scope "
      "(each is a theorem):", flush=True)
audited = {}
for J in range(62, J_HI + 1, 2):
    t0 = time.time()
    ok, why = V.audit(a_rat, J, J_HI)
    audited[J] = bool(ok)
    print("  J=%3d: %s  (%.0fs)  %s" % (J, "PASS" if ok else "FAIL",
          time.time() - t0, ("" if ok else str(why)[:70])), flush=True)
    if not ok:
        print("  (stopping extension at first FAIL -- exact scope ends at J=%d)"
              % (J - 2), flush=True)
        break
json.dump({"scan": scan, "audited_past40": audited},
          open(TAILOUT, "w"), indent=1)
print("\nwrote %s" % TAILOUT, flush=True)
