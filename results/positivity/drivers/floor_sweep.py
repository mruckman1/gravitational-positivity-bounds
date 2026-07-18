"""TIER-1 EXPERIMENT: measure the HARDENED floor geometry.

Question: is the tail-hardened extended-u floor below 3.000? Known bracket:
[2.96514, 3.0401], where 3.0401 is a ONE-SHOT certificate accepted from a
give=10% reserve band whose RAW LP optimum was never printed.

 (a) RAW hardened LP optimum (stage-1, no reserve) at p_max in
     {1, 33/32, 17/16, 9/8, 5/4} with the champion basis + margin shaping.
     The J<=40 sweep found 9/8 the unique certifiable sweet spot -- under the
     WRONG (truncated) objective. The hardened tangency structure differs.
 (b) give-sweep at the best p_max: certify with give in {0.03, 0.01, 0.003},
     j_audit=60 in-loop, persisting each certificate that lands. This measures
     how close to the raw optimum the tangency tax lets the audit certify.
"""
import sys, os, json, time
REPO = "/Users/mruckman1/Desktop/dev/quantum_gravity"
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "tasks/grav_extu"))
os.chdir(REPO)
import importlib.util
import numpy as np
import sympy as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tail_hardened_margin as thm
from tail_hardened_margin import MarginVerifier
import qgse.verifiers.gravity_susy as gs
from qgse.verifiers.gravity_susy import certify_g0_upper


def _load(p, n):
    s = importlib.util.spec_from_file_location(n, p)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m


ev = _load(os.path.join(REPO, "tasks/grav_extu/evaluate.py"), "ev_fs")
champ = _load(os.path.join(REPO, "results/grav_extu_prod/best/main.py"), "champ_fs")
cfg = champ.run_experiment()
(fp, hp, epw, xp, x6p, kp, e1p, pm0, cap, give0, mr, clipped, ja) = ev._parse(cfg)


def raw_opt(V, cap):
    """Stage-1 margin LP only: the RAW hardened optimum (no reserve band)."""
    rows, tags = V.rows_grid(40, 300, (0.25, 80.0, 240))
    nf, nh = len(V.f_powers), len(V.h_powers)
    ne = (len(V.e_powers) + len(V.x4_powers) + len(V.x6_powers)
          + len(V.k_powers) + len(V.e1_powers)
          + len(V.x8_powers) + len(V.x10_powers))
    n = nf + nh + ne
    Pf = float(V.p_max)
    gG = np.array([Pf**(p - 1) / (p - 1) for p in V.f_powers] + [0.0] * (nh + ne))
    g0 = np.array([0.0] * nf + [Pf**(i + 1) / (i + 1) for i in V.h_powers]
                  + [0.0] * ne)
    A_eq = g0.reshape(1, -1); b_eq = np.array([-1.0])
    if V.k_powers:
        off = nf + nh + len(V.e_powers) + len(V.x4_powers) + len(V.x6_powers)
        g2r = np.zeros(n); g3r = np.zeros(n)
        for j, i in enumerate(V.k_powers):
            g2r[off + j] = 2.0 * Pf**(i + 1) / (i + 1)
            g3r[off + j] = Pf**(i + 3) / (i + 3)
        A_eq = np.vstack([A_eq, g2r, g3r]); b_eq = np.append(b_eq, [0.0, 0.0])
    margins = np.zeros(len(rows))
    for i, t in enumerate(tags):
        if t[0] == "bessel":
            margins[i] = (16.0 * thm.EPS_A2 if t[1] == "asym_a2"
                          else thm.DELTA * np.abs(rows[i]).sum())
    res = gs.linprog(gG, A_ub=-rows, b_ub=-margins, A_eq=A_eq, b_eq=b_eq,
                     bounds=[(-cap, cap)] * n, method="highs")
    return (float(res.fun) if res.success else None,
            res.message if not res.success else "")


# ---- (a) p_max sweep: raw hardened optima ---------------------------------
print("=== (a) RAW hardened LP optima, champion basis + margin shaping ===",
      flush=True)
PMS = [sp.Rational(1), sp.Rational(33, 32), sp.Rational(17, 16),
       sp.Rational(9, 8), sp.Rational(5, 4)]
raw = {}
for pm in PMS:
    t0 = time.time()
    try:
        V = MarginVerifier(f_powers=fp, h_powers=hp, e_powers=epw,
                           x4_powers=xp, x6_powers=x6p, k_powers=kp,
                           e1_powers=e1p, p_max=pm)
        r, msg = raw_opt(V, cap)
    except Exception as e:
        r, msg = None, str(e)[:80]
    raw[str(pm)] = r
    print("  p_max=%-6s raw hardened opt = %s   (%.0fs)%s"
          % (pm, ("%.6f" % r) if r is not None else "INFEASIBLE",
             time.time() - t0, ("  [" + msg + "]") if msg else ""), flush=True)
json.dump(raw, open("results/positivity/artifacts/floor_sweep_raw.json", "w"),
          indent=1)

# ---- (b) give-sweep at the best certifiable candidates --------------------
# pick p_max candidates with a finite raw optimum, best-first
cands = sorted([(v, k) for k, v in raw.items() if v is not None])
print("\n=== (b) give-sweep (certify, j_audit=60 in-loop) ===", flush=True)
best = None
for v, pmk in cands[:2]:
    pm = sp.Rational(pmk)
    for give in (0.03, 0.01, 0.003):
        tag = "pm%s_give%g" % (str(pm).replace("/", "o"), give)
        art = "results/positivity/artifacts/floor_%s_functional.json" % tag
        print("  [certify] p_max=%s give=%g ..." % (pm, give), flush=True)
        t0 = time.time()
        V = MarginVerifier(f_powers=fp, h_powers=hp, e_powers=epw,
                           x4_powers=xp, x6_powers=x6p, k_powers=kp,
                           e1_powers=e1p, p_max=pm)
        try:
            r = certify_g0_upper(j_max=40, j_audit=60, max_refine=4,
                                 log=lambda m: print("      " + str(m), flush=True),
                                 verifier=V, cap=cap, give=give,
                                 save_functional=art)
            print("  [certify] p_max=%s give=%g -> c = %.9f (%.0fs)"
                  % (pm, give, r["c_cert"], time.time() - t0), flush=True)
            if best is None or r["c_cert"] < best[0]:
                best = (r["c_cert"], str(pm), give, art)
        except Exception as e:
            print("  [certify] p_max=%s give=%g -> REFUSED/FAILED: %s (%.0fs)"
                  % (pm, give, str(e)[:60], time.time() - t0), flush=True)
print("\n=== BEST CERTIFIED HARDENED VALUE THIS SWEEP ===", flush=True)
print(best, flush=True)
