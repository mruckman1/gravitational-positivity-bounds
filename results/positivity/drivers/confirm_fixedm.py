"""EXACT confirmation of the scan-level tail diagnosis for the 4d-C functional
(the reviewer's demand, and the project's own rule: a scan is not a proof; the
corner episode showed +0.6 scans over a -1.8e7 exact density).
Checks, in exact rational arithmetic (rational R0/R1/R2 at rational w, 200-dps
transcendentals):
 (1) J=204 sliver: is E genuinely negative near m2 ~ 21.0 (w ~ 0.2182)?
 (2) J=240 near m2 ~ 23.2 (w ~ 0.2075)?
 (3) fixed-m growth: E at the SAME m2=21.0 for J=204 vs 240 — does |E| grow
     ~J(J+7) (the null-column J^2 signature) or not?
"""
import sys, os, json, time
REPO = "/Users/mruckman1/Desktop/dev/quantum_gravity"
sys.path.insert(0, REPO); os.chdir(REPO)
import sympy as sp, mpmath as mp
from qgse.verifiers.gravity_susy import SusyR4Verifier, D
from qgse.verifiers.gravity_positivity import _M2
from qgse.verifiers.gravity_lp import _W

d = json.load(open("results/positivity/artifacts/extu_soft_C_functional.json"))
c = d["config"]
a = [sp.Rational(s) for s in d["a_rat"]]
V = SusyR4Verifier(**{k: tuple(v) for k, v in c.items() if k.endswith("powers")},
                   p_max=sp.Rational(c["p_max"]))
Pq = V.p_max; w = _W


def exactE(J, w0_list, label):
    t0 = time.time()
    cols = V._columns(J)
    integ = sum(a[i] * cols[i] for i in range(len(cols)))
    Eq, r0, r1, const = V._exact_smear_integral(integ, D, P=Pq)
    sub = {_M2: 1 / w**2}
    R0 = sp.cancel(sp.together(Eq.subs(sub) / const.subs(sub))) if Eq != 0 else sp.Integer(0)
    R1 = sp.cancel(sp.together((r0 * w).subs(sub) / const.subs(sub))) if r0 != 0 else sp.Integer(0)
    R2 = sp.cancel(sp.together(r1.subs(sub) / (2 * const.subs(sub)))) if r1 != 0 else sp.Integer(0)
    mp.mp.dps = 200
    Pfm = mp.mpf(Pq.p) / Pq.q
    out = []
    for w0 in w0_list:
        r0v = sp.Rational(R0.subs(w, w0)) if R0 != 0 else sp.Integer(0)
        r1v = sp.Rational(R1.subs(w, w0)) if R1 != 0 else sp.Integer(0)
        r2v = sp.Rational(R2.subs(w, w0)) if R2 != 0 else sp.Integer(0)
        x = mp.mpf(w0.p) / w0.q
        E = (mp.mpf(r0v.p) / r0v.q + (mp.mpf(r1v.p) / r1v.q) * mp.atan(Pfm * x)
             + (mp.mpf(r2v.p) / r2v.q) * mp.log(1 + (Pfm * x)**2))
        out.append((w0, E))
        print("  [%s] J=%d w=%s (m2=%.3f): exact E = %s"
              % (label, J, w0, float(1 / w0**2), mp.nstr(E, 8)), flush=True)
    print("  (kernel build+eval %.0fs)" % (time.time() - t0), flush=True)
    return out

# (1) J=204 sliver around m2=21.0: w = 1/sqrt(21) ~ 0.21822; scan +/- window
ws204 = [sp.Rational(q, 100000) for q in range(21400, 22300, 100)]
r204 = exactE(204, ws204, "J204-sliver")
# (2) J=240 around m2=23.2: w ~ 0.20756
ws240 = [sp.Rational(q, 100000) for q in range(20300, 21200, 150)]
r240 = exactE(240, ws240, "J240-sliver")
# (3) fixed-m growth at m2=21.0 exactly: w = 1/sqrt(21) irrational — use the
# closest rational from the J=204 minimum instead; compare same w0 across J
w_star = min(r204, key=lambda t: t[1])[0]
print("\nfixed-m growth check at w=%s:" % w_star, flush=True)
E204 = dict(r204)[w_star]
r240b = exactE(240, [w_star], "fixedm-J240")
E240 = r240b[0][1]
mp.mp.dps = 30
ratio = E240 / E204 if abs(E204) > 0 else mp.mpf('nan')
jj = (240 * 247) / (204 * 211)
print("  E(240)/E(204) at fixed w = %s ; J(J+7) ratio = %.4f"
      % (mp.nstr(ratio, 6), jj), flush=True)
print("done", flush=True)
